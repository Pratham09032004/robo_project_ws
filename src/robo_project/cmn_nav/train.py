#!/usr/bin/env python3
"""Train the local occupancy predictor (paper Section IV-B).

Paper settings: Adam, lr 1e-4, weight decay 1e-7, pixels scaled to [0, 1],
binary cross-entropy.  Extra here: ImageNet normalisation, mild colour
jitter, and a mirror augmentation (flip every view, swap left/right, flip the
label left-right), which is exact for a 4-view panorama.

    python3 src/robo_project/cmn_nav/train.py --data data/imr_lab8.npz data/denmark.npz \
        --out src/robo_project/ml_runs/cmn_local_occ.pth
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_grid import LOCAL_RES, LOCAL_SIZE                          # noqa: E402
from local_occ_model import PanoLocalOccNet, images_to_tensor        # noqa: E402

MIRROR_VIEW_ORDER = [0, 3, 2, 1]   # front, left, back, right -> front, right, back, left


def load(paths):
    imgs, occ, scene = [], [], []
    for p in paths:
        d = np.load(p)
        imgs.append(d["images"]); occ.append(d["occ"])
        scene += [str(d["scene"])] * len(d["occ"])
        views = [str(v) for v in d["views"]]
    return np.concatenate(imgs), np.concatenate(occ), np.array(scene), views


def batch_tensors(imgs, occ, idx, augment, rng):
    x = imgs[idx]
    y = occ[idx].astype(np.float32)
    if augment:
        flip = rng.random(len(idx)) < 0.5
        x[flip] = x[flip][:, MIRROR_VIEW_ORDER, :, ::-1]
        y[flip] = y[flip][:, :, ::-1]
    xt = images_to_tensor(x)
    if augment:   # brightness / contrast jitter per sample
        b = torch.empty(len(idx), 1, 1, 1, 1).uniform_(-0.3, 0.3)
        c = torch.empty(len(idx), 1, 1, 1, 1).uniform_(0.8, 1.2)
        xt = xt * c + b
    return xt, torch.from_numpy(np.ascontiguousarray(y))


@torch.no_grad()
def evaluate(model, imgs, occ, idx, bs=128):
    model.eval()
    loss = inter = union = correct = total = 0.0
    for s in range(0, len(idx), bs):
        x, y = batch_tensors(imgs, occ, idx[s:s + bs], False, None)
        logits = model(x)
        loss += F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
        pred = (logits > 0).float()
        inter += (pred * y).sum().item(); union += ((pred + y) > 0).float().sum().item()
        correct += (pred == y).float().sum().item(); total += y.numel()
    return {"bce": loss / total, "acc": correct / total, "occ_iou": inter / max(union, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--init", help="checkpoint to continue training from")
    args = ap.parse_args()

    torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    imgs, occ, scene, views = load(args.data)
    n = len(occ)
    perm = rng.permutation(n)
    n_val = n_test = n // 10
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    print(f"{n} samples ({dict(zip(*np.unique(scene, return_counts=True)))}): "
          f"train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}", flush=True)

    model_args = dict(num_views=imgs.shape[1], img_hw=imgs.shape[2:4], out_size=occ.shape[1])
    model = PanoLocalOccNet(**model_args)
    if args.init:
        model.load_state_dict(torch.load(args.init, weights_only=False)["state_dict"])
        print(f"initialised from {args.init}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)

    best = None
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0
        order = rng.permutation(train_idx)
        for s in range(0, len(order), args.batch):
            x, y = batch_tensors(imgs, occ, order[s:s + args.batch], True, rng)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * len(x)
        val = evaluate(model, imgs, occ, val_idx)
        print(f"epoch {epoch:3d}  train_bce {run / len(order):.4f}  val_bce {val['bce']:.4f}  "
              f"val_acc {val['acc']:.3f}  val_occ_iou {val['occ_iou']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if best is None or val["bce"] < best:
            best = val["bce"]
            torch.save({"state_dict": model.state_dict(), "model_args": model_args, "views": views,
                        "local_size": LOCAL_SIZE, "local_res": LOCAL_RES, "epoch": epoch, "val": val,
                        "data": [os.path.basename(p) for p in args.data]}, args.out)

    ckpt = torch.load(args.out, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, imgs, occ, test_idx)
    for sc in np.unique(scene):
        sub = test_idx[scene[test_idx] == sc]
        print(f"test [{sc}] ({len(sub)}): {evaluate(model, imgs, occ, sub)}")
    ckpt["test"] = test
    torch.save(ckpt, args.out)
    print(f"best epoch {ckpt['epoch']}  test {test}  -> {args.out}")


if __name__ == "__main__":
    main()
