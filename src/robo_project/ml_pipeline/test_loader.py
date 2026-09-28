import os

from dataset_loader import HabitatDataset

dataset_path = os.path.expanduser("~/robo_project_ws/src/dataset")

dataset = HabitatDataset(dataset_path)

print("Dataset size:", len(dataset))

image, target = dataset[0]

print("Image shape:", image.shape)

print("Target:", target)
