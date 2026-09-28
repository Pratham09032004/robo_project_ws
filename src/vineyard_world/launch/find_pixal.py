from pathlib import Path

from PIL import Image
import numpy as np

PKG_DIR = Path(__file__).resolve().parents[1]  # src/vineyard_world


image = Image.open(PKG_DIR / 'materials' / 'png' / 'sommerach_vineyard_smooth_heightmap_512.png').convert('L')
image_array = np.array(image)


min_pixel = image_array.min()
max_pixel = image_array.max()

print(f"Minimum pixel value: {min_pixel}")
print(f"Maximum pixel value: {max_pixel}")
