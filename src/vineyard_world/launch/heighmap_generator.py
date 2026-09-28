import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Output path: src/vineyard_world/materials/png
output_dir = str(Path(__file__).resolve().parents[1] / "materials" / "png")
os.makedirs(output_dir, exist_ok=True)

# Image dimensions
width = 512
height = 512

# Arc parameters
x0, y0 = 0, 10     # Start point of the arc
x1, y1 = 50, 0     # End point
xc, yc = 50, 130   # Arc center

# Calculate arc angle range
theta0 = np.arctan2(y0 - yc, x0 - xc)
theta1 = np.arctan2(y1 - yc, x1 - xc)
theta = np.linspace(theta0, theta1, width)

# Radius of the arc
r = np.sqrt((x0 - xc)**2 + (y0 - yc)**2)

# Get y-values along the arc
x_arc = xc + r * np.cos(theta)
y_arc = yc + r * np.sin(theta)

# Normalize y values to grayscale (0–255)
z_arc = (y_arc - y_arc.min()) / (y_arc.max() - y_arc.min()) * 255

# Create base 2D heightmap
heightmap = np.tile(z_arc, (height, 1))

# Generate smoothed random surface noise
np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=10.0, size=(height, width))  # moderate noise
smoothed_noise = gaussian_filter(noise, sigma=3)

# Add noise to heightmap
heightmap += smoothed_noise
heightmap = np.clip(heightmap, 0, 255).astype(np.uint8)

# Save the image
filename = "vineyard_arc_slope_heightmap_bumpy_smooth_512.png"
img = Image.fromarray(heightmap)
img.save(os.path.join(output_dir, filename))

#Show preview
plt.imshow(heightmap, cmap='gray')
plt.title("vineyard_arc_slope_heightmap_bumpy_smooth_512 (512x512)")
plt.axis('off')
plt.show()
