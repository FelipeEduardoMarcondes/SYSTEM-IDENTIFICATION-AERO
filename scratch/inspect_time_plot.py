import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('data/experimentos_novos/EXP001_legado_narx-mpc/2_mpc_python/MPC_Computational_Time.png')
H, W, _ = img.shape
print(f"Image shape: H={H}, W={W}")

# Find black axes lines (values close to 0)
# Let's find horizontal line at the bottom and vertical line at the left
gray = img[:, :, :3].mean(axis=2)
# Pixels that are very dark (axes box or ticks)
dark = gray < 0.2

# Find the bounding box of the grid/axes
# The axes box typically has dark borders
# Find where the blue line is
r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
blue_mask = (b > 0.6) & (r < 0.3) & (g < 0.6)
y_idx, x_idx = np.where(blue_mask)
print(f"Blue line x range: [{x_idx.min()}, {x_idx.max()}], y range: [{y_idx.min()}, {y_idx.max()}]")

# Let's inspect x values: time goes from 0 to 25s (from original code tvec_sim has 2500 samples * 0.01 = 25s)
# y goes from 0 to ~0.5s
# Let's find the axes box corners
# Look for the left spine (vertical line of dark pixels near x_idx.min())
for x in range(x_idx.min() - 5, x_idx.min() + 5):
    col = dark[:, x]
    if col.sum() > 50:
        print(f"Spine found at x={x}, y bounds: [{np.where(col)[0].min()}, {np.where(col)[0].max()}]")
