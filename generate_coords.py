import numpy as np
import os

# Đường dẫn lưu file
DRR_PARAMS_PATH = './DRR_Parameters/'
os.makedirs(DRR_PARAMS_PATH, exist_ok=True)

# Kích thước volume và spacing
H, W, D = 128, 128, 128
spacing = np.array([2.5, 2.5, 2.5])

# Tạo lưới tọa độ 3D
x = np.arange(0, H) * spacing[0]  # [0, 2.5, 5.0, ..., 317.5]
y = np.arange(0, W) * spacing[1]
z = np.arange(0, D) * spacing[2]
coords_3d = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)  # Shape: (128, 128, 128, 3)

# Lưu file
np.save(os.path.join(DRR_PARAMS_PATH, 'coords_3D.npy'), coords_3d)
print("Created coords_3D.npy")
print(f"Shape: {coords_3d.shape}")