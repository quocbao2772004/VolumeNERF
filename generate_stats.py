import numpy as np
import os

# Thư mục chứa DRR
XRAY_PATH = '/home/anonymous/code/AI/VolumeNeRF/drr_Xray'  # Đường dẫn từ configs.py
DRR_PARAMS_PATH = './DRR_Parameters/'

# Load tất cả file DRR
drr_files = [os.path.join(XRAY_PATH, f) for f in sorted(os.listdir(XRAY_PATH)) if f.endswith('.npy')]
drr_data = [np.load(f) for f in drr_files]

# Stack thành tensor (N, 128, 128), N là số file
drr_stack = np.stack(drr_data, axis=0)

# Tính mean và std theo từng pixel
mean_xray = np.mean(drr_stack, axis=0)  # Shape: (128, 128)
std_xray = np.std(drr_stack, axis=0)    # Shape: (128, 128)

# Lưu file
np.save(os.path.join(DRR_PARAMS_PATH, 'mean_xray.npy'), mean_xray)
np.save(os.path.join(DRR_PARAMS_PATH, 'std_xray.npy'), std_xray)
print("Created mean_xray.npy and std_xray.npy")
print(f"Mean shape: {mean_xray.shape}, Std shape: {std_xray.shape}")