import numpy as np
import os

# Đường dẫn
CT_PATH = './crop_image'
DRR_PARAMS_PATH = './DRR_Parameters/'
os.makedirs(DRR_PARAMS_PATH, exist_ok=True)

# Load tất cả file CT
ct_files = [os.path.join(CT_PATH, f) for f in sorted(os.listdir(CT_PATH)) if f.endswith('.npy')]
if not ct_files:
    raise FileNotFoundError(f"No .npy files found in {CT_PATH}. Run simulate_DRR.py first.")

ct_data = [np.load(f) for f in ct_files]  # Mỗi file shape (128, 128, 128)
ct_stack = np.stack(ct_data, axis=0)      # Shape: (N, 128, 128, 128)

# Tính mean
mean_ct = np.mean(ct_stack, axis=0)       # Shape: (128, 128, 128)

# Lưu file
np.save(os.path.join(DRR_PARAMS_PATH, 'mean_CT.npy'), mean_ct)
print("Created mean_CT.npy")
print(f"Shape: {mean_ct.shape}")