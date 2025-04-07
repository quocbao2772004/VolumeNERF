import numpy as np
import os


CT_TRAIN = './crop_image/train/'
XRAY_TRAIN = './drr_Xray/train/'
DRR_PARAMS_PATH = './DRR_Parameters/'
os.makedirs(DRR_PARAMS_PATH, exist_ok=True)

ct_files = [os.path.join(CT_TRAIN, f) for f in sorted(os.listdir(CT_TRAIN)) if f.endswith('.npy')]
if not ct_files:
    raise FileNotFoundError(f"No .npy files found in {CT_TRAIN}. Run simulate_DRR.py and split_data.py first.")
ct_data = [np.load(f) for f in ct_files]
ct_stack = np.stack(ct_data, axis=0)
mean_ct = np.mean(ct_stack, axis=0)
np.save(os.path.join(DRR_PARAMS_PATH, 'mean_CT.npy'), mean_ct)
print("Created mean_CT.npy")
print(f"Shape: {mean_ct.shape}")

xray_files = [os.path.join(XRAY_TRAIN, f) for f in sorted(os.listdir(XRAY_TRAIN)) if f.endswith('.npy')]
if not xray_files:
    raise FileNotFoundError(f"No .npy files found in {XRAY_TRAIN}. Run simulate_DRR.py and split_data.py first.")
xray_data = [np.squeeze(np.load(f)) for f in xray_files]
xray_stack = np.stack(xray_data, axis=0)
mean_xray = np.mean(xray_stack, axis=0)
std_xray = np.std(xray_stack, axis=0)
np.save(os.path.join(DRR_PARAMS_PATH, 'mean_xray.npy'), mean_xray)
np.save(os.path.join(DRR_PARAMS_PATH, 'std_xray.npy'), std_xray)
print("Created mean_xray.npy and std_xray.npy")
print(f"Mean X-ray shape: {mean_xray.shape}, Std X-ray shape: {std_xray.shape}")