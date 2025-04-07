import os
import numpy as np
from shutil import move

# Đường dẫn gốc
CT_PATH = './crop_image_intersection/'
XRAY_PATH = '././synthentic_Xray_intersection/'

# Đường dẫn subfolder
CT_TRAIN = './crop_image_intersection/train/'
CT_TEST = './crop_image_intersection/test/'
CT_VAL = './crop_image_intersection/val/'
XRAY_TRAIN = '././synthentic_Xray_intersection/train/'
XRAY_TEST = '././synthentic_Xray_intersection/test/'
XRAY_VAL = '././synthentic_Xray_intersection/val/'

# Tạo thư mục
for path in [CT_TRAIN, CT_TEST, CT_VAL, XRAY_TRAIN, XRAY_TEST, XRAY_VAL]:
    os.makedirs(path, exist_ok=True)

# Lấy danh sách file
ct_files = sorted([f for f in os.listdir(CT_PATH) if f.endswith('.npy')])
xray_files = sorted([f for f in os.listdir(XRAY_PATH) if f.endswith('.npy')])

# Kiểm tra số lượng file khớp nhau
assert len(ct_files) == len(xray_files), "Số file CT và X-ray không khớp!"

# Số lượng file
n_total = len(ct_files)
n_train = int(0.8 * n_total)  # 80% train
n_test = int(0.1 * n_total)   # 10% test
n_val = n_total - n_train - n_test  # Còn lại cho val

# Shuffle index
indices = np.random.permutation(n_total)

# Chia tập
train_idx = indices[:n_train]
test_idx = indices[n_train:n_train + n_test]
val_idx = indices[n_train + n_test:]

# Di chuyển file
for idx in train_idx:
    move(os.path.join(CT_PATH, ct_files[idx]), os.path.join(CT_TRAIN, ct_files[idx]))
    move(os.path.join(XRAY_PATH, xray_files[idx]), os.path.join(XRAY_TRAIN, xray_files[idx]))

for idx in test_idx:
    move(os.path.join(CT_PATH, ct_files[idx]), os.path.join(CT_TEST, ct_files[idx]))
    move(os.path.join(XRAY_PATH, xray_files[idx]), os.path.join(XRAY_TEST, xray_files[idx]))

for idx in val_idx:
    move(os.path.join(CT_PATH, ct_files[idx]), os.path.join(CT_VAL, ct_files[idx]))
    move(os.path.join(XRAY_PATH, xray_files[idx]), os.path.join(XRAY_VAL, xray_files[idx]))

print(f"Đã chia dữ liệu:")
print(f"- Train: {n_train} files")
print(f"- Test: {n_test} files")
print(f"- Val: {n_val} files")