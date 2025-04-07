import os
from shutil import copy

# Đường dẫn
SYNTHETIC_XRAY_PATH = '/home/anonymous/code/AI/VolumeNeRF/synthentic_Xray/'
CROP_IMAGE_PATH = '/home/anonymous/code/AI/VolumeNeRF/crop_image/'
INTERSECTION_PATH_XRAY = '/home/anonymous/code/AI/VolumeNeRF/synthentic_Xray_intersection/'
INTERSECTION_PATH_CT = '/home/anonymous/code/AI/VolumeNeRF/crop_image_intersection/'
os.makedirs(INTERSECTION_PATH_XRAY, exist_ok=True)
os.makedirs(INTERSECTION_PATH_CT, exist_ok=True)

# Lấy danh sách file
synthetic_files = sorted([f for f in os.listdir(SYNTHETIC_XRAY_PATH) if f.endswith('.npy')])
crop_files = sorted([f for f in os.listdir(CROP_IMAGE_PATH) if f.endswith('.npy')])

# Tìm tập giao dựa trên tên file
synthetic_set = set(synthetic_files)
crop_set = set(crop_files)
common_files = synthetic_set.intersection(crop_set)  # Tập giao

# Chuyển file chung từ synthetic_Xray
for file_name in common_files:
    src_synthetic = os.path.join(SYNTHETIC_XRAY_PATH, file_name)
    dst_synthetic = os.path.join(INTERSECTION_PATH_XRAY, f"{file_name}")
    copy(src_synthetic, dst_synthetic)
    print(f"Copied X-ray: {file_name} to {dst_synthetic}")

# Chuyển file chung từ crop_image
for file_name in common_files:
    src_crop = os.path.join(CROP_IMAGE_PATH, file_name)
    dst_crop = os.path.join(INTERSECTION_PATH_CT, f"{file_name}")
    copy(src_crop, dst_crop)
    print(f"Copied CT: {file_name} to {dst_crop}")

print(f"Đã chuyển {len(common_files)} file chung vào {INTERSECTION_PATH_CT} và {INTERSECTION_PATH_XRAY}")
print(f"- Synthetic X-ray files: {len(synthetic_files)}")
print(f"- Crop image files: {len(crop_files)}")
print(f"- Common files: {len(common_files)}")