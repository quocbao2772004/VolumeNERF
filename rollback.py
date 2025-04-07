import os
from shutil import move


CT_PATH = './crop_image/'
XRAY_PATH = './drr_Xray/'

CT_SUBFOLDERS = ['./crop_image/train/', './crop_image/test/', './crop_image/val/']
XRAY_SUBFOLDERS = ['./drr_Xray/train/', './drr_Xray/test/', './drr_Xray/val/']

for subfolder in CT_SUBFOLDERS:
    if os.path.exists(subfolder):
        ct_files = [f for f in os.listdir(subfolder) if f.endswith('.npy')]
        for f in ct_files:
            move(os.path.join(subfolder, f), os.path.join(CT_PATH, f))
        print(f"Moved {len(ct_files)} files from {subfolder} to {CT_PATH}")

for subfolder in XRAY_SUBFOLDERS:
    if os.path.exists(subfolder):
        xray_files = [f for f in os.listdir(subfolder) if f.endswith('.npy')]
        for f in xray_files:
            move(os.path.join(subfolder, f), os.path.join(XRAY_PATH, f))
        print(f"Moved {len(xray_files)} files from {subfolder} to {XRAY_PATH}")

for subfolder in CT_SUBFOLDERS + XRAY_SUBFOLDERS:
    if os.path.exists(subfolder) and not os.listdir(subfolder): 
        os.rmdir(subfolder)
        print(f"Removed empty folder: {subfolder}")

print("Đã chuyển tất cả file về thư mục gốc!")