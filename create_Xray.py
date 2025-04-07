import os
import numpy as np
from PIL import Image

# Đường dẫn
INPUT_PATH = '/home/anonymous/code/AI/VolumeNeRF/images2D'  # Đường dẫn đến thư mục chứa .jpg của mày
OUTPUT_PATH = '/home/anonymous/code/AI/VolumeNeRF/synthentic_Xray'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Hàm chuẩn hóa ảnh grayscale
def normalize_gray(image):
    # Đảm bảo là numpy array
    img = np.array(image, dtype=np.float32)
    # Chuẩn hóa về [0, 1]
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Thêm epsilon tránh chia 0
    # Thêm chiều kênh để thành (1, 128, 128)
    img = np.expand_dims(img, axis=0)  # Từ (128, 128) thành (1, 128, 128)
    return img

# Load và chuyển tất cả file .jpg
jpg_files = sorted([f for f in os.listdir(INPUT_PATH) if f.endswith('.jpg')])
if not jpg_files:
    raise FileNotFoundError(f"No .jpg files found in {INPUT_PATH}")

for i, jpg_file in enumerate(jpg_files):
    # Load ảnh
    img_path = os.path.join(INPUT_PATH, jpg_file)
    img = Image.open(img_path)  # Load bằng PIL
    
    # Kiểm tra shape
    img_array = np.array(img)
    if img_array.shape != (128, 128):
        raise ValueError(f"Image {jpg_file} has shape {img_array.shape}, expected (128, 128)")
    if len(img_array.shape) > 2:
        raise ValueError(f"Image {jpg_file} is not grayscale, shape: {img_array.shape}")

    # Chuẩn hóa và thêm chiều kênh
    gray_img = normalize_gray(img_array)  # Shape: (1, 128, 128), giá trị [0, 1]
    
    # Tạo tên file .npy
    npy_name = f"{jpg_file[:-4]}.npy"  # Giữ tên gốc, bỏ .jpg
    npy_path = os.path.join(OUTPUT_PATH, npy_name)
    
    # Lưu thành .npy
    np.save(npy_path, gray_img)
    print(f"Saved {npy_name} with shape {gray_img.shape}, min: {gray_img.min():.2f}, max: {gray_img.max():.2f}")

print(f"Đã chuyển {len(jpg_files)} file từ .jpg sang .npy")