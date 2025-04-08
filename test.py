import os
import numpy as np
import torch
import cv2
from generator import renderers
import configs as configs
from tqdm import tqdm

def negative_likelihood(image, mean, std):
    return torch.log(std) + (image - mean) ** 2 / (2 * (std ** 2))

def normalize(photo):
    # Điều chỉnh contrast trước khi normalize
    photo = np.clip(photo, np.percentile(photo, 5), np.percentile(photo, 95))  # Tăng contrast
    photo = (photo - np.min(photo)) * 255 / (np.max(photo) - np.min(photo))
    return photo.astype(np.uint8)

def test_model(checkpoint_path, test_dir, output_dir, mean_xray_path, std_xray_path, config_name='X2CT', device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU.")

    # Load config
    config_module = getattr(configs, config_name)
    if callable(config_module):
        config = config_module()
    else:
        config = config_module

    # Load generator
    generator = getattr(renderers, config['generator']['class'])(**config['generator']['kwargs'])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.to(device)
    generator.eval()

    # Load mean_xray và std_xray để tính prob_XRAY
    mean_xray = np.load(mean_xray_path).astype(np.float32)
    mean_xray = torch.from_numpy(mean_xray).to(device)
    std_xray = np.load(std_xray_path).astype(np.float32)
    std_xray = torch.from_numpy(std_xray).to(device)

    # Lấy danh sách file trong test_dir
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.npy')])
    if not test_files:
        raise FileNotFoundError(f"No .npy files found in {test_dir}")

    # Test loop
    for test_file in tqdm(test_files, desc="Testing"):
        test_file_path = os.path.join(test_dir, test_file)

        # Load và chuẩn hóa X-ray giống snapshot
        xray = np.load(test_file_path).astype(np.float32)  # Shape: (1, 128, 128)
        xray = (xray - np.min(xray)) / (np.max(xray) - np.min(xray))  # Scale về [0, 1]
        xray = torch.from_numpy(xray).to(device)

        # Tính prob_XRAY giống snapshot
        prob_xray = negative_likelihood(xray, mean_xray, std_xray)

        # Thêm batch dimension
        input_xray = torch.unsqueeze(xray, dim=0)
        prob_xray = torch.unsqueeze(prob_xray, dim=0)

        # Inference
        with torch.no_grad():
            volume, drr, _ = generator(input_xray, prob_xray)

        # Chuyển về CPU và numpy
        volume = volume[0, 0].cpu().numpy()  # Shape: (128, 128, 128)
        drr = drr[0, 0].cpu().numpy()        # Shape: (128, 128)

        # Debug giá trị thô
        print(f"\nFile: {test_file}")
        print(f"X-ray min: {xray.min().item()}, max: {xray.max().item()}")
        print(f"prob_XRAY min: {prob_xray.min().item()}, max: {prob_xray.max().item()}")
        print(f"DRR (raw) min: {drr.min()}, max: {drr.max()}")
        print(f"Volume (raw) min: {volume.min()}, max: {volume.max()}")

        # Lưu kết quả giống snapshot
        base_name = os.path.splitext(test_file)[0]
        output_subdir = os.path.join(output_dir, base_name)
        os.makedirs(output_subdir, exist_ok=True)

        np.save(os.path.join(output_subdir, f"{base_name}_volume.npy"), volume)
        cv2.imwrite(os.path.join(output_subdir, "drr.png"), normalize(drr))

        # Tạo ảnh cho cả 3 mặt cắt
        slices = range(0, 128, 10)  # Lấy slice cách nhau 10

        # Mặt cắt ngang (axial, trục Z)
        for i in slices:
            slice_img = volume[i, :, :]  # Shape: (128, 128)
            cv2.imwrite(os.path.join(output_subdir, f"CT_axial_{i}.png"), normalize(slice_img))

        # Mặt cắt đứng (coronal, trục Y)
        for j in slices:
            slice_img = volume[:, j, :]  # Shape: (128, 128)
            cv2.imwrite(os.path.join(output_subdir, f"CT_coronal_{j}.png"), normalize(slice_img))

        # Mặt cắt dọc (sagittal, trục X)
        for k in slices:
            slice_img = volume[:, :, k]  # Shape: (128, 128)
            cv2.imwrite(os.path.join(output_subdir, f"CT_sagittal_{k}.png"), normalize(slice_img))

        torch.cuda.empty_cache()

    print(f"Test completed! Results saved in {output_dir}")

if __name__ == "__main__":
    checkpoint_path = "/teamspace/studios/this_studio/VolumeNERF/results/Monday_07_April_2025_11h_51m_21s/step009600_generator.pth"
    test_dir = "/teamspace/studios/this_studio/VolumeNERF/synthentic_Xray_intersection/test"  # Chạy trên toàn bộ folder test
    output_dir = "/teamspace/studios/this_studio/VolumeNERF/test_results"
    mean_xray_path = "/teamspace/studios/this_studio/VolumeNERF/DRR_Parameters/mean_xray.npy"
    std_xray_path = "/teamspace/studios/this_studio/VolumeNERF/DRR_Parameters/std_xray.npy"
    test_model(checkpoint_path, test_dir, output_dir, mean_xray_path, std_xray_path)
