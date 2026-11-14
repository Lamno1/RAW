import sys
import os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
import plotly.graph_objects as go
from dataset import BuildingNetDataset
from model import PointCloudTransformer
import random

def visualize_random_prediction():
    # --- 1. Cấu hình ---
    DATA_DIR = r'd:\raw'
    MODEL_PATH = 'checkpoints/best_pct_model.pth'
    NUM_CLASSES = 53
    
    print("[*] Đang nạp tập dữ liệu Test...")
    test_dataset = BuildingNetDataset(root_dir=DATA_DIR, split='test')
    idx_to_label = {v: k for k, v in test_dataset.label_to_idx.items()}
    
    # Lấy ngẫu nhiên một mẫu từ tập Test
    sample_idx = random.randint(0, len(test_dataset) - 1)
    sample = test_dataset[sample_idx]
    points = sample['point_cloud'] # Kích thước: [1024, 3]
    
    true_label_idx = sample['label'].item()
    true_label_name = idx_to_label[true_label_idx]
    
    # --- 2. Xử lý Suy luận (Inference) ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pred_label_name = "Chưa có Model (Đang huấn luyện)"
    
    if os.path.exists(MODEL_PATH):
        model = PointCloudTransformer(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        with torch.no_grad():
            pts_tensor = points.unsqueeze(0).to(device) # Tạo batch size = 1
            output = model(pts_tensor)
            _, pred_idx = torch.max(output, 1)
            pred_label_name = idx_to_label[pred_idx.item()]
    else:
        print("[!] Chưa tìm thấy checkpoint. Sẽ chỉ hiển thị Ground Truth.")

    # --- 3. Dựng hình 3D bằng Plotly ---
    print(f"[*] Đang kết xuất HTML: Thực tế [{true_label_name}] | Dự đoán [{pred_label_name}]")
    
    points_np = points.numpy()
    x, y, z = points_np[:, 0], points_np[:, 1], points_np[:, 2]
    
    # Scatter3d với màu sắc (colorscale) ánh xạ theo độ cao Z để làm nổi bật hình khối
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3.5,
            color=z,
            colorscale='Viridis',
            opacity=0.9
        )
    )])

    fig.update_layout(
        title=f"<b>Phân tích Hình học 3D</b><br>Thực tế: {true_label_name} | Dự đoán: {pred_label_name}",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            aspectmode='data' # Giữ nguyên tỷ lệ gốc của cấu kiện
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    output_file = 'visualization_3d.html'
    fig.write_html(output_file)
    print(f"[*] Hoàn tất! File đã tạo tại '{os.path.abspath(output_file)}'. Hãy mở bằng trình duyệt (Chrome/Edge) để tương tác xoay 360 độ.")

if __name__ == '__main__':
    visualize_random_prediction()
