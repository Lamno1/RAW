# Thứ tự chạy pipeline BuildingNet — legacy baseline

> **Trạng thái (2026-09-21): chỉ dùng để tái lập baseline lịch sử.**
> Không dùng thứ tự này để triển khai CAMO-Pyramid. Nguồn thẩm quyền mới là [`CAMO_PYRAMID_ARCHITECTURE.md`](CAMO_PYRAMID_ARCHITECTURE.md) và [`plans/CAMO_PYRAMID_BLUEPRINT.md`](plans/CAMO_PYRAMID_BLUEPRINT.md). Mọi agent chạy thí nghiệm mới phải bắt đầu từ S0 trong blueprint.

Chạy các lệnh dưới đây trong PowerShell tại thư mục `D:\raw`.

## 1. Kiểm tra môi trường và dữ liệu

```powershell
cd D:\raw
python -m py_compile dataset.py model.py train.py eval_metrics.py cache_and_audit.py
python dataset.py
python model.py
```

Kết quả đúng phải có input `(1024, 9)` và output mô hình
`(batch, 1024, 31)`.

## 2. Tạo cache và audit — chỉ chạy một lần

```powershell
python cache_and_audit.py
```

Hiện bước này đã hoàn tất. Không cần chạy lại trừ khi dữ liệu gốc thay đổi.
Các đầu ra là `cache_npz/`, `class_counts.npy`, `class_weights.npy` và
`class_audit.json`.

## 3. Huấn luyện baseline Cross-Entropy

```powershell
python train.py --points 1024 --batch-size 8 --epochs 50 --seed 42 --loss ce
```

Checkpoint tốt nhất được ghi vào
`checkpoints/best_segmentation_model.pth`.

## 4. Đánh giá baseline trên validation

Checkpoint được chọn tự động bằng validation Part IoU trong lúc train. Không chạy
test để điều chỉnh kiến trúc hoặc siêu tham số.

## 5. Lưu riêng checkpoint baseline

Trước khi chạy thí nghiệm tiếp theo, sao chép checkpoint để không bị ghi đè:

```powershell
Copy-Item checkpoints\best_segmentation_model.pth checkpoints\baseline_ce_seed42.pth
```

## 6. Huấn luyện Weighted Cross-Entropy

```powershell
python train.py --points 1024 --batch-size 8 --epochs 50 --seed 42 --loss weighted_ce
Copy-Item checkpoints\best_segmentation_model.pth checkpoints\weighted_ce_seed42.pth
```

## 7. Lặp lại để có ba seed

Chạy lại bước 3 và 6 với `--seed 123` và `--seed 2026`, đồng thời đổi tên
checkpoint theo seed ngay sau mỗi lần chạy.

## 8. Đánh giá test — chỉ sau khi khóa thiết kế

```powershell
python eval_metrics.py --checkpoint checkpoints\baseline_ce_seed42.pth --points 1024 --batch-size 8
python eval_metrics.py --checkpoint checkpoints\weighted_ce_seed42.pth --points 1024 --batch-size 8
```

Các metric cần ghi nhận là Class Accuracy, Part IoU và Shape IoU. Không sử dụng
test để lựa chọn loss, kiến trúc hoặc hyperparameter.

## Khi gặp CUDA out of memory

Giảm batch size trước, không giảm số lớp hoặc thay đổi dữ liệu:

```powershell
python train.py --points 1024 --batch-size 4 --epochs 50 --seed 42 --loss ce
```
