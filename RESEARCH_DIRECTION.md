# Hướng Nghiên Cứu Được Chấp Nhận — BuildingNet

**Trạng thái:** Đã chấp nhận  
**Ngày cập nhật:** 2026-09-20  
**Mục tiêu:** Nâng cấp toàn diện hệ thống semantic part segmentation trên BuildingNet, ưu tiên tính đúng đắn của pipeline, khả năng bảo toàn bộ phận nhỏ và hiệu quả trên phân bố long-tail.

## 1. Kết luận chiến lược

Không giới hạn nghiên cứu ở roadmap cũ hoặc chỉ bổ sung các module vào GeometryAwarePCT. Trước khi thiết kế một kiến trúc hoàn toàn mới, cần xác định liệu nút thắt chính nằm ở:

- pipeline dữ liệu và ánh xạ nhãn;
- số điểm lấy mẫu quá thấp;
- cách tính metric hoặc chia tập dữ liệu;
- hiện tượng overfitting;
- mất cân bằng lớp;
- hay bản thân kiến trúc mạng.

Hướng chính được chấp nhận là xây dựng một hệ thống **đa độ phân giải, nhận biết long-tail và bảo toàn chi tiết hình học**, thay vì chỉ thay Global Attention bằng local kNN attention.

## 2. Nhận định về baseline hiện tại

Baseline GeometryAwarePCT đạt:

- Val mIoU trung bình: **14.18% ± 0.39%**;
- Val accuracy trung bình: **62.44% ± 0.90%**;
- Train mIoU cuối quá trình: khoảng **35.27%**.

Khoảng cách lớn giữa train và validation có thể phản ánh overfitting, sai khác phân phối hoặc vấn đề trong pipeline validation. Vì vậy, chưa có đủ bằng chứng để kết luận rằng Global Attention là nguyên nhân chính.

Baseline MinkowskiNet chính thức của BuildingNet báo cáo khoảng **33.5 pIoU** trên point cloud với normal và RGB. Chỉ được so sánh trực tiếp sau khi xác minh cùng split, label mapping, dữ liệu đầu vào và cách tính metric. Tuy nhiên, khoảng cách lớn này là lý do bắt buộc phải audit pipeline trước khi phát triển kiến trúc mới.

## 3. Các giả thuyết nghiên cứu chính

### H1 — Độ phân giải đầu vào là nút thắt quan trọng

Lấy mẫu 1.024 điểm cho toàn bộ một công trình có thể làm các lớp nhỏ như gate, lighting, shutter, dormer hoặc awning biến mất hoàn toàn hoặc chỉ còn vài điểm.

Nếu thông tin đã mất trong quá trình sampling, attention và loss function không thể khôi phục được.

### H2 — Sampling bảo toàn chi tiết quan trọng hơn loss weighting đơn thuần

Weighted Cross-Entropy và Focal Loss chỉ thay đổi gradient của các điểm còn tồn tại. Chúng không tạo thêm quan sát cho lớp hiếm đã bị loại trong quá trình lấy mẫu.

Do đó cần ưu tiên class-aware, boundary-aware hoặc multi-crop sampling trước khi kết luận hiệu quả của loss.

### H3 — Pretraining có lợi hơn việc huấn luyện backbone phức tạp từ đầu

BuildingNet chỉ có khoảng 1.849 mẫu cache. Fine-tune một backbone point-cloud pretrained như Sonata/PTv3 có khả năng tổng quát hóa tốt hơn việc huấn luyện một Transformer hoặc Mamba mới hoàn toàn từ đầu.

Đây là giả thuyết cần được kiểm nghiệm; không mặc định rằng pretraining từ dữ liệu indoor/outdoor sẽ chuyển miền hoàn hảo sang building mesh point clouds.

### H4 — Ngữ nghĩa phân cấp phù hợp hơn classifier phẳng 31 lớp

Các bộ phận kiến trúc có quan hệ tự nhiên như structural, opening, roof, ground và decorative. Auxiliary hierarchical supervision có thể giúp các lớp hiếm học đặc trưng ổn định hơn.

## 4. Kiến trúc mục tiêu: Building-Hierarchy Network

Tên làm việc: **Building-Hierarchy Network (BHN)**.

```text
Toàn bộ point cloud
        |
        +-- Nhánh toàn cục dạng voxel/serialized
        |     +-- hình dáng tổng thể, tầng, mặt đứng, quan hệ xa
        |
        +-- Nhánh patch cục bộ mật độ cao
        |     +-- cửa, lan can, cột và các chi tiết nhỏ
        |
        +-- Đặc trưng hình học bổ sung
        |     +-- relative height, normal, curvature,
        |         local density và kNN offsets
        |
        +-- Cross-scale gated fusion
                    |
              classifier + auxiliary heads
```

### 4.1. Đầu vào và sampling

- Không phụ thuộc vào một mẫu ngẫu nhiên 1.024 điểm duy nhất.
- Thử nghiệm 1.024, 4.096 và nếu tài nguyên cho phép là 8.192–16.384 điểm.
- Kết hợp:
  - uniform/FPS sampling để giữ hình dạng toàn cục;
  - class-aware hoặc boundary-aware sampling để giữ chi tiết nhỏ;
  - overlapping crops và voting khi inference.

### 4.2. Nhánh toàn cục

- Ưu tiên serialized attention kiểu Point Transformer V3 hoặc sparse voxel backbone.
- Tránh Global Attention đầy đủ có độ phức tạp \(O(N^2)\) trên toàn bộ điểm.
- PointMamba có thể được dùng làm backbone đối chứng hoặc nhánh toàn cục thử nghiệm, nhưng không phải giả thuyết trung tâm ban đầu.

### 4.3. Nhánh cục bộ

- Dùng kNN/EdgeConv đa tỉ lệ, ví dụ \(k=8,16,32\).
- Học offsets, chênh lệch normal, curvature, mật độ cục bộ và relative height.
- Chỉ áp dụng xử lý hàng xóm chi tiết tại các tầng cần thiết để kiểm soát chi phí tính toán.

### 4.4. Fusion và supervision

- Gated cross-scale fusion giữa đặc trưng toàn cục và cục bộ.
- Head chính dự đoán 31 lớp.
- Auxiliary boundary head.
- Auxiliary superclass head cho các nhóm kiến trúc cấp cao.
- Có thể thử prototype hoặc supervised contrastive regularization cho các lớp tail sau khi baseline chính ổn định.

## 5. Roadmap thực nghiệm mới

### Phase 0 — Data, metric và pipeline audit

- [ ] So khớp train/validation/test split với BuildingNet chính thức.
- [ ] Xác minh ánh xạ đầy đủ của 31 lớp và ID sau khi bỏ `undetermined`.
- [ ] Kiểm tra việc bỏ `undetermined` diễn ra trước hay sau sampling.
- [ ] Đếm số điểm mỗi lớp trước và sau sampling.
- [ ] Đếm số mẫu mà từng lớp hiếm biến mất hoàn toàn sau sampling.
- [ ] Xác minh công thức pIoU, sIoU, class accuracy và cách xử lý lớp absent.
- [ ] Xuất confusion matrix và per-class IoU cho từng seed.
- [ ] Kiểm tra data leakage, normalization và khác biệt giữa train/validation preprocessing.
- [ ] Chạy oracle experiment: giữ nguyên mô hình và tăng 1.024 lên 4.096 điểm.

**Cổng quyết định:** Nếu tăng số điểm hoặc multi-crop tạo cải thiện lớn, ưu tiên sửa sampling trước mọi thay đổi kiến trúc.

### Phase 1 — Baseline mạnh và có thể so sánh

- [ ] GeometryAwarePCT + CE + 1.024 điểm.
- [ ] GeometryAwarePCT + CE + 4.096 điểm.
- [ ] GeometryAwarePCT + multi-crop/voting inference.
- [ ] Sparse voxel U-Net hoặc Minkowski-style baseline.
- [ ] Point Transformer V3 huấn luyện từ đầu.
- [ ] Point Transformer V3/Sonata pretrained và fine-tune.

Các baseline phải dùng cùng split, preprocessing, augmentation và metric.

### Phase 2 — Input và sampling ablation

- [ ] XYZ.
- [ ] XYZ + normal.
- [ ] XYZ + normal + RGB.
- [ ] Uniform random sampling.
- [ ] FPS sampling.
- [ ] Class-aware sampling.
- [ ] Boundary-aware sampling.
- [ ] Hybrid global + local patch sampling.

### Phase 3 — Long-tail learning

- [ ] Cross-Entropy.
- [ ] Weighted Cross-Entropy với trọng số được giới hạn.
- [ ] Effective-number Cross-Entropy.
- [ ] Focal Loss.
- [ ] Class-balanced Focal Loss.
- [ ] CE kết hợp Dice hoặc Lovász loss.
- [ ] Logit adjustment hoặc Balanced Softmax.
- [ ] Sampling-aware loss experiments.

Không dùng nghịch đảo tần suất thô làm lựa chọn mặc định vì có nguy cơ tạo trọng số quá lớn cho lớp cực hiếm.

### Phase 4 — Building-Hierarchy Network

Thêm từng thành phần theo thứ tự để giữ khả năng diễn giải ablation:

1. Local multi-scale geometry branch.
2. Global serialized hoặc sparse branch.
3. Cross-scale gated fusion.
4. Boundary auxiliary head.
5. Hierarchical semantic head.
6. Tail-class prototype/contrastive regularization nếu cần.

### Phase 5 — Đánh giá toàn diện

- [ ] Ít nhất ba seed cho cấu hình cuối và các baseline quan trọng.
- [ ] Đánh giá trên validation và test theo protocol chính thức.
- [ ] Báo cáo accuracy, pIoU, sIoU và per-class IoU.
- [ ] Báo cáo head/mid/tail mIoU và tail-class recall.
- [ ] Báo cáo số tham số, FLOPs, VRAM, training time và inference latency.
- [ ] Phân tích lỗi bằng confusion matrix và trực quan hóa point cloud.
- [ ] Kiểm tra hiệu quả trong chế độ ít dữ liệu.

### Phase 6 — Bài báo và reproducibility

- [ ] Cập nhật phương pháp, protocol và ablation vào bản thảo LaTeX.
- [ ] Ghi lại cấu hình, seed và checkpoint của mọi experiment.
- [ ] Công bố chính xác cách sampling, label mapping và metric.
- [ ] Phân biệt rõ kết quả quan sát, giả thuyết và diễn giải.

## 6. Tiêu chí thành công

Không chỉ tối ưu overall accuracy hoặc một con số mIoU duy nhất. Hệ thống được coi là cải thiện thực sự khi:

- tăng pIoU/sIoU hoặc mIoU theo protocol đã xác minh;
- cải thiện rõ rệt head/mid/tail mIoU;
- tăng recall cho các lớp hiếm mà không làm sụp đổ các lớp phổ biến;
- ổn định giữa các seed;
- có chi phí bộ nhớ và inference chấp nhận được;
- đóng góp của mỗi module được chứng minh bằng ablation.

## 7. Nguyên tắc ra quyết định

1. Không quy lỗi cho kiến trúc trước khi audit dữ liệu và metric.
2. Không thêm module nếu chưa xác định giả thuyết mà module đó kiểm nghiệm.
3. Không tuyên bố state-of-the-art nếu protocol không trùng với benchmark chính thức.
4. Ưu tiên thay đổi đơn giản có bằng chứng mạnh trước thay đổi phức tạp.
5. Chấp nhận loại bỏ hoặc thay đổi kiến trúc đề xuất nếu kết quả thực nghiệm bác bỏ giả thuyết.
6. Mọi kết luận chính phải dựa trên nhiều seed và per-class analysis.

## 8. Tài liệu tham khảo định hướng

- BuildingNet: <https://buildingnet.org/>
- BuildingNet official MinkowskiNet baseline: <https://github.com/buildingnet/buildingnet_dataset/blob/main/MinkowskiNet/README.md>
- Point Transformer V3: <https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html>
- PointMamba: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html>
- Sonata: <https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.pdf>
- Focal Loss: <https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html>
- Class-Balanced Loss: <https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf>

