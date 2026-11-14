# Kiến Trúc Frontier — Conservative Adaptive Manifold Operator for BuildingNet

> **Status update (2026-09-21): superseded as implementation authority.** This file preserves the original CAMO-Build hypothesis and Prototype-1 history. The mathematically reviewed successor is [`CAMO_PYRAMID_ARCHITECTURE.md`](CAMO_PYRAMID_ARCHITECTURE.md); execution must follow [`plans/CAMO_PYRAMID_BLUEPRINT.md`](plans/CAMO_PYRAMID_BLUEPRINT.md). If the documents conflict, the successor and blueprint take precedence.

**Tên làm việc:** CAMO-Build  
**Trạng thái:** Giả thuyết kiến trúc mới, triển khai song song với hướng an toàn  
**Ngày:** 2026-09-20  
**Ràng buộc mục tiêu:** GPU 4 GB, dữ liệu khoảng 1.849 công trình, point cloud gốc có thể tới 100k điểm, 31 lớp long-tail.

> **Prototype 1 freeze record (2026-09-20):** Các điều kiện toán học và phần cứng của Prototype 1 đã được chấp nhận chính thức. Đặc tả có thẩm quyền nằm trong `PROTOTYPE1_MATHEMATICAL_REVIEW.md`; nếu có mâu thuẫn, đặc tả đã đóng băng được ưu tiên.

> **Prototype 1 decision (2026-09-21):** 6/6 full runs hoàn tất. Density-mode mass weighting cải thiện val mIoU +0.00071 (+0.56%) trên 3/3 seed nhưng chưa có ý nghĩa thống kê (p = 0.0673) — bằng chứng xu hướng, chưa kết luận. Chi tiết: `PROTOTYPE1_EXECUTION_REPORT.md`. Quyết định: tiến hành **Prototype 2 — OT coarsening** (§13 dưới đây).

## 1. Tham vọng và giới hạn của tuyên bố

Mục tiêu dài hạn là tạo một họ mô hình mới đồng thời:

1. chính xác hơn trên bộ phận nhỏ và lớp hiếm;
2. gần tuyến tính theo số điểm;
3. ổn định khi mật độ và cách lấy mẫu thay đổi;
4. xử lý được nhiều điểm hơn trong cùng ngân sách VRAM;
5. có thể pretrain và chuyển giao giữa các point cloud có độ phân giải khác nhau.

Không tuyên bố trước rằng mô hình sẽ vượt mọi mô hình hiện tại. Tuyên bố khoa học chỉ được hình thành sau khi CAMO-Build vượt các baseline mạnh dưới cùng protocol, ngân sách tính toán và số seed.

## 2. Ý tưởng trung tâm

Point cloud không được xem là một chuỗi token tùy ý mà là một **độ đo rời rạc trên mặt đa tạp**:

\[
\mu_N = \sum_{i=1}^{N} m_i\,\delta_{(x_i,a_i)},
\]

trong đó:

- \(x_i\in\mathbb{R}^3\) là tọa độ;
- \(a_i\) chứa normal, RGB và thuộc tính bổ sung;
- \(m_i\) là khối lượng diện tích mà điểm đại diện;
- \(\delta\) là Dirac measure.

Tư tưởng thống nhất là: **sampling, pooling, message passing và decoding đều phải tôn trọng khối lượng của độ đo**. Một chi tiết nhỏ không được biến mất chỉ vì nó có ít điểm hoặc vì bước downsampling không nhận biết cấu trúc.

CAMO-Build gồm năm cơ chế gắn kết:

1. mass-aware geometric lifting;
2. conservative optimal-transport coarsening;
3. local–global manifold neural operator;
4. uncertainty-routed detail refinement;
5. hierarchical long-tail decision geometry.

## 3. Đối xứng đúng cho công trình

Đẳng biến \(SE(3)\) đầy đủ không phải lựa chọn mặc định phù hợp. Nếu quay công trình 180 độ quanh trục ngang, roof và ground sẽ đổi vai trò vật lý dù hình học tương đối vẫn tương tự.

CAMO-Build dùng nhóm biến đổi bảo toàn trọng lực:

\[
G = \mathbb{R}^3 \rtimes SO(2)_z,
\]

gồm tịnh tiến và quay quanh trục đứng. Đặc trưng hình học bất biến cơ bản giữa hai điểm là:

\[
r_{ij}=\sqrt{(x_j-x_i)^2+(y_j-y_i)^2},\quad
\Delta z_{ij}=z_j-z_i,
\]

\[
c_{ij}=n_i^\top n_j,\quad
v_i=n_i^\top e_z,\quad
d_{ij}=\|x_j-x_i\|_2.
\]

Nhờ đó mô hình bất biến với góc phương vị nhưng vẫn phân biệt chiều cao, mặt đứng, mặt ngang, mái và mặt đất.

## 4. Stage A — Mass-aware geometric lifting

Khối lượng ban đầu của điểm được ước lượng từ mật độ lân cận:

\[
m_i = \frac{\rho_i^{-1}}{\sum_j \rho_j^{-1}},
\qquad
\rho_i = \frac{k}{\sum_{j\in\mathcal N_k(i)}(d_{ij}+\epsilon)^2}.
\]

Điểm ở vùng được lấy mẫu dày không được có ảnh hưởng lớn một cách giả tạo chỉ vì có nhiều điểm hơn.

Lifting đầu vào:

\[
h_i^0 = \phi_0[x_i, n_i, c_i, z_i-z_{\min}, \log\rho_i, \kappa_i],
\]

với \(c_i\) là RGB và \(\kappa_i\) là curvature ước lượng. Curvature thủ công chỉ là kênh phụ; operator sẽ học lại hình học từ quan hệ lân cận.

## 5. Stage B — Conservative Optimal-Transport Coarsening

Thay FPS hoặc random sampling bằng ánh xạ vận chuyển mềm từ \(N_l\) điểm sang \(N_{l+1}\) đại diện:

\[
P_l^* = \arg\min_{P\geq 0}
\langle P,C_l\rangle
+\varepsilon \sum_{ij}P_{ij}(\log P_{ij}-1),
\]

với ràng buộc:

\[
P\mathbf 1=m^{(l)},\qquad
P^\top\mathbf 1=m^{(l+1)}.
\]

Chi phí vận chuyển không chỉ là khoảng cách Euclid:

\[
C_{ij}=\alpha\|x_i-q_j\|_2^2
+\beta(1-n_i^\top \bar n_j)
+\gamma|s_i-s_j|
+\eta b_i.
\]

Trong đó \(s_i\) là saliency/uncertainty và \(b_i\) là boundary evidence. Capacity của đại diện được điều chỉnh để vùng curvature cao, biên mạnh hoặc uncertainty cao nhận nhiều token hơn.

Pooling bảo toàn khối lượng:

\[
m_j' = \sum_i P_{ij},\qquad
h_j'=\frac{1}{m_j'+\epsilon}\sum_iP_{ij}h_i.
\]

Unpooling dùng cùng transport plan thay vì nội suy độc lập. Điều này tạo consistency tự nhiên giữa encoder và decoder.

### Điểm mới cần kiểm nghiệm

Không phải “learned sampling” nói chung, mà là **sampling như vận chuyển khối lượng có ràng buộc**, được tối ưu cùng segmentation. Mục tiêu là ổn định theo discretization: hai cách lấy mẫu khác nhau của cùng bề mặt phải tạo biểu diễn gần nhau.

## 6. Stage C — Local–Global Conservative Manifold Operator

### 6.1. Toán tử cục bộ

Với mỗi tầng, toán tử tích phân rời rạc là:

\[
(\mathcal K_{\theta}^{\mathrm{loc}}h)_i =
\frac{1}{Z_i}
\sum_{j\in\mathcal N_k(i)}
m_j\,\kappa_\theta(g_{ij})\,V_\theta h_j,
\]

trong đó:

\[
g_{ij}=[r_{ij},\Delta z_{ij},d_{ij},c_{ij},v_i,v_j,
\Delta RGB_{ij},\log(\rho_j/\rho_i)].
\]

Chuẩn hóa:

\[
Z_i=\sum_{j\in\mathcal N_k(i)}m_j\,\kappa_\theta(g_{ij})+\epsilon.
\]

Khác attention thông thường, hệ số \(m_j\) làm phép tổng gần với tích phân trên bề mặt, thay vì phụ thuộc trực tiếp vào số điểm được lấy mẫu.

### 6.2. Toán tử toàn cục hạng thấp

Global attention \(O(N^2)\) được thay bằng kernel hạng thấp:

\[
(\mathcal K_{\theta}^{\mathrm{glob}}h)(x_i)
=\sum_{r=1}^{R}\phi_r(x_i)
\left(\sum_{j=1}^{N}m_j\psi_r(x_j)V_rh_j\right),
\]

với \(R\ll N\). Độ phức tạp là \(O(NRD)\), gần tuyến tính theo \(N\).

Các mode \(\phi_r,\psi_r\) được điều kiện hóa bởi height band, hướng normal và OT landmarks. Vì vậy global context không phụ thuộc vào một thứ tự serialization duy nhất như SSM/Mamba.

### 6.3. Cập nhật operator

\[
h_i^{l+1}=h_i^l+sigma\left(
W_lh_i^l+
\mathcal K_l^{\mathrm{loc}}h_i^l+
g_i^l\odot\mathcal K_l^{\mathrm{glob}}h_i^l
\right),
\]

\[
g_i^l=\operatorname{sigmoid}(G_l[h_i^l,u_i^l]),
\]

trong đó \(u_i^l\) là uncertainty. Điểm rõ ràng dùng ít global mixing; điểm mơ hồ hoặc bộ phận nhỏ được nhận thêm ngữ cảnh toàn công trình.

## 7. Stage D — Uncertainty-Routed Detail Refinement

Mô hình chạy một coarse pass rẻ trên toàn bộ công trình và ước lượng:

\[
u_i = H(p_i)
+\lambda_1\|p_i-\bar p_{\mathcal N(i)}\|_1
+\lambda_2 b_i,
\]

trong đó \(H(p_i)=-\sum_cp_{ic}\log p_{ic}\) là entropy.

Router chọn ngân sách patch cố định:

\[
\mathcal S=\operatorname{TopK}_{\mathrm{diff}}(u,B).
\]

Chỉ các patch khó, biên hoặc giàu chi tiết được đưa qua local operator độ phân giải cao. Dự đoán cuối:

\[
z_i=z_i^{\mathrm{coarse}}+
r_i\,\Delta z_i^{\mathrm{detail}},
\qquad r_i\in[0,1].
\]

Đây là conditional computation: tăng độ chính xác nơi cần thiết mà không buộc toàn bộ 100k điểm đi qua backbone nặng.

## 8. Stage E — Hierarchical Long-Tail Decision Geometry

### 8.1. Cây nhãn

Mỗi lớp lá \(c\) thuộc một superclass \(g(c)\), ví dụ structural, opening, roof, ground hoặc decorative:

\[
p(y=c\mid x)=p(g(c)\mid x)\,p(c\mid g(c),x).
\]

Điều này chia một quyết định 31-way khó thành quyết định coarse-to-fine có cấu trúc.

### 8.2. Classifier prototype chuẩn hóa

\[
z_{ic}=s\frac{h_i^\top w_c}{\|h_i\|\|w_c\|}-\tau\log(\pi_c+\epsilon),
\]

trong đó \(\pi_c\) là prior lớp chỉ tính từ train set. Prototype được regularize để các lớp anh em có margin đủ lớn nhưng vẫn gần superclass prototype.

### 8.3. Objective tổng hợp

\[
\mathcal L=
\mathcal L_{\mathrm{balanced\ CE}}
+\lambda_{IoU}\mathcal L_{\mathrm{macro\ Lovasz}}
+\lambda_h\mathcal L_{\mathrm{hier}}
+\lambda_b\mathcal L_{\mathrm{boundary}}
+\lambda_t\mathcal L_{\mathrm{transport}}
+\lambda_c\mathcal L_{\mathrm{consistency}}
+\lambda_q\mathcal L_{\mathrm{budget}}.
\]

- Balanced CE ổn định optimization ban đầu.
- Macro Lovász hướng trực tiếp hơn tới mIoU.
- Transport loss phạt mất coverage hoặc co cụm đại diện.
- Consistency loss ép hai sampling khác nhau của cùng công trình cho dự đoán tương thích.
- Budget loss giữ số patch chi tiết trong giới hạn VRAM/thời gian.

Không bật toàn bộ loss từ epoch đầu. Dùng curriculum: CE → hierarchy → transport/consistency → IoU/detail routing.

## 9. Pretraining mới thay vì chỉ mượn backbone

CAMO-Build có thể khởi tạo từ Sonata/PTv3, nhưng mục tiêu xa hơn là pretrain chính operator bằng ba nhiệm vụ:

1. **Measure consistency:** hai discretization của cùng bề mặt phải có biểu diễn OT-aligned gần nhau.
2. **Masked attribute recovery:** che RGB/normal/geometry luân phiên để tránh geometric shortcut.
3. **Cross-scale transport prediction:** dự đoán transport plan giữa point cloud thưa và dày.

Loss pretraining:

\[
\mathcal L_{pre}=
\lambda_m W_\varepsilon(F_\theta(\mu_a),F_\theta(\mu_b))
+\lambda_r\mathcal L_{recon}
+\lambda_p\operatorname{KL}(P^*\|\hat P_\theta).
\]

Đây là hướng foundation model cho hình học kiến trúc, không chỉ fine-tune một backbone scene segmentation có sẵn.

## 10. Độ phức tạp mục tiêu

Với \(N\) điểm, \(k\) hàng xóm, global rank \(R\), \(B\) patch được refine và mỗi patch có \(P\) điểm:

\[
T(N)=O(NkD+NRD+BPkD),
\]

so với global attention:

\[
T_{attn}(N)=O(N^2D).
\]

Để chạy trên 4 GB VRAM:

- coarse branch dùng 4k–8k OT representatives;
- operator width 64–128;
- \(k\in\{8,16\}\), \(R\in[16,64]\);
- mixed precision và gradient accumulation;
- chỉ 4–16 patch khó được refine mỗi lượt;
- kNN graph và geometric invariants được cache khi augmentation cho phép.

## 11. Các đóng góp có khả năng trở thành paper claim

Chỉ giữ claim nào được ablation xác nhận:

1. Conservative OT coarsening bảo toàn chi tiết tốt hơn FPS/random/learned top-k.
2. Measure-weighted manifold operator ổn định hơn khi đổi sampling density.
3. Low-rank global operator đạt trade-off accuracy–memory tốt hơn global attention và serialized SSM.
4. Uncertainty routing tăng tail-IoU dưới ngân sách FLOPs cố định.
5. Gravity-preserving symmetry phù hợp building segmentation hơn \(SE(3)\) đầy đủ hoặc augmentation quay tùy ý.
6. Cross-discretization pretraining cải thiện chuyển giao trong chế độ ít dữ liệu.

## 12. Falsification tests

Kiến trúc phải được bác bỏ hoặc đơn giản hóa nếu:

- OT sampling không tốt hơn FPS dưới cùng số token;
- thay đổi sampling density vẫn làm output biến động mạnh;
- low-rank operator kém PTv3/SSM ở cùng VRAM và latency;
- router chỉ chọn wall/roof thay vì boundary và tail regions;
- hierarchical head tăng head classes nhưng giảm tail recall;
- chi phí Sinkhorn lớn hơn lợi ích thu được.

Các kiểm tra bắt buộc:

- density stress test: 1k/2k/4k/8k points;
- point-drop/noise stress test;
- rotation quanh trục đứng và translation consistency;
- per-class retained-mass sau mỗi lần coarsening;
- accuracy–VRAM–latency Pareto curve;
- ít nhất ba seed cho kết quả cuối.

## 13. Kế hoạch nguyên mẫu theo rủi ro

### Prototype 1 — Operator tối thiểu

- Measure weights + local conservative operator.
- Không OT, không router, không hierarchy.
- So với EdgeConv/local attention trên cùng 1.024 và 4.096 điểm.

### Prototype 1.5 — Lovász-Softmax loss ablation (quyết định 2026-09-21)

Chạy trước Prototype 2, trên cùng kiến trúc Prototype 1 (LMAO + balanced CE), chỉ đổi objective:

\[
\mathcal L=\mathcal L_{B\text{-}CE}+\lambda_L\mathcal L_{\mathrm{Lovász}},
\]

với \(\lambda_L\) cố định (không grid-search rộng), cùng 4.096 điểm, cùng seed {42, 123, 2026}, cùng chế độ mass (`none` hoặc `density`), cùng optimizer/lr schedule của Prototype 1. Với 31 lớp và nhiều lớp vắng mặt trong batch, Lovász phải bỏ qua lớp không xuất hiện trong batch thay vì tính IoU giả.

Lý do ưu tiên bước này trước OT/backend mới: tác động trực tiếp vào mIoU/IoU đang tối ưu, không có rủi ro dependency/CUDA ABI, và trả lời câu hỏi rõ ràng: LMAO + balanced CE **vs** LMAO + balanced CE + Lovász.

Báo cáo bắt buộc: mIoU, head/mid/tail IoU, per-class IoU, độ ổn định giữa seed, gradient norm, peak VRAM, epoch time.

**Chính sách cài dependency mới (áp dụng cho toàn bộ kế hoạch nguyên mẫu):** không cài PyKeOps, mamba-ssm hay torch-cluster trước khi có kết quả ablation loss này.

1. Lovász-Softmax (bước này) — không rủi ro môi trường, làm trước.
2. Hardware kNN acceleration (pyg_lib/torch_scatter/torch_sparse theo đúng cặp PyTorch 2.6 + CUDA 12.4 từ tài liệu PyG chính thức) — chỉ cài nếu profiling chứng minh `cKDTree` là bottleneck.
3. Sparse/Unbalanced OT (Prototype 2, xem dưới) — PyKeOps (nếu dùng) chỉ giảm bộ nhớ nhờ LazyTensor matrix-free, **không** biến Sinkhorn dense \(O(NM)\) thành \(O(N)\); vẫn bắt buộc sparse kNN support + log-domain Sinkhorn + marginal residual + dustbin/unbalanced fallback trước khi backprop qua transport.
4. Gradient checkpointing — tối ưu nội bộ PyTorch, thêm sau khi kiến trúc và loss đã ổn định; không dùng để che một tensor dense hoặc graph kNN thiết kế sai.
5. Mamba (mamba-ssm) — để cuối cùng; không phải dependency nền tảng cho giả thuyết CAMO hiện tại, cài đặt chính thức có thể cần compile selective-scan CUDA extension thủ công.

### Prototype 2 — OT coarsening

- Thêm Sinkhorn transport giữa hai tầng.
- Đo retained mass, Chamfer coverage, tail-point survival và mIoU.

### Prototype 3 — Low-rank global branch

- So với global attention, PTv3-style serialization và bidirectional SSM.
- Vẽ Pareto accuracy–VRAM–latency.

### Prototype 4 — Conditional detail routing

- Coarse 4k + top-B high-resolution patches.
- Kiểm tra router có chọn đúng vùng khó hay không.

### Prototype 5 — Hierarchy và pretraining

- Chỉ thêm sau khi representation backbone đã chứng minh hiệu quả.
- Fine-tune pretrained external backbone là control bắt buộc.

## 14. Vị trí so với hướng an toàn

Hướng an toàn trả lời: “Làm thế nào cải thiện BuildingNet bằng các kỹ thuật đã được kiểm chứng?”

CAMO-Build trả lời câu hỏi xa hơn:

> Có thể xây dựng một segmentation operator gần tuyến tính, bảo toàn độ đo, ổn định theo discretization và tự phân bổ compute vào chi tiết hiếm hay không?

Hai hướng chạy song song. Pipeline audit, metric, split, visualization và strong baselines được dùng chung; CAMO-Build không được miễn khỏi các tiêu chuẩn kiểm chứng của hướng an toàn.

## 15. Nền tảng học thuật

- Point Transformer V3 — serialized point processing: <https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html>
- PointMamba — linear-complexity state-space processing: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html>
- Sonata — self-supervised point representation: <https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.pdf>
- CDSegNet — single-step conditional diffusion segmentation: <https://openaccess.thecvf.com/content/CVPR2025/html/Qu_An_End-to-End_Robust_Point_Cloud_Semantic_Segmentation_Network_with_Single-Step_CVPR_2025_paper.html>
- Geometric neural operators for point-cloud representations: <https://doi.org/10.1088/2632-2153/ae1bf8>
- Geometric operator learning with optimal transport: <https://jmlr.org/beta/papers/v26/25-1380.html>
- SampleNet — differentiable point-cloud sampling: <https://openaccess.thecvf.com/content_CVPR_2020/papers/Lang_SampleNet_Differentiable_Point_Cloud_Sampling_CVPR_2020_paper.pdf>
- Lovász-Softmax — direct IoU surrogate: <https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html>
- SE(3)-Transformer — geometric equivariance foundation: <https://papers.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf>

## 16. Rigor revision: các sửa đổi bắt buộc trước triển khai

Phần này có quyền ưu tiên cao hơn các mô tả ban đầu nếu có mâu thuẫn.

### 16.1. Sửa tuyên bố về hội tụ của measure weights

Việc đưa \(m_j\) vào tổng rời rạc **không tự động bảo đảm** hội tụ về tích phân Riemann. Kết luận đó chỉ có thể được phát biểu dưới các giả thiết về:

- bề mặt compact và đủ regular;
- sampling không suy biến;
- estimator mật độ nhất quán;
- bán kính kNN tiến về 0 trong khi \(k\to\infty\) và \(k/N\to0\);
- kernel bị chặn và đủ liên tục.

Trong thực nghiệm hữu hạn, measure weight chỉ là một estimator cần được kiểm chứng bằng density stress test. Khối lượng ổn định dùng:

\[
\tilde m_i=\operatorname{clip}\left((\rho_i+\epsilon)^{-1},m_{\min},m_{\max}\right),
\qquad
m_i=\frac{\tilde m_i}{\sum_j\tilde m_j+\epsilon}.
\]

Không cho gradient đi qua density estimator trong Prototype 1. Chỉ mở learned mass correction nếu estimator cố định đã chứng minh ích lợi.

### 16.2. Sửa dấu trong boundary-aware transport cost

Cost ban đầu có hạng \(+\eta b_i\), điều này làm điểm biên đắt hơn để vận chuyển và có thể tạo hiệu ứng ngược mục tiêu. Phiên bản đúng phải đưa boundary vào capacity hoặc giảm chi phí giữ điểm biên:

\[
C_{ij}=\alpha\|x_i-q_j\|^2+
\beta(1-n_i^\top\bar n_j)+\gamma|s_i-s_j|
-\eta b_i r_j,
\]

hoặc ưu tiên phương án ổn định hơn:

\[
m_j^{target}\propto 1+\eta\,\bar b_j+\xi\,\bar u_j.
\]

Mọi cost phải được chuẩn hóa robust theo median/MAD và clamp trước Sinkhorn.

### 16.3. OT chỉ được phép trên đồ thị thưa

Dense cost \(C\in\mathbb R^{N_l\times N_{l+1}}\) bị cấm khi \(N_l>4000\). CAMO-Build chỉ cho phép transport trên edge set \(E\):

\[
P_{ij}=0\quad\text{khi }(i,j)\notin E,
\qquad |E|\approx N_l k.
\]

Sparse entropic OT:

\[
P_{ij}=\exp\left(
\frac{f_i+g_j-C_{ij}}{\varepsilon}
\right),\quad(i,j)\in E.
\]

Update dual phải ở log-domain bằng segmented LogSumExp. Không materialize tensor \([N_l,N_{l+1}]\).

Độ phức tạp với \(T\) vòng Sinkhorn:

\[
T_{OT}=O(T|E|),\qquad M_{OT}=O(|E|+N_l+N_{l+1}).
\]

Sparse support có thể làm bài toán không khả thi nếu một target không có cạnh vào hoặc Hall-type support condition bị vi phạm. Bắt buộc:

- tạo edge hai chiều source→target và target→source;
- bảo đảm mỗi node có ít nhất một cạnh;
- thêm fallback/dustbin edges;
- theo dõi marginal residual;
- nếu residual không giảm, chuyển sang unbalanced OT hoặc weighted FPS.

KeOps chỉ giúp tính kernel theo kiểu matrix-free và giảm peak memory; nó **không** biến dense OT từ \(O(NM)\) time thành \(O(N)\). Backend ưu tiên cho sparse prototype là `torch_scatter`/`pyg-lib` hoặc CUDA kernel chuyên biệt. Không dùng Python loop.

### 16.4. Stabilized sparse Sinkhorn

Với dual log-potentials \(f,g\), update dạng:

\[
f_i\leftarrow \varepsilon\left[
\log m_i-
\operatorname{LSE}_{j:(i,j)\in E}
\left(\frac{g_j-C_{ij}}{\varepsilon}\right)
\right],
\]

\[
g_j\leftarrow \varepsilon\left[
\log n_j-
\operatorname{LSE}_{i:(i,j)\in E}
\left(\frac{f_i-C_{ij}}{\varepsilon}\right)
\right].
\]

Quy tắc số học:

- \(m_i,n_j\ge10^{-8}\) trước log;
- tính dual và LogSumExp bằng FP32 dù backbone dùng AMP;
- clamp scaled cost và dual về khoảng hữu hạn;
- kiểm tra `isfinite` mỗi số vòng cố định;
- dừng theo marginal residual, không chỉ số iteration;
- dùng epsilon scaling thay vì bắt đầu với \(\varepsilon\) quá nhỏ.

### 16.5. Cắt giảm objective và xử lý gradient conflict

Không tối ưu đồng thời bảy loss trong prototype đầu.

Lịch khóa cứng:

1. **P1:** \(\mathcal L=\mathcal L_{B\text{-}CE}\), trong đó B-CE là balanced multiclass cross-entropy, không phải binary cross-entropy.
2. **P2:** \(\mathcal L=\mathcal L_{B\text{-}CE}+\lambda_t\mathcal L_{transport}\).
3. **P3:** thêm đúng một objective theo giả thuyết đang kiểm tra.
4. Chỉ sau khi từng loss chứng minh hiệu quả độc lập mới thử multi-objective schedule.

Trước mỗi lần thêm loss, đo cosine similarity giữa gradient của loss mới và loss chính trên shared parameters:

\[
\cos(g_a,g_b)=\frac{g_a^\top g_b}{\|g_a\|\|g_b\|+\epsilon}.
\]

Nếu conflict kéo dài, không tăng số \(\lambda\) thủ công. Chọn một trong ba phương án được ablate riêng: alternating updates, gradient projection, hoặc loại loss phụ.

Transport plan được `detach` trong baseline P2a để đo lợi ích forward trước; P2b mới cho gradient qua số vòng Sinkhorn giới hạn. Điều này tách hiệu quả sampling khỏi chi phí backpropagation OT.

### 16.6. Chống mất tần số cao của global low-rank operator

Low-rank branch không chịu trách nhiệm biểu diễn chi tiết tần số cao. Phân rã bắt buộc:

\[
h^{l+1}=h^l+
\mathcal K_{local}^{high}h^l+
\mathcal K_{global,R}^{low}h^l.
\]

Residual high-frequency được định nghĩa bởi:

\[
h_{high}=h-\Pi_R h,
\]

và không được truyền qua bottleneck hạng thấp. Kiểm tra spectral-energy retention trên graph Laplacian hoặc proxy local variation. Nếu tăng \(R\) không cải thiện validation nhưng tăng latency, dừng ở rank nhỏ; nếu chi tiết mất đi, tăng local bandwidth thay vì chỉ tăng rank.

Router động không được giả định nhanh hơn. Hai chế độ phải được benchmark:

- dynamic scatter/gather;
- static packed top-B patches với shape cố định.

Trên GPU nhỏ, static packed routing là mặc định vì kernel utilization và predictability tốt hơn.

## 17. Hardware gate cho RTX 3050 Ti 4 GB

### 17.1. Dry-run tensor đầu vào `[B=8, N=100k, D=9]`

Số phần tử:

\[
8\times100000\times9=7{,}200{,}000.
\]

Chỉ riêng input:

- FP32: \(7.2M\times4=28.8\) MB, khoảng 27.5 MiB;
- FP16/BF16: \(14.4\) MB, khoảng 13.7 MiB.

Input không phải nút thắt. Feature activations và graph mới là nút thắt:

- `[8,100k,64]` FP16: 102.4 MB cho **một** activation;
- cùng tensor FP32: 204.8 MB;
- kNN indices `[8,100k,16]` int64: 102.4 MB;
- edge features `[8,100k,16,16]` FP16: 409.6 MB cho một tensor;
- autograd có thể giữ nhiều bản sao của các tensor này.

Do đó **B=8, N=100k không được đưa nguyên khối qua backbone trainable** trên 4 GB. Batch 8 chỉ hợp lệ cho coarse representations nhỏ hơn hoặc preprocessing không giữ graph gradient.

### 17.2. Chính sách bộ nhớ bắt buộc

- 100k điểm được giữ trên CPU/pinned memory hoặc xử lý chunked.
- Full-resolution geometric statistics được cache ngoài computational graph.
- GPU coarse branch nhận tối đa khoảng 4k–8k representatives mỗi building ở giai đoạn đầu.
- Full-resolution refinement dùng static packed patches.
- kNN indices ưu tiên int32 nếu backend hỗ trợ.
- Không materialize edge feature tensor bốn chiều; tính message theo fused gather/kernel.
- AMP dùng cho backbone, nhưng mass normalization, reductions và Sinkhorn dual chạy FP32.
- Peak allocated và peak reserved phải được ghi bằng CUDA memory profiler cho mỗi module.
- Hardware gate thất bại nếu peak allocated vượt 3.2 GB; giữ phần còn lại cho CUDA context, fragmentation và workspace.

### 17.3. Gate cho từng module

Mỗi module trước khi merge phải báo cáo:

1. phương trình được tối ưu;
2. time và space Big-O;
3. tensor lớn nhất và ước lượng byte;
4. peak CUDA allocated/reserved thực đo;
5. latency forward/backward;
6. kiểm tra NaN/Inf;
7. kết quả với \(N=1k,4k,8k\), và stress test 100k ở preprocessing/inference path.

Không module nào được phép tạo dense attention hoặc dense OT \(O(N^2)\) khi \(N>4000\).
