# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`= 1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
   Trả lời: Nói về cả bức ảnh.
   Lý do: Đây là bài toán phân loại ảnh (Image Classification). Mô hình đưa ra nhãn tổng quan cho toàn bộ bức ảnh và đầu ra không có tọa độ bounding box để chỉ định từng vật thể riêng biệt.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
   Trả lời: Đi kèm theo taxonomy/checkpoint của mô hình (cụ thể là ImageNet-1K với 1.000 lớp cố định). Mô hình không thể tự nghĩ ra tên nhãn mới.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?  
   Trả lời: Để định danh duy nhất và tránh nhập nhằng nhãn. Tên lớp (class_name) dễ trùng nghĩa giữa các bộ dữ liệu; taxonomy_name xác định tập danh mục chuẩn đang dùng; còn class_id giúp máy định vị chính xác chỉ số của lớp đó.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
   Trả lời: Guideline cần quy định tiêu chí chọn nhãn duy nhất (ví dụ: ưu tiên vật thể chiếm diện tích lớn nhất hoặc nằm ở trung tâm/tiêu điểm chính) và có quy trình báo lên cấp trên (escalation) khi không xác định được chủ thể.
6. Vì sao không được lấy `score` của model làm ground truth?  
  Trả lời: Vì score chỉ là độ tự tin (xác suất dự đoán) của mô hình chứ không phải sự thật (mô hình có thể tự tin 99% nhưng vẫn đoán sai). Ground truth bắt buộc phải do con người kiểm duyệt và gán nhãn chuẩn theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`= 0.912625 ; `bbox_xyxy`= [385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
   Góc trên-trái: Tọa độ (385.33, 69.24) px — Nằm ở vùng trên bên phải ảnh.Góc dưới-phải: Tọa độ (498.92, 348.92) px — Nằm ở vùng giữa bên phải ảnh.Kích thước: Rộng 113.58 px, cao 279.68 px.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  
   Ngưỡng high (0.60): Giữ lại 6 predictions (chỉ các vật thể độ tin cậy cao: người, bát, lò nướng).
   Ngưỡng low (0.20): Số predictions tăng lên đáng kể (khoảng 14–16 predictions do giữ thêm các vật thể nhỏ/mờ như cốc, thìa, chai).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
   Reviewer phải tốn thêm thời gian lọc/xóa các khung đoán sai (False Positives), loại bỏ các khung bị lặp/chồng chéo lên cùng một vật thể, và kiểm tra lại những khung có điểm tự tin thấp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
   Khung bao (bounding box) phải ôm sát các viền ngoài cùng của phần vật thể nhìn thấy được, tuyệt đối không cắt lẹm vào vật thể và không để chừa quá 3 pixel khoảng trống thừa ở mỗi cạnh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
   Vật thể phát hiện: Người ở mép trái ảnh ($x_{min} \approx 0$) bị khung hình cắt mất một phần thân.Xử lý: Viết thành quy tắc trong Guideline chứ không cần hỏi Lab Coach (escalation).Lý do: Đây là tình huống rất phổ biến. Guideline chỉ cần quy định: "Vẽ khung ôm sát phần nhìn thấy được chạm sát viền ảnh và bật thuộc tính truncated=True".

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= person ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
   Trả lời: So với box chỉ là một khung chữ nhật bao ngoài, polygon cung cấp đường bao thực tế theo từng đường cong, góc cạnh và hình dáng giải phẫu của đối tượng. Nhờ đó ta biết chính xác từng điểm ảnh (pixel) nào thuộc về vật thể và pixel nào thuộc về hậu cảnh/nền, loại bỏ được hoàn toàn diện tích nền thừa trong hộp.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
   Trả lời: `instance_id` dùng để phân biệt và định danh riêng từng cá thể/vật thể độc lập trong cùng một bức ảnh (ví dụ người thứ nhất là kitchen-001, người thứ hai là kitchen-009). 
   - Nó không phải `class_id` vì nhiều cá thể có thể cùng chung một lớp (đều là person có class_id=0).
   - Nó không phải `tracking_id` vì nó chỉ có giá trị phân biệt cục bộ trong duy nhất bức ảnh tĩnh hiện tại, không có ý nghĩa liên kết hay theo dõi cùng đối tượng đó qua chuỗi các khung hình video tiếp theo.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
   Trả lời: "Vẽ đa giác (polygon) bám khít theo đường viền thực tế của vật thể nhìn thấy, không đưa phần nền hoặc bóng đổ của vật thể vào trong mask; đối với các cạnh thẳng hoặc viền cong nhẹ, khoảng cách giữa các đỉnh không quá xa để giữ đúng hình dáng vật thể."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
   Trả lời: Vị trí chiếc bát (`bowl`) đặt trên bàn hoặc người đứng tựa/chạm sát cạnh bàn bếp có vùng tiếp xúc bị bóng tối che khuất, làm đường viền phân tách giữa hai vật thể bị mờ. Tôi sẽ dùng guideline: "Trường hợp hai vật thể chạm nhau hoặc vật này che một phần vật kia, ưu tiên vẽ trọn vẹn đường biên của vật nằm ở lớp trước (gần camera hơn); phần tiếp giáp mờ thì ước lượng dựa theo đường cong tự nhiên của vật thể lớp trước." Quy tắc này đủ rõ ràng để xử lý đa số các ca va chạm thông thường, không cần làm phiền Lab Coach trừ khi hoàn toàn không thể nhận diện được hình dạng vật thể.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn danh mục duy nhất cho cả ảnh (vd: `cab`, `class_id=468`). | Ảnh `traffic` có nhiều loại xe khác nhau (taxi, minibus, car), mô hình chỉ gán `cab` với score thấp 0.51; ảnh `kitchen` mô hình lại gán nhãn `gong` (score 0.42) do nhầm cái chảo/khay treo. | Quan sát toàn bộ bức ảnh và chọn đúng 1 nhãn theo quy tắc đối tượng chính trong guideline. | Kiểm tra xem nhãn được chọn có nằm trong taxonomy quy định và có đúng quy tắc ưu tiên chủ thể hay không. |
| Phát hiện vật thể | Danh sách các hộp toạ độ pixel `[x_min, y_min, x_max, y_max]` kèm tên lớp cho từng vật thể. | Người ở góc trái `kitchen` bị mép ảnh cắt (`x_min=0.12`), các chiếc bát nhỏ và cốc ở xa có score thấp (~0.38-0.45) dễ bị sót hộp. | Kéo hộp chữ nhật ôm sát từng vật thể nhìn thấy, đánh dấu cờ nếu vật bị cắt mép hoặc che khuất. | Kiểm tra độ khít của các bounding box, rà soát xem có bỏ sót vật thể nhỏ nào không và xoá các box thừa. |
| Instance segmentation | Tập hợp các toạ độ điểm khép kín `polygon_xy` tạo thành mask độc lập cho từng cá thể (`instance_id`). | Người `kitchen-001` có viền phức tạp (348 điểm), phần chân và thân chạm vào tủ/bàn bếp khó tách bạch viền chính xác; thìa/dĩa nhỏ rất khó phân đoạn. | Dùng công cụ chấm điểm đa giác bao sát từng đường nét của từng cá thể đối tượng riêng lẻ. | Soi kỹ đường viền mask xem có bị lẹm vào nền hoặc chồng chéo sai lệch giữa các vật thể hay không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
   Trả lời: Chỉ sử dụng dữ liệu hình ảnh được cung cấp cho mục đích học tập trong khuôn khổ bài lab, tuyệt đối không tự ý sao chép, phát tán ra ngoài hoặc đưa thông tin nhận dạng cá nhân vào báo cáo và repository.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
   Trả lời: Tôi sẽ dừng thao tác trên tệp dữ liệu đó ngay lập tức và báo cáo trực tiếp cho **Lab Coach / Giảng viên hướng dẫn** để được xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
