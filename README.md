# 2 tuần phá đảo Computer Vision

Mình bắt đầu repo này để học Computer Vision trong 14 ngày, dự kiến dành 10 tiếng mỗi ngày. Trước đó mình đã học FastAPI và biết sơ qua IoU, recall, nhưng chưa có nền tảng xử lý ảnh.

Mục tiêu là hiểu được từ dữ liệu ảnh đến mô hình, rồi làm một project kiểm lỗi sản phẩm có thể tự giải thích và demo khi ứng tuyển. Repo lưu bài tập và ghi chép theo tiến độ thực tế của mình.

## Ngày 1 - 07/09/2026

Hôm nay bắt đầu với cách biểu diễn ảnh bằng NumPy. Đã làm xong hai bài về tạo ảnh và chỉnh sáng một vùng ảnh.

### Những gì đã học

- Ảnh màu có shape `(H, W, C)`. Truy cập pixel bằng `image[y, x]`, hàng trước, cột sau.
- Với ảnh BGR, `[0, 0, 255]` là đỏ, `[0, 255, 0]` là xanh lá. Chỉ nhìn shape thì không biết thứ tự kênh màu.
- `uint8` chứa giá trị từ 0 đến 255, mỗi phần tử chiếm 1 byte. Ảnh `(480, 640, 3)` chiếm `480 * 640 * 3 = 921600` byte dữ liệu, tức 900 KiB. Dung lượng này khác dung lượng file PNG/JPEG.
- Cộng trực tiếp trên mảng `uint8` có thể tràn số: `255 + 10` thành `9`. Muốn tăng sáng an toàn thì đổi sang kiểu rộng hơn, cộng, clip rồi chuyển về `uint8`.
- Basic slicing như `image[40:120, 60:180]` tạo view dùng chung dữ liệu. Sửa vùng này có thể làm ảnh gốc thay đổi.
- `.copy()` tạo bản sao độc lập. `new_image = image` chỉ thêm một tên tham chiếu tới cùng mảng.
- Với ROI cắt bước 1, tọa độ trong ảnh gốc bằng tọa độ trong ROI cộng offset của góc trên trái.

Lúc đầu mình nhầm `[0, 0, 255]` là xanh lá và thay nhầm kích thước khi tính dung lượng ảnh. Công thức đúng nhưng vẫn phải kiểm tra lại dữ liệu đầu vào.

### Bài đã làm

**[exercise_01.py](exercise_01.py)**: tạo ảnh đen 240 x 320, tô hình chữ nhật đỏ bằng slicing và lưu PNG.

```text
Shape của vùng hình chữ nhật: (80, 120, 3)
Pixel tại (y=50, x=70): [  0   0 255]
```

**[exercise_02.py](exercise_02.py)**: viết hàm `brighten_roi` tăng hoặc giảm sáng trong ROI, giữ nguyên ảnh đầu vào.

Các bước mình dùng: copy ảnh, chuyển ROI sang `int16`, cộng `delta`, clip về `[0, 255]`, chuyển về `uint8` và ghi lại vào ảnh kết quả. Bài này giả sử tọa độ hợp lệ và `delta` là số nguyên trong `[-255, 255]`.

| Pixel ban đầu | Delta | Kết quả đã kiểm tra |
| --- | --- | --- |
| `[250, 100, 0]` | 20 | `[255, 120, 20]` |
| `[10, 100, 255]` | -30 | `[0, 70, 225]` |
| `[10, 100, 255]` | 0 | `[10, 100, 255]` |

Đã kiểm tra ảnh gốc và vùng ngoài ROI không đổi, đầu ra giữ shape, dtype và không dùng chung bộ nhớ với đầu vào.

Điểm cần nhớ: `.astype(np.int16)` ở đây tạo mảng mới. Nếu chỉ sửa `roi` mà quên ghi lại vào `result`, ảnh trả về sẽ chưa được chỉnh sáng.

### Mask và broadcasting

- Boolean mask chọn pixel theo điều kiện. Mask có shape `(H, W)`, kết hợp điều kiện bằng `&` và bọc từng điều kiện trong ngoặc.
- `image[mask]` lấy ra mảng `(N, C)` với N pixel được chọn, không giữ bố cục ảnh. Đọc bằng boolean indexing tạo bản sao, còn `image[mask] = ...` ghi trực tiếp vào ảnh.
- Broadcasting cho phép gán một bộ ba màu vào cả vùng ảnh hoặc nhân từng kênh với một hệ số riêng.

## Ngày 2 - 08/09/2026

Hôm nay học tiếp từ mask đến pipeline tách vùng và đếm vật thể. Mình đã học xong mask, broadcasting, HSV, histogram, threshold, lọc ảnh, morphology và contour.

### Những điểm cần nhớ

- Mask chọn pixel theo điều kiện, ROI chọn vùng theo tọa độ. `mask[..., None]` thêm chiều kênh để nhân mask `(H, W, 1)` với ảnh `(H, W, 3)`.
- HSV tách sắc màu, độ bão hòa và mức sáng. Với ảnh `uint8` chuyển bằng `COLOR_BGR2HSV`, H nằm trong 0-179, S và V trong 0-255. Màu đỏ nằm ở hai đầu vòng H nên thường cần ghép hai khoảng.
- `cv2.inRange` trả mask 0/255. Đếm pixel được chọn bằng `np.count_nonzero`, không lấy tổng giá trị mask làm số pixel.
- Histogram đếm số pixel ở từng mức sáng, không lưu vị trí của chúng. Hai ảnh khác nhau vẫn có thể có cùng histogram.
- Threshold cố định dùng một ngưỡng mình chọn. Otsu tự chọn một ngưỡng chung từ histogram. Adaptive threshold tính ngưỡng cục bộ, hữu ích khi ánh sáng không đều.
- Mean lấy trung bình, Gaussian dùng trọng số ưu tiên gần tâm, median lấy trung vị. Lọc mạnh có thể xóa cả chi tiết lỗi nhỏ, không chỉ nhiễu.
- Erosion co vùng trắng, dilation nở vùng trắng. Opening là erosion rồi dilation; closing là dilation rồi erosion. Phải biết vùng nào cần giữ trước khi làm sạch mask.
- Contour biểu diễn đường bao bằng các điểm. `RETR_EXTERNAL` lấy biên ngoài; `boundingRect` trả `(x, y, w, h)` để vẽ khung hoặc cắt ROI.
- Diện tích contour là diện tích hình học, không nhất thiết bằng số pixel trắng. Đếm contour cũng không mặc nhiên là đếm đúng vật thể ngoài đời.

### Bài đã làm

**[practice.py](practice.py)**: tạo mask có hai hình chữ nhật và một vùng nhiễu nhỏ, tìm contour, lọc diện tích và vẽ bounding box. Kết quả giữ lại 2 vùng.

**[exercise_03.py](exercise_03.py)**: tạo hai vật thể tối trên nền sáng, thêm nhiễu rồi chạy pipeline:

```text
Ảnh grayscale -> Otsu đảo -> Opening -> Contour -> Lọc diện tích -> Đếm vùng
```

Thử thêm một cầu nối giữa hai vật thể để so sánh:

```text
separate: 2 vung
connected: 1 vung
```

Điểm mình rút ra: hai vật thể nối nhau thành một vùng trắng thì contour ngoài chỉ còn một. Khi đếm sai, cần xem mask trước. Tăng kernel để phá cầu nối cũng có thể phá mất chi tiết cần giữ.

### Sobel, Canny và kiểm tra lý thuyết

Đã học xong Sobel và Canny, sau đó làm bài kiểm tra lý thuyết hai ngày đầu.

- Sobel đo gradient: Gx đo thay đổi theo chiều ngang nên phản ứng mạnh với biên đứng. Dùng kiểu như `float32` để giữ gradient âm và giá trị lớn hơn 255.
- Canny làm mảnh biên và dùng hai ngưỡng gradient. Biên yếu chỉ được giữ khi nối được với biên mạnh qua các ứng viên phù hợp.
- Biên có thể đến từ bóng đổ, chữ in hoặc texture; ảnh Canny chưa phải mask đặc của vật thể.
- Qua bài kiểm tra, mình cần giải thích kỹ hơn cơ chế broadcasting và sự khác nhau giữa diện tích contour với số pixel trắng.
- Tăng ngưỡng diện tích chỉ giữ hoặc loại contour, không tách một contour thành hai. Muốn đếm hai vật thể đang nối nhau trong pipeline hiện tại, phải xử lý vùng nối trước.

## Ngày 3 - 09/09/2026

Hôm nay học về dữ liệu, đánh giá mô hình và ôn lại cơ chế training. Trước đó mình từng làm bài nhận diện chữ số, tự định nghĩa model và training loop; lần này học lại để nối các khái niệm với bài toán kiểm lỗi.

### Dữ liệu và nhãn

- Classification gán nhãn toàn ảnh; detection dùng box; segmentation gán nhãn từng pixel. Multiclass chọn một lớp, multilabel cho phép nhiều nhãn cùng lúc.
- Anomaly detection học đặc điểm bình thường để tìm khác biệt. Khác thường chưa chắc là lỗi; vẫn cần dữ liệu có nhãn để đánh giá.
- Chia train/validation/test theo đơn vị cần tổng quát: sản phẩm, video, lô hoặc thời gian. Không random các frame gần giống nhau vào nhiều tập.
- Train dùng cập nhật trọng số; validation dùng chọn cấu hình, checkpoint và ngưỡng; test đánh giá phương án đã chốt.
- Chia tập trước augmentation. Nếu tự tính mean/std, chỉ tính từ train. Ảnh gần trùng hoặc patch cùng sản phẩm không nên lọt giữa các tập khi đánh giá trên sản phẩm mới.

### Metric và ngưỡng quyết định

Quy ước positive là sản phẩm lỗi: TP là phát hiện đúng lỗi, FP là báo nhầm hàng tốt, FN là bỏ sót lỗi, TN là nhận đúng hàng tốt.

```text
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

- Accuracy cao vẫn có thể bỏ sót toàn bộ lỗi khi dữ liệu mất cân bằng.
- Thay ngưỡng chỉ đổi cách chuyển score thành nhãn, không làm trọng số học tốt hơn. Hạ ngưỡng không làm recall giảm trên cùng dữ liệu và score cố định, nhưng precision không nhất thiết giảm ở mọi bước.
- PR curve cho thấy precision và recall qua nhiều ngưỡng. F1 cao nhất chưa chắc đáp ứng chi phí vận hành tốt nhất.
- Chọn ngưỡng bằng validation rồi mới đánh giá test. Luôn xem số mẫu lỗi thực tế, không chỉ phần trăm.

### Overfitting và mất cân bằng

- Train loss giảm nhưng validation loss tăng là dấu hiệu cần kiểm tra overfitting; khoảng cách cũng có thể do lệch dữ liệu hoặc tiền xử lý sai.
- Early stopping theo dõi tiêu chí validation và lưu checkpoint tốt nhất, không mặc định lấy epoch cuối.
- Oversampling tăng tần suất gặp lớp lỗi; undersampling giảm mẫu lớp nhiều; weighted loss tăng đóng góp của lớp cần ưu tiên. Lặp ảnh không tạo thêm thông tin mới.
- Không mặc định cân bằng validation/test như batch train. Precision phụ thuộc tỷ lệ lỗi trong dữ liệu sử dụng.

### Tiền xử lý và augmentation

- Phân biệt chia 255 với chuẩn hóa `(x - mean) / std`. Sau chuẩn hóa, giá trị âm hoặc lớn hơn 1 là bình thường.
- Train và API phải khớp thứ tự kênh, kích thước, thang giá trị và mean/std. Với pretrained model, theo cấu hình tiền xử lý của trọng số.
- Resize quá nhỏ có thể làm mất vết xước. Có thể cân nhắc ROI hoặc patch nhưng phải giữ cách chia tập độc lập.
- Crop mất hết lỗi mà vẫn giữ nhãn lỗi tạo label noise: input và label mâu thuẫn.
- Biến đổi hình học phải đồng bộ với box/mask. Resize mask mã lớp thường dùng nearest-neighbor để không sinh mã lớp không hợp lệ.

### Baseline và thí nghiệm

Baseline là mốc so sánh có thể chạy và đánh giá lại. Khi thử cải tiến, ghi giả thuyết, cấu hình, split, seed, metric và tốc độ. Thay từng yếu tố giúp tìm nguyên nhân; các thay đổi có thể tương tác khi kết hợp.

Mình đã phân tích tình huống F1 tăng từ 0,80 lên 0,81 nhưng inference chậm gấp ba. Cần xem precision/recall, FP/FN, loại lỗi được cải thiện, độ ổn định qua seed và yêu cầu latency/throughput, bộ nhớ.

Chỗ mình cần sửa: ban đầu muốn xem test để quyết định thay baseline. Việc chọn model phải dựa vào validation và yêu cầu vận hành; test dành cho đánh giá cuối. Nếu tiếp tục chỉnh theo test thì tập đó không còn độc lập cho phiên bản sau.

### Model, loss và training loop

- Model tính đầu ra từ dữ liệu và tham số. Với mô hình tuyến tính nhị phân, `z = w*x + b` là logit; sigmoid chuyển logit thành score trong khoảng 0-1.
- BCE phạt dự đoán sai tự tin mạnh hơn. `BCEWithLogitsLoss` nhận logits trực tiếp, không thêm sigmoid trước loss này.
- `backward()` tính và lưu gradient; `optimizer.step()` mới cập nhật tham số. `zero_grad()` xóa gradient cũ vì PyTorch mặc định cộng dồn gradient.
- `model.eval()` thay đổi hành vi các lớp như Dropout/BatchNorm, không tự tắt tính gradient.
- Batch là nhóm mẫu; epoch là một lượt qua dữ liệu; step là lần cập nhật. Với 1.000 mẫu, batch 100 và một lần cập nhật mỗi batch: 10 step/epoch.
- Loss batch có thể là trung bình loss các mẫu. Batch lớn hơn không mặc định tốt hơn và có thể cần chỉnh learning rate.
- Shuffle trộn thứ tự, không tự cân bằng lớp. Gradient accumulation tích lũy qua nhiều batch trước khi step, cần điều chỉnh thang loss phù hợp.

Buổi học dừng ở batch, epoch, shuffle và gradient accumulation. Phần tiếp theo sẽ tiếp tục nền tảng mô hình và CNN.

## Chạy bài

Môi trường hiện tại: Python 3.12, NumPy và OpenCV. Chạy từ thư mục repo:

```powershell
python -m pip install numpy opencv-python
python exercise_01.py
python practice.py
python exercise_03.py
```

Các bài tạo ảnh PNG trong thư mục đang chạy: bài 1 tạo `exercise_01.png`, bài contour tạo `contours.png`, bài 3 tạo các ảnh `separate_*` và `connected_*` để so sánh từng bước. Bài 2 hiện chỉ định nghĩa hàm, có thể thử bằng:

```python
import numpy as np
from exercise_02 import brighten_roi

image = np.array([[[250, 100, 0]]], dtype=np.uint8)
result = brighten_roi(image, 0, 0, 1, 1, 20)
print(result[0, 0])  # [255 120  20]
print(image[0, 0])   # [250 100   0]
```
