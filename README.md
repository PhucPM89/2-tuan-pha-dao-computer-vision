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

Tối học tiếp Sobel và Canny.

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
