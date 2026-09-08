import cv2
import numpy as np

mask = np.zeros((240, 320), dtype=np.uint8)
mask[30:100, 40:120] = 255
mask[130:210, 180:290] = 255
mask[10:13, 10:13] = 255

contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
count = 0

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 100:
        continue

    x, y, w, h = cv2.boundingRect(contour)
    count += 1

    cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1),
                  (0, 255, 0), 2)

cv2.imwrite("contours.png", output)
print("So vung giu lai:", count)  # 2