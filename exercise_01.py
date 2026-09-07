import cv2
import numpy as np

image = np.zeros((240,320,3), dtype=np.uint8)

image[40:120, 60:180] = [0,0,255]

rectangle = image[40:120, 60:180]

cv2.imwrite("exercise_01.png", image)

print("Shape của vùng hình chữ nhật:", rectangle.shape)
print("Pixel tại (y=50, x=70):", image[50, 70])