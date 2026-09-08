import cv2
import numpy as np

def count_objects(gray):
    #Vật thể tối thanh trắng, nền sáng thanh đen
    _, mask = cv2.threshold(gray, 0 , 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    count  = 0
    
    for contour in contours: 
        if cv2.contourArea(contour) < 100:
            continue
        
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x,y), (x+w-1, y+h-1), (0,255,0),2)
        count +=1
        
    return mask, cleaned, output, count

#Hai vật theer tách rời
separate  = np.full((200,300), 230, dtype=np.uint8)
separate[50:150, 40:110] = 30
separate[50:150, 180:250] = 30

#Nhiễu tối nhỏ
separate[20,20] = 30
separate[175:177, 270:272] = 30

#Thêm cầu nối giữa 2 vật thể
connected = separate.copy()
connected[90:110, 110:180] = 30

for name, gray in [("separate", separate), ("connected", connected)]:
    mask, cleaned, output, count = count_objects(gray)
    
    cv2.imwrite(f"{name}_input.png", gray)
    cv2.imwrite(f"{name}_mask.png", mask)
    cv2.imwrite(f"{name}_cleaned.png", cleaned)
    cv2.imwrite(f"{name}_result.png", output)

    print(f"{name}: {count} vung")

