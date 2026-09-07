import numpy as np

def brighten_roi(image, x1, y1, x2, y2, delta):
    result =  image.copy()
    roi = result[y1:y2, x1:x2].astype(np.int16)
    roi = roi + delta
    roi = np.clip(roi,0,255)
    result[y1:y2, x1:x2] = roi.astype(np.uint8)
    return result
