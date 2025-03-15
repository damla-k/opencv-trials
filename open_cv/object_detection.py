import cv2
import numpy as np

base = cv2.imread('ImageHandler.jpeg',0)
basketball = cv2.imread('basketball.png',0)
base_image_2 = base.copy()
h, w = basketball.shape
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
for method in methods: 
    base2 = base.copy()

    result = cv2.matchTemplate(base2, basketball, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    bottom_right = (location[0] + w, location[1] + h)    
    cv2.rectangle(base2, location, bottom_right, 255, 5)
    cv2.imshow('Match', base2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()