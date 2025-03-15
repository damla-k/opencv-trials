import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read(0)
    width = int(cap.get(3))
    height = int(cap.get(4))

    #img = cv2.line(frame, (0,0),(width,height),(255,0,0), 10)
    #img = cv2.line(img, (0,height),(width,0),(0,255,0), 5)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)

    result = cv2.bitwise_and(frame,frame,mask=mask)
    
    cv2.imshow('frame', result)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



