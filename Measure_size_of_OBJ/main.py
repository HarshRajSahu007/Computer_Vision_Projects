import cv2
import numpy as np

img = cv2.imread('/Users/harshrajsahu/Desktop/Computer_Vision_projects/Computer_Vision_Projects/Measure_size_of_OBJ/Harsh_image.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


ret,thresh = cv2.threshold(
    gray, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU
)

contours,hierarchy = cv2.findContours(
    thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:
    area = cv2.contourArea(cnt)

    x,y,w,h = cv2.boundingRect(cnt)

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img,str(area),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()