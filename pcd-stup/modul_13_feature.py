import cv2
img=cv2.imread('image.jpg',0)
_,th=cv2.threshold(img,127,255,0)
cont,_=cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in cont:
 print(cv2.contourArea(c))