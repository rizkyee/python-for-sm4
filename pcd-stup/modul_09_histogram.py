import cv2
img=cv2.imread('image.jpg',0)
eq=cv2.equalizeHist(img)
cv2.imwrite('hist.jpg',eq)