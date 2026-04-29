import cv2
img=cv2.imread('image.jpg',0)
_,th=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('segment.jpg',th)