import cv2,numpy as np
img=cv2.imread('image.jpg',0)
_,th=cv2.threshold(img,127,255,0)
k=np.ones((5,5),np.uint8)
eros=cv2.erode(th,k)
cv2.imwrite('erosion.jpg',eros)