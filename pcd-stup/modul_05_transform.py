import cv2
img=cv2.imread('image.jpg')
res=cv2.resize(img,(300,300))
cv2.imwrite('resize.jpg',res)