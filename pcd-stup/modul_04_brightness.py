import cv2
img=cv2.imread('image.jpg')
bright=cv2.convertScaleAbs(img,alpha=1,beta=50)
cv2.imwrite('bright.jpg',bright)