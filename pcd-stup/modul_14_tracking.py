import cv2
img=cv2.imread('image.jpg')
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(hsv,(35,50,50),(85,255,255))
cv2.imwrite('mask.jpg',mask)