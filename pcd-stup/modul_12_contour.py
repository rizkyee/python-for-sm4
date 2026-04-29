import cv2
img=cv2.imread('image.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,th=cv2.threshold(gray,127,255,0)
cont,_=cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,cont,-1,(0,255,0),2)
cv2.imwrite('contour.jpg',img)