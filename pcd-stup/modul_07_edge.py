import cv2
img=cv2.imread('image.jpg',0)
edges=cv2.Canny(img,100,200)
cv2.imwrite('edges.jpg',edges)