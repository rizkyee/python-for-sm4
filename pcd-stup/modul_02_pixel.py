import cv2
img=cv2.imread('image.jpg')
print(img.shape)
img[100:200,100:200]=[255,0,0]
cv2.imwrite('output_pixel.jpg',img)