import cv2, matplotlib.pyplot as plt
img=cv2.imread('image.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img);plt.axis('off')