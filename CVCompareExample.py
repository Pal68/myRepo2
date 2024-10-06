import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('./tmp/_7piece_metr6_135.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
h, w = img1.shape

img2 = cv2.imread('./tmp/_9piece_metr6.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.imread('./tmp/_11piece_metr6_135.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

def error(img1, img2):
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   msre = np.sqrt(mse)
   return mse, diff

match_error12, diff12 = error(img1, img2)
match_error13, diff13 = error(img1, img3)
match_error23, diff23 = error(img2, img3)

print("Image matching Error between image 1 and image 2:",match_error12)
print("Image matching Error between image 1 and image 3:",match_error13)
print("Image matching Error between image 2 and image 3:",match_error23)

plt.subplot(221), plt.imshow(diff12,'gray'),plt.title("image1 - Image2"),plt.axis('off')
plt.subplot(222), plt.imshow(diff13,'gray'),plt.title("image1 - Image3"),plt.axis('off')
plt.subplot(223), plt.imshow(diff23,'gray'),plt.title("image2 - Image3"),plt.axis('off')
plt.show()