from scipy.stats import uniform
from skimage import feature
import numpy as np
import cv2
import matplotlib.pyplot as plt
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist

def getCOSSimilarity(v1,v2):
    result = np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
    return result
#end getCOSSimilarity

img1 = cv2.imread('./tmp/_9piece_metr6.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('./tmp/_11piece_metr6.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.imread('./tmp/_12piece_metr6.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

desc=LocalBinaryPatterns(32,4)
hist1=desc.describe(img1,100)
hist2=desc.describe(img2,100)
hist3=desc.describe(img3,100)
s1_2=getCOSSimilarity(hist1,hist2)
s1_3=getCOSSimilarity(hist1,hist3)
s2_3=getCOSSimilarity(hist2,hist3)

print(s1_2,s1_3,s2_3)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img1, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('LBP1')
lbp1=feature.local_binary_pattern(img1,32,4, method="uniform")
plt.imshow(lbp1, cmap='gray')
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img2, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('LBP1')
lbp2=feature.local_binary_pattern(img2,32,4, method="uniform")
plt.imshow(lbp2, cmap='gray')
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img3, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('LBP1')
lbp3=feature.local_binary_pattern(img3,32,4, method="uniform")
plt.imshow(lbp3, cmap='gray')
plt.show()
#print(hist1)