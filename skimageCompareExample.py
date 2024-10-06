from skimage import feature
import numpy as np
import cv2
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

img1 = cv2.imread('./tmp/_7piece_metr6_135.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('./tmp/_9piece_metr6.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.imread('./tmp/_11piece_metr6_135.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

desc=LocalBinaryPatterns(255,8)
hist1=desc.describe(img1)
hist2=desc.describe(img2)
hist3=desc.describe(img1)
s1_2=getCOSSimilarity(hist1,hist2)
s1_3=getCOSSimilarity(hist1,hist3)
s2_3=getCOSSimilarity(hist2,hist2)

print(s1_2,s1_3,s2_3)
print(hist1)