import numpy as np
import cv2
img = cv2.imread("./tmp/new2.bmp", cv2.IMREAD_GRAYSCALE)
filtered = cv2.bilateralFilter(img, 9, 75, 75)
edges = cv2.Canny(filtered, 50, 200, apertureSize = 3)
cv2.imwrite('./tmp/1/edges.jpg',edges)
lines = cv2.HoughLines(edges, 3, np.pi/180, 200)

for i in range(len(lines[:, 0, 0])):
	for rho, theta in lines[i]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
		cv2.imwrite('./tmp/1/hough.jpg',img)
