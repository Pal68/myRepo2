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






# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ----------------------------Tamura
#Textures - ------------------------------
# Coded  by  Sudhir  Sornapudi
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# PROGRAM  DESCRIPTION
# # This # program  extracts the Tamura Features(Coarseness, Contrast, Directionality)
# from the segmented histology images of different CIN levels.
# #
# # DATA & FUNCTION
# DICTIONARY
#

import cv2 as cv
import numpy as npy
import math
import scipy

IColor = cv.imread('./tmp/_11piece_metr3.jpg')
I = cv.cvtColor(IColor, cv.COLOR_BGR2GRAY) # Converts RGB image to grayscale

r=I.shape[0]
c =I.shape[1] # size of array
G = I

# # -------------------Coarseness - ------------------

# initialization # Average of neighbouring pixels
A1 = npy.zeros((r,c))
A2 = npy.zeros((r,c))
A3 = npy.zeros((r,c))
A4 = npy.zeros((r,c))
A5 = npy.zeros((r,c))
A6 = npy.zeros((r,c))
# Sbest for coarseness
Sbest = npy.zeros((r,c))
# Subtracting for Horizontal and Vertical case
E1h = npy.zeros((r,c))
E1v = npy.zeros((r,c))
E2h = npy.zeros((r,c))
E2v = npy.zeros((r,c))
E3h = npy.zeros((r,c))
E3v = npy.zeros((r,c))
E4h = npy.zeros((r,c))
E4v = npy.zeros((r,c))
E5h = npy.zeros((r,c))
E5v = npy.zeros((r,c))
E6h = npy.zeros((r,c))
E6v = npy.zeros((r,c))
flag = 0 # To avoid errors

# 2x2 E1h and E1v
# subtracting average of neighbouring 2x2 pixels
for x in range(2,r):
    for y in range(2,c):
        A1[x, y] = (sum(sum(G[x - 1:x, y-1:y])))

for x in  range(2,r - 1):
    for y in range(2,c - 1):
        E1h[x, y] = A1[x + 1, y] - A1[x - 1, y]
        E1v[x, y] = A1[x, y + 1] - A1[x, y - 1]

E1h = E1h / 2 ** (2 * 1)
E1v = E1v / 2 ** (2 * 1)

# 4x4 E2h and E2v
if (r < 4 or c < 4):
    flag = 1

# subtracting average of neighbouring 4x4 pixels
if (flag == 0):
    for x in range(3,r - 1):
        for y in range(3,c - 1):
            A2[x, y] = (sum(sum(G[x - 2:x+1, y-2:y+1])))

for x in range(3,r - 2):
    for y in range(3,c - 2):
        E2h[x, y] = A2[x + 2, y] - A2[x - 2, y]
        E2v[x, y] = A2[x, y + 2] - A2[x, y - 2]

E2h = E2h / 2 ** (2 * 2)
E2v = E2v / 2 ** (2 * 2)

# 8x8 E3h and E3v
if (r < 8 or c < 8):
    flag = 1

# subtracting average of neighbouring 8x8 pixels
if (flag == 0):
    for x in range(5,r - 3):
        for y in range(5,c - 3):
            A3[x, y] = (sum(sum(G[x - 4:x+3, y-4:y+3])))

    for x in range(5,r - 4):
        for y in range(5,c - 4):
            E3h[x, y] = A3[x + 4, y] - A3[x - 4, y]
            E3v[x, y] = A3[x, y + 4] - A3[x, y - 4]

E3h = E3h / 2 ** (2 * 3)
E3v = E3v / 2 ** (2 * 3)

# 16x16 E4h and E4v
if (r < 16 or c < 16):
    flag = 1

# subtracting average of neighbouring 16x16 pixels
if (flag == 0):
    for x in range(9,r - 7):
        for y in range(9,c - 7):
            A4[x, y] = (sum(sum(G[x - 8:x+7, y-8:y+7])))

    for x in range(9,r - 8):
        for y in range(9,c - 8):
            E4h[x, y] = A4[x + 8, y] - A4[x - 8, y]
            E4v[x, y] = A4[x, y + 8] - A4[x, y - 8]

E4h = E4h / 2 ** (2 * 4)
E4v = E4v / 2 ** (2 * 4)

# 32x32 E5h and E5v
if (r < 32 or c < 32):
    flag = 1

# subtracting average of neighbouring 32x32 pixels
if (flag == 0):
    for x in range(17,r - 15):
        for y in range(17,c - 15):
            A5[x, y] = (sum(sum(G[x - 16:x+15, y-16:y+15])))

    for x in range(17,r - 16):
        for y in range(17,c - 16):
            E5h[x, y] = A5[x + 16, y] - A5[x - 16, y]
            E5v[x, y] = A5[x, y + 16] - A5[x, y - 16]

E5h = E5h / 2 ** (2 * 5)
E5v = E5v / 2 ** (2 * 5)

# 64x64 E6h and E6v
if (r < 64 or c < 64):
    flag = 1

# subtracting average of neighbouring 64x64 pixels
if (flag == 0):
    for x in range(33,r - 31):
        for y in range(33,c - 31):
            A6[x, y] = (sum(sum(G[x - 32:x+31, y-32:y+31])))
    for x in range(33,r - 32):
        for y in range(33,c - 32):
            E6h[x, y] = A6[x + 32, y] - A6[x - 32, y]
            E6v[x, y] = A6[x, y + 32] - A6[x, y - 32]

E6h = E6h / 2 ** (2 * 6)
E6v = E6v / 2 ** (2 * 6)

# plots
# figure
# subplot(131)
# imshow(IColor)
# title('Original image')
# subplot(132)
# imshow(E1h)
# title('Horizontal case')
# subplot(133)
# imshow(E1v)
# title('Vertical case')

# at each point pick best size "Sbest", which gives highest output value
for i in range(1,r):
    for j in range(1,c):
        tmp=[abs(E1h[i, j]), abs(E1v[i, j]), abs(E2h[i, j]), abs(E2v[i, j]), abs(E3h[i, j]), abs(E3v[i, j]), abs(E4h[i, j]), abs(E4v[i, j]), abs(E5h[i, j]), abs(E5v[i, j]), abs(E6h[i, j]), abs(E6v[i, j])]
        maxv = max(tmp)
        index=npy.argmax(tmp)
        k = math.floor((index + 1) / 2) # 'k'corresponding to highest E in either direction
        Sbest[i, j] = 2. ** k

# figure
# plot(Sbest)
# title('Output of best size detector')

# Coarseness Value
Fcoarseness = sum(sum(Sbest)) / (r * c)

# #
# -------------------Contrast - ------------------
# #
counts, graylevels = npy.histogram(I,bins=[0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	23,	24,	25,	26,	27,	28,	29,	30,	31,	32,	33,	34,	35,	36,	37,	38,	39,	40,	41,	42,	43,	44,	45,	46,	47,	48,	49,	50,	51,	52,	53,	54,	55,	56,	57,	58,	59,	60,	61,	62,	63,	64,	65,	66,	67,	68,	69,	70,	71,	72,	73,	74,	75,	76,	77,	78,	79,	80,	81,	82,	83,	84,	85,	86,	87,	88,	89,	90,	91,	92,	93,	94,	95,	96,	97,	98,	99,	100,	101,	102,	103,	104,	105,	106,	107,	108,	109,	110,	111,	112,	113,	114,	115,	116,	117,	118,	119,	120,	121,	122,	123,	124,	125,	126,	127,	128,	129,	130,	131,	132,	133,	134,	135,	136,	137,	138,	139,	140,	141,	142,	143,	144,	145,	146,	147,	148,	149,	150,	151,	152,	153,	154,	155,	156,	157,	158,	159,	160,	161,	162,	163,	164,	165,	166,	167,	168,	169,	170,	171,	172,	173,	174,	175,	176,	177,	178,	179,	180,	181,	182,	183,	184,	185,	186,	187,	188,	189,	190,	191,	192,	193,	194,	195,	196,	197,	198,	199,	200,	201,	202,	203,	204,	205,	206,	207,	208,	209,	210,	211,	212,	213,	214,	215,	216,	217,	218,	219,	220,	221,	222,	223,	224,	225,	226,	227,	228,	229,	230,	231,	232,	233,	234,	235,	236,	237,	238,	239,	240,	241,	242,	243,	244,	245,	246,	247,	248,	249,	250,	251,	252,	253,	254,	255
], range=(0,255)) # histogram of image
graylevels=graylevels[0:255]
# # figure
# imhist(I)
# title('Gray-level distribution')
PI = counts / (r * c)
averagevalue = sum(graylevels * PI) # mean value
u4 = sum((graylevels - npy.tile(averagevalue, [256, 1])) ** 4. * PI) # 4 th moment about mean
variance = sum((graylevels - npy.tile(averagevalue, [256, 1])) ** 2. * PI) # variance(2nd moment about mean)
alpha4 = u4 / variance ** 2 # kurtosis

# Contrast Value
Fcontrast = npy.sqrt(variance) / alpha4 ** (1 / 4)

# #
# -------------------Directionality - ------------------
# #
PrewittH = [[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]] # for measuring horizontal differences
PrewittV = [[1, 1, 1],[0, 0, 0],[-1, - 1, - 1]] # for measuring vertical differences

# Applying PerwittH operator
deltaH = npy.zeros((r, c))
for i in range(2,r-1 ):
    for j in range(2,c-1):
        deltaH[i, j] = sum(sum(G[i - 1:i + 2, j - 1:j + 2]*PrewittH))

# Modifying borders
for j in range(2,c - 1):
    deltaH[1, j] = G[1, j + 1] - G[1, j]
    deltaH[r-1, j] = G[r-1, j + 1] - G[r-1, j]

for i in (1,r-1):
    deltaH[i, 1] = G[i, 2] - G[i, 1]
    deltaH[i, c-1] = G[i, c-1] - G[i, c - 2]


# Applying PerwittV operator
deltaV = npy.zeros((r, c))
for i in range(2,r - 1):
    for j in range(2,c - 1):
        deltaV[i, j] = sum(sum(G[i - 1:i + 2, j - 1: j + 2]*PrewittV))

# Modifying borders
for j in range(1,c):
    deltaV[1, j] = G[2, j] - G[1, j]
    deltaV[r-1, j] = G[r-1, j] - G[r - 2, j]

for i in range(2,r - 1):
    deltaV[i, 1] = G[i + 1, 1] - G[i, 1]
    deltaV[i, c-1] = G[i + 1, c-1] - G[i, c-1]

# Magnitude
deltaG = (abs(deltaH) + abs(deltaV)) / 2

# Local edge direction(0 <= theta < pi)
theta = npy.zeros((r, c))
for i in range(1,r):
    for j in range(1,c):
        if (deltaH[i, j] == 0) and (deltaV[i, j] == 0):
            theta[i, j] = 0
        elif deltaH[i, j] == 0:
            theta[i, j] = math.pi
        else:
            theta[i, j] = math.atan(deltaV[i, j] / deltaH[i, j]) + math.pi / 2


deltaGt = deltaG[:]
theta1 = theta[:]

# Set a Threshold value for delta G
n = 16
HD = npy.zeros((n))
Threshold = 12
counti = 0
for m in range(0,(n - 1)):
    countk = 0
    for k in range(1,len(deltaGt)):
        if ((all(deltaGt[k] >= Threshold)) and (theta1[k] >= (2 * m - 1) * math.pi / (2 * n)) and (theta1[k] < (2 * m + 1) * math.pi / (2 * n))):
            countk = countk + 1
            counti = counti + 1
    HD[m+ 1] = countk

HDf = HD / counti
# figure
# plot(HDf)
# title('Local Directionality Histogram HDf')
# peakdet function to find peak values
m,p = scipy.signal.find_peaks(HDf, 0.000005)
Fd = 0
for np in range(1,len(m)):
    phaiP = m(n) * (math.pi / n)
    for phi in range(1,len(HDf)):
        Fd = Fd + (phi * (math.pi / n) - phaiP) ** 2 * HDf(phi)

r = (1 / n)
Fdirection = 1 - r * n * Fd

print(Fcoarseness, Fcontrast, Fdirection)


