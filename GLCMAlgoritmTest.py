# Convert image to grayscale and normalize pixel values image_gray = np.dot(image[...,:3], [0.299,0.587,0.114])
import cv2
import numpy as np
image_gray = cv2.imread("./tmp/_1piece_metr6.jpg", cv2.IMREAD_GRAYSCALE)
image_gray = image_gray /255.0 # Define the co-occurrence matrix

def get_glcm_matrix(image):
    G = 256 # number of gray levels
    glcm_matrix = np.zeros((G, G))
 # Calculate the co-occurrence matrix
    d = 1
    for i in range(image_gray.shape[0]):
        for j in range(image_gray.shape[1]):
            for k in range(-d, d+1):
                for l in range(-d, d+1):
                    if (i+k >=0) and (i+k < image_gray.shape[0]) and (j+l >=0) and (j+l < image_gray.shape[1]):
                        gray_level_i = int(image_gray[i, j] * (G-1))
                        gray_level_j = int(image_gray[i+k, j+l] * (G-1))
                        glcm_matrix[gray_level_i, gray_level_j] +=1
    # Normalize the co-occurrence matrix
    glcm_matrix = glcm_matrix / (image_gray.shape[0] * image_gray.shape[1])
    return glcm_matrix

glcm_matrix = get_glcm_matrix(image_gray)

U, s, Vh = np.linalg.svd(glcm_matrix, full_matrices = False)

print("U:", U.shape)
print("s:", s.shape)
print("Vh:", Vh.shape)
