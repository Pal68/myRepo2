
import numpy as np

def haralick_matrix(image, direction='horizontal'):
    """
    Calculate the Haralick matrix for an image.

    Parameters:
    image (numpy array): Input image
    direction (str): Direction to scan the image (horizontal, vertical, or diagonal)

    Returns:
    haralick_matrix (numpy array): Haralick matrix
    """
    N = 256  # Number of gray levels
    haralick_matrix = np.zeros((N, N))

    if direction == 'horizontal':
        for i in range(image.shape[0]):
            for j in range(image.shape[1] - 1):
                gray_level_i = image[i, j]
                gray_level_j = image[i, j + 1]
                haralick_matrix[gray_level_i, gray_level_j] += 1
    elif direction == 'vertical':
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1]):
                gray_level_i = image[i, j]
                gray_level_j = image[i + 1, j]
                haralick_matrix[gray_level_i, gray_level_j] += 1
    elif direction == 'diagonal':
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1] - 1):
                gray_level_i = image[i, j]
                gray_level_j = image[i + 1, j + 1]
                haralick_matrix[gray_level_i, gray_level_j] += 1

    haralick_matrix /= haralick_matrix.sum()
    return haralick_matrix