def getColorChanelMatrix(img,chanel):
    if chanel in range(0,3):
        result_matrix=img[:,:,chanel]
    else:
        result_matrix = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return result_matrix
#end getColorChanelMatrix
