import cv2 as cv
import kakNaRabote
import excelFunc
import numpy as np

def get_pure_colors_image(image):
    pure_color_matrix = (image[:, :, :] >= 127).astype(int) * 255
    return pure_color_matrix


def get_pure_colors_props(image):
    pure_color_matrix = (image[:,:,:] >= 127).astype(int) * 255
    long_color_matrix = pure_color_matrix[:, :, 0] + 256 * pure_color_matrix[:, :, 1] + (256*256) * pure_color_matrix[:, :, 2]
    tmp_bin = set(long_color_matrix.flat)
    input_bin = [ 0, 255, 65280, 65535, 16711680, 16711935,16776960,16777215,16777215]
    # input_bin=np.sort(np.array(list(tmp_bin)))
    pure_color_hist, pure_color_bin = np.histogram(long_color_matrix, bins=input_bin, density=False)
    pure_color_hist=np.array(pure_color_hist)
    pure_color_hist=pure_color_hist[:] / np.sum(pure_color_hist)
    return pure_color_hist, pure_color_bin

img=cv.imread("./FotoCore/_ 1piece_metr2.bmp")
hist, bin = get_pure_colors_props(img)
print(bin)
print(hist)

wb=excelFunc.getExcelWorkBook('./Book1.xlsx')
sheet = wb.worksheets[0]

#--------------------------------------------------------------
blok_size=64
start_row=19
end_row=33
h_tresold = 0.9
#-----------------------------------------------------------------


pure_color_props_arr=[]
for exl_row in range (start_row,end_row+1):
            image_C=cv.imread("./FotoCore/"+sheet.cell(exl_row,1).value)
            top_Y=sheet.cell(exl_row,13).value
            base_Y = sheet.cell(exl_row, 14).value
            image_C = kakNaRabote.copyPartMatrix(image_C, 10, 138, top_Y, base_Y)
            tmp_image = get_pure_colors_image(image_C)
            cv.imwrite("./tmp/" + sheet.cell(exl_row,1).value[0:len(sheet.cell(exl_row,1).value)-4]+"_PC.jpg", tmp_image)
            pure_color_hist, pure_color_bin = get_pure_colors_props(image_C)
            pure_color_props_arr.append([sheet.cell(exl_row,1).value+"-"+sheet.cell(exl_row+1,1).value, pure_color_hist])

for i in range(0, len(pure_color_props_arr)-1):
    cos_similarity = kakNaRabote.getCOSSimilarity(pure_color_props_arr[i][1], pure_color_props_arr[i+1][1])
    print(pure_color_props_arr[i][0], " cos_sim- ", cos_similarity)