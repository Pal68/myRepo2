import cv2 as cv
import main
import excelFunc
def get_pure_colors(image):
    # result_matrix=image[:,:,:]*1.0
    result_matrix = (image[:,:,:] > 127).astype(int) * 255
    return result_matrix

image=cv.imread("./tmp/21piece_metr1_pc.jpg")
tmp=get_pure_colors(image)
cv.imwrite("./tmp/tmp.jpg",tmp)


wb=excelFunc.getExcelWorkBook('./Book1.xlsx')
sheet = wb.worksheets[0]

#--------------------------------------------------------------
blok_size=32
start_row=3
end_row=18
h_tresold = 0.9
#-----------------------------------------------------------------



for exl_row in range (start_row,end_row+1):
            image_C=cv.imread("./FotoCore/"+sheet.cell(exl_row,1).value)
            top_Y=sheet.cell(exl_row,13).value
            base_Y = sheet.cell(exl_row, 14).value
            image_C = main.copyPartMatrix(image_C, 10, 138, top_Y, base_Y)
            cv.imwrite("./tmp/" + sheet.cell(exl_row,1).value[0:len(sheet.cell(exl_row,1).value)-4]+".jpg", image_C)
            image_C = get_pure_colors(image_C)
            cv.imwrite("./tmp/" + sheet.cell(exl_row, 1).value[0:len(sheet.cell(exl_row, 1).value) - 4] + "_PureC.jpg",
                       image_C)
