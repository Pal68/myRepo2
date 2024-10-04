def getColorChanelMatrix(img,chanel):
    if chanel in range(0,3):
        result_matrix=img[:,:,chanel]
    else:
        result_matrix = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return result_matrix
#end getColorChanelMatrix

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2 as cv
import numpy as np
import os as os
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, column_index_from_string

import excelFunc


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


def getLBPMatrix(img,chanel):
    if chanel in range(0,3):
        result_matrix=img[:,:,chanel]
    else:
        result_matrix = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    lbp = np.zeros((result_matrix.shape[0], result_matrix.shape[1]))
    for i in range(result_matrix.shape[0]):
       for j in range(result_matrix.shape[1]):
            # Calculate LBP for red channel
            data=((result_matrix[i, j] - result_matrix[max(0, i-1):min(result_matrix.shape[0], i+2), max(0, j-1):min(result_matrix.shape[1], j+2)]) >= 0)
            c=np.zeros((3,3))
            s = data.size
            match s :
                case 4:
                    if i==0 and j==0:#Левый верхний угол
                       c[1:3, 1:3] = data
                    elif i!=0 and j!=0:#правый нижний угол
                       c[0:2, 0:2] = data
                    elif i!=0 and j==0:#Левый нижний угол
                        c[0:2, 1:3] = data
                    else:
                        c[1:3, 0:2] = data#правый верхний угол
                case 6 :
                    if i==result_matrix.shape[0]-1 and j>0:#нижняя сторона
                        c[0:2,0:3]=data
                    elif i==0 and j>0:#верхняя сторона
                        c[1:3, 0:4] = data
                    elif i!=0 and j==0:
                        c[0:4, 1:4] = data#левая сторона
                    elif i!=0 and j==result_matrix.shape[1]-1:
                        c[0:4, 0:2] = data  # правая сторона
                case _:
                    c[0:4,0:4]=data

            tmpV=np.zeros(8)
            jj=0
            for ii in range(0,c.shape[1]):
                tmpV[jj]=c[0,ii]
                jj=jj+1
            for ii in range(1, c.shape[0]):
                tmpV[jj]=c[ii,c.shape[0]-1]
                jj=jj+1
            for ii in range(c.shape[1]-2,-1,-1):
                tmpV[jj]=c[c.shape[0]-1,ii]
                jj=jj+1
            for ii in range(c.shape[0]-2,0,-1):
                tmpV[jj] = c[ii,0]
            lbp[i, j] = np.sum(tmpV * 2 ** np.arange(8))
    return lbp
#end getLBPmatrix



def getGrayLBPMatrix(img):
    # Convert RGB image to grayscale
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    lbp = np.zeros((gray.shape[0], gray.shape[1]))
    for i in range(gray.shape[0]):
       for j in range(gray.shape[1]):
            # Calculate LBP for red channel
            data=((gray[i, j] - gray[max(0, i-1):min(gray.shape[0], i+2), max(0, j-1):min(gray.shape[1], j+2)]) >= 0)
            c=np.zeros((3,3))
            s = data.size
            match s :
                case 4:
                    if i==0 and j==0:#Левый верхний угол
                       c[1:3, 1:3] = data
                    elif i!=0 and j!=0:#правый нижний угол
                       c[0:2, 0:2] = data
                    elif i!=0 and j==0:#Левый нижний угол
                        c[0:2, 1:3] = data
                    else:
                        c[1:3, 0:2] = data#правый верхний угол
                case 6 :
                    if i==gray.shape[0]-1 and j>0:#нижняя сторона
                        c[0:2,0:3]=data
                    elif i==0 and j>0:#верхняя сторона
                        c[1:3, 0:4] = data
                    elif i!=0 and j==0:
                        c[0:4, 1:4] = data#левая сторона
                    elif i!=0 and j==gray.shape[1]-1:
                        c[0:4, 0:2] = data  # правая сторона
                case _:
                    c[0:4,0:4]=data

            tmpV=np.zeros(8)
            jj=0
            for ii in range(0,c.shape[1]):
                tmpV[jj]=c[0,ii]
                jj=jj+1
            for ii in range(1, c.shape[0]):
                tmpV[jj]=c[ii,c.shape[0]-1]
                jj=jj+1
            for ii in range(c.shape[1]-2,-1,-1):
                tmpV[jj]=c[c.shape[0]-1,ii]
                jj=jj+1
            for ii in range(c.shape[0]-2,0,-1):
                tmpV[jj] = c[ii,0]
            lbp[i, j] = np.sum(tmpV * 2 ** np.arange(8))
    return lbp
#end getGrayLBPmatrix


def getUniFormLBPHist(matr):
    tmp = ((57) not in {1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120,
                       124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227,
                       231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254})
    matr=matr*tmp
    lbpHist, bins = np.histogram(a=matr,bins=57,range=(0,57),density=True)
    return lbpHist, bins
#end     getUnoFormLBPHist

def copyPartMatrix(matr, leftX, rightX, topY, baseY):
    result = matr[topY:baseY, leftX:rightX, 0:3]
    return result
#end     copyPartMatrix

def getLBPPropsForPiece(matr, blok_size):
    # в зависимости от размера изображения расчитываем количестов итераций и размер вектора свойств для слоя
    end_w_iteration = matr.shape[1] // blok_size
    if end_w_iteration == 0:
        w_over = 0
        end_w_iteration = 1
    else:
        w_over = ((matr.shape[1]) - end_w_iteration * blok_size)
    end_h_iteration = (matr.shape[0]) // blok_size
    if end_h_iteration == 0:
        h_over = 0
        end_h_iteration = 1
    else:
        h_over = (matr.shape[0]) - end_h_iteration * blok_size
    '''
    if w_over > 3:
        if h_over > 3:
            gray_props_vector = np.zeros(((end_w_iteration + 1) * (end_h_iteration + 1)) * 57)
            red_props_vector = np.zeros(((end_w_iteration + 1) * (end_h_iteration + 1)) * 57)
            green_props_vector = np.zeros(((end_w_iteration + 1) * (end_h_iteration + 1)) * 57)
            blue_props_vector = np.zeros(((end_w_iteration + 1) * (end_h_iteration + 1)) * 57)
        else:
            gray_props_vector = np.zeros((end_w_iteration + 1) * end_h_iteration * 57)
            red_props_vector = np.zeros((end_w_iteration + 1) * end_h_iteration * 57)
            green_props_vector = np.zeros((end_w_iteration + 1) * end_h_iteration * 57)
            blue_props_vector = np.zeros((end_w_iteration + 1) * end_h_iteration * 57)
    else:
        if h_over > 3:
            gray_props_vector = np.zeros(end_w_iteration * (end_h_iteration + 1) * 57)
            red_props_vector = np.zeros(end_w_iteration * (end_h_iteration + 1) * 57)
            green_props_vector = np.zeros(end_w_iteration * (end_h_iteration + 1) * 57)
            blue_props_vector = np.zeros(end_w_iteration * (end_h_iteration + 1) * 57)
        else:
            gray_props_vector = np.zeros(end_w_iteration * end_h_iteration * 57)
            red_props_vector = np.zeros(end_w_iteration * end_h_iteration * 57)
            green_props_vector = np.zeros(end_w_iteration * end_h_iteration * 57)
            blue_props_vector = np.zeros(end_w_iteration * end_h_iteration * 57)
    '''
    gray_lbp_list = []
    red_lbp_list = []
    green_lbp_list = []
    blue_lbp_list = []
    for ii in range(0, end_h_iteration):
        for jj in range(0, end_w_iteration):
            cyr_blok_matrix = copyPartMatrix(matr, jj * blok_size, jj * blok_size + (blok_size - 1), ii * blok_size,
                                           ii * blok_size + (blok_size - 1))
            gray_blok_matrix = getLBPMatrix(cyr_blok_matrix,3)
            red_blok_matrix = getLBPMatrix(cyr_blok_matrix, 0)
            green_blok_matrix = getLBPMatrix(cyr_blok_matrix, 1)
            blue_blok_matrix = getLBPMatrix(cyr_blok_matrix, 2)
            gray_LBP_hist, gray_LBP_bin = getUniFormLBPHist(gray_blok_matrix)
            red_LBP_hist, red_LBP_bin = getUniFormLBPHist(red_blok_matrix)
            green_LBP_hist, green_LBP_bin = getUniFormLBPHist(green_blok_matrix)
            blue_LBP_hist, blue_LBP_bin = getUniFormLBPHist(blue_blok_matrix)
            for k in range(0, gray_LBP_hist.shape[0]):
                gray_lbp_list.append(gray_LBP_hist[k])
                red_lbp_list.append(red_LBP_hist[k])
                green_lbp_list.append(green_LBP_hist[k])
                blue_lbp_list.append(blue_LBP_hist[k])
        if w_over > 3:
            cyr_blok_matrix = copyPartMatrix(matr, matr.shape[1] - w_over + 1, matr.shape[1] - 1, ii * blok_size,
                                           ii * blok_size + (blok_size - 1))
            gray_blok_matrix = getLBPMatrix(cyr_blok_matrix,3)
            red_blok_matrix = getLBPMatrix(cyr_blok_matrix, 0)
            green_blok_matrix = getLBPMatrix(cyr_blok_matrix, 1)
            blue_blok_matrix = getLBPMatrix(cyr_blok_matrix, 2)
            gray_LBP_hist, gray_LBP_bin = getUniFormLBPHist(gray_blok_matrix)
            red_LBP_hist, red_LBP_bin = getUniFormLBPHist(red_blok_matrix)
            green_LBP_hist, green_LBP_bin = getUniFormLBPHist(green_blok_matrix)
            blue_LBP_hist, blue_LBP_bin = getUniFormLBPHist(blue_blok_matrix)
            for k in range(0, gray_LBP_hist.shape[0]):
                gray_lbp_list.append(gray_LBP_hist[k])
                red_lbp_list.append(red_LBP_hist[k])
                green_lbp_list.append(green_LBP_hist[k])
                blue_lbp_list.append(blue_LBP_hist[k])
    if h_over > 3:
        for jj in range(0, end_w_iteration):
            cyr_blok_matrix = copyPartMatrix(matr, jj * blok_size, jj * blok_size + (blok_size - 1),
                                           matr.shape[0] - h_over + 1, matr.shape[0] - 1)
            gray_blok_matrix = getLBPMatrix(cyr_blok_matrix,3)
            red_blok_matrix = getLBPMatrix(cyr_blok_matrix, 0)
            green_blok_matrix = getLBPMatrix(cyr_blok_matrix, 1)
            blue_blok_matrix = getLBPMatrix(cyr_blok_matrix, 2)
            gray_LBP_hist, gray_LBP_bin = getUniFormLBPHist(gray_blok_matrix)
            red_LBP_hist, red_LBP_bin = getUniFormLBPHist(red_blok_matrix)
            green_LBP_hist, green_LBP_bin = getUniFormLBPHist(green_blok_matrix)
            blue_LBP_hist, blue_LBP_bin = getUniFormLBPHist(blue_blok_matrix)
            for k in range(0, gray_LBP_hist.shape[0]):
                gray_lbp_list.append(gray_LBP_hist[k])
                red_lbp_list.append(red_LBP_hist[k])
                green_lbp_list.append(green_LBP_hist[k])
                blue_lbp_list.append(blue_LBP_hist[k])
    if w_over > 3:
        cyr_blok_matrix = copyPartMatrix(matr, matr.shape[1] - w_over + 1, matr.shape[1] - 1, matr.shape[0] - h_over + 1,
                                       matr.shape[0] - 1)
        gray_blok_matrix = getLBPMatrix(cyr_blok_matrix,3)
        red_blok_matrix = getLBPMatrix(cyr_blok_matrix, 0)
        green_blok_matrix = getLBPMatrix(cyr_blok_matrix, 1)
        blue_blok_matrix = getLBPMatrix(cyr_blok_matrix, 2)
        gray_LBP_hist, gray_LBP_bin = getUniFormLBPHist(gray_blok_matrix)
        red_LBP_hist, red_LBP_bin = getUniFormLBPHist(red_blok_matrix)
        green_LBP_hist, green_LBP_bin = getUniFormLBPHist(green_blok_matrix)
        blue_LBP_hist, blue_LBP_bin = getUniFormLBPHist(blue_blok_matrix)
        for k in range(0, gray_LBP_hist.shape[0]):
            gray_lbp_list.append(gray_LBP_hist[k])
            red_lbp_list.append(red_LBP_hist[k])
            green_lbp_list.append(green_LBP_hist[k])
            blue_lbp_list.append(blue_LBP_hist[k])

    return gray_lbp_list, red_lbp_list, green_lbp_list, blue_lbp_list
#end def getLBPPropsForPiece

def getLBPPropsForLayer(matr, blokSize):
# в зависимости от размера изображения расчитываем количестов итераций и размер вектора свойств для слоя
    endWIteration = matr.shape[1]//blokSize
    if endWIteration == 0:
        wOver = 0
        endWIteration=1
    else:
        wOver = ((matr.shape[1] ) - endWIteration * blokSize)
    endHIteration = (matr.shape[0])//blokSize
    if endHIteration == 0:
        hOver = 0
        endHIteration=1
    else:
        hOver = (matr.shape[0] ) - endHIteration * blokSize
    if wOver > 3:
        if hOver > 3:
                cyrPiecePropVector = np.zeros(((endWIteration + 1) * (endHIteration + 1))*57)
        else:
                cyrPiecePropVector = np.zeros((endWIteration + 1) * endHIteration * 57)
    else:
        if hOver > 3:
            cyrPiecePropVector = np.zeros(endWIteration * (endHIteration+1) * 57)
        else:
            cyrPiecePropVector = np.zeros(endWIteration * endHIteration * 57)

    lbpList=[]
    for ii in range(0,endHIteration):
        for jj in range(0,endWIteration):
            cyrBlokMatrix = copyPartMatrix(matr,jj * blokSize, jj * blokSize + (blokSize - 1), ii * blokSize, ii * blokSize + (blokSize - 1))
            cyrBlokMatrix=getGrayLBPMatrix(cyrBlokMatrix)
            cyrLBPHist, cyrLBPBin=getUniFormLBPHist(cyrBlokMatrix)
            #cyrPiecePropVector[(ii+jj) * 57:(ii+jj) * 57 + 57]=cyrLBPHist
            for k in range(0,cyrLBPHist.shape[0]):
                lbpList.append(cyrLBPHist[k])
        if wOver > 3:
            cyrBlokMatrix = copyPartMatrix(matr, matr.shape[1] - wOver + 1, matr.shape[1] - 1, ii * blokSize, ii * blokSize + (blokSize - 1))
            cyrBlokMatrix = getGrayLBPMatrix(cyrBlokMatrix)
            cyrLBPHist, cyrLBPBin = getUniFormLBPHist(cyrBlokMatrix)
            #cyrPiecePropVector[(ii+jj+1) * 57:(ii+jj+1) * 57 + 57] = cyrLBPHist
            for k in range(0,cyrLBPHist.shape[0]):
                lbpList.append(cyrLBPHist[k])
    if hOver > 3:
        for jj in range(0, endWIteration):
            cyrBlokMatrix = copyPartMatrix(matr, jj * blokSize, jj * blokSize + (blokSize - 1), matr.shape[0]-hOver+1, matr.shape[0]-1)
            cyrBlokMatrix = getGrayLBPMatrix(cyrBlokMatrix)
            cyrLBPHist, cyrLBPBin = getUniFormLBPHist(cyrBlokMatrix)
            #cyrPiecePropVector[(ii+jj) * 57:(ii+jj) * 57 + 57] = cyrLBPHist
            for k in range(0,cyrLBPHist.shape[0]):
                lbpList.append(cyrLBPHist[k])
    if wOver > 3:
        cyrBlokMatrix = copyPartMatrix(matr, matr.shape[1]-wOver+1, matr.shape[1]-1, matr.shape[0] - hOver + 1, matr.shape[0] - 1)
        cyrBlokMatrix = getGrayLBPMatrix(cyrBlokMatrix)
        cyrLBPHist, cyrLBPBin = getUniFormLBPHist(cyrBlokMatrix)
        #cyrPiecePropVector[(ii+jj) * 57:(ii+jj) * 57 + 57] = cyrLBPHist
        for k in range(0, cyrLBPHist.shape[0]):
            lbpList.append(cyrLBPHist[k])
    return lbpList
#end getPropsForLayer(matr)

def getCOSSimilarity(v1,v2):
    result = np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
    return result
#end getCOSSimilarity







wb=excelFunc.getExcelWorkBook('./Book1.xlsx')
sheet = wb.worksheets[0]
print(type(sheet))
gray_prop_vect_arr =[]
red_prop_vect_arr =[]
green_prop_vect_arr =[]
blue_prop_vect_arr =[]
blok_size=32
for cellObj in sheet['A65':'A79']:
      for cell in cellObj:
              image_C=cv.imread("C:/Users/Palaguto_va/PycharmProjects/pythonProject1/FotoCore/"+cell.value)
              top_Y=sheet.cell(cell.row,13).value
              base_Y = sheet.cell(cell.row, 14).value
              image_C=copyPartMatrix(image_C, 10, 138, top_Y, base_Y)
              gray_prop_V, red_prop_V, green_prop_V, blue_prop_V = getLBPPropsForPiece(image_C, blok_size)
              gray_prop_vect_arr.append(gray_prop_V)
              red_prop_vect_arr.append(red_prop_V)
              green_prop_vect_arr.append(green_prop_V)
              blue_prop_vect_arr.append(blue_prop_V)
              print('property vectors '+cell.value, gray_prop_V)
              print('property vectors ' + cell.value, red_prop_V)
              print('property vectors ' + cell.value, green_prop_V)
              print('property vectors ' + cell.value, blue_prop_V)
#-----------------------------------------------------------------------------------
#добиваем нулями
for i in range(0, len(gray_prop_vect_arr)-1):
    gray_prop_v1=gray_prop_vect_arr[i]
    gray_prop_v2=gray_prop_vect_arr[i+1]
    red_prop_v1 = red_prop_vect_arr[i]
    red_prop_v2 = red_prop_vect_arr[i + 1]
    green_prop_v1 = green_prop_vect_arr[i]
    green_prop_v2 = green_prop_vect_arr[i + 1]
    blue_prop_v1 = blue_prop_vect_arr[i]
    blue_prop_v2 = blue_prop_vect_arr[i + 1]
    if len(gray_prop_v1)>=len(gray_prop_v2):
        tmp_prop_v=np.zeros(len(gray_prop_v1))
        tmp_prop_v[0:len(gray_prop_v2)]=gray_prop_v2[0:len(gray_prop_v2)]
        cos_similarity = getCOSSimilarity(gray_prop_v1, tmp_prop_v)
        tmp_prop_v = np.zeros(len(red_prop_v1))
        tmp_prop_v[0:len(red_prop_v2)] = red_prop_v2[0:len(red_prop_v2)]
        cos_similarity = cos_similarity + getCOSSimilarity(red_prop_v1, tmp_prop_v)
        tmp_prop_v = np.zeros(len(green_prop_v1))
        tmp_prop_v[0:len(green_prop_v2)] = green_prop_v2[0:len(green_prop_v2)]
        cos_similarity = cos_similarity + getCOSSimilarity(green_prop_v1, tmp_prop_v)
        tmp_prop_v = np.zeros(len(blue_prop_v1))
        tmp_prop_v[0:len(blue_prop_v2)] = blue_prop_v2[0:len(blue_prop_v2)]
        cos_similarity = (cos_similarity + getCOSSimilarity(blue_prop_v1, tmp_prop_v))/4
    else:
        tmp_prop_v=np.zeros(len(gray_prop_v2))
        tmp_prop_v[0:len(gray_prop_v1)]=gray_prop_v1[0:len(gray_prop_v1)]
        cos_similarity = getCOSSimilarity(gray_prop_v2, tmp_prop_v)
        tmp_prop_v = np.zeros(len(red_prop_v2))
        tmp_prop_v[0:len(red_prop_v1)] = red_prop_v1[0:len(red_prop_v1)]
        cos_similarity = cos_similarity + getCOSSimilarity(red_prop_v2, tmp_prop_v)
        tmp_prop_v = np.zeros(len(green_prop_v2))
        tmp_prop_v[0:len(green_prop_v1)] = green_prop_v1[0:len(green_prop_v1)]
        cos_similarity = cos_similarity + getCOSSimilarity(green_prop_v2, tmp_prop_v)
        tmp_prop_v = np.zeros(len(blue_prop_v2))
        tmp_prop_v[0:len(blue_prop_v1)] = blue_prop_v1[0:len(blue_prop_v1)]
        cos_similarity = (cos_similarity + getCOSSimilarity(blue_prop_v2, tmp_prop_v))/4
    print(sheet.cell(i+65,1).value,"-",sheet.cell(i+66,1).value,"                ",cos_similarity)
    #if cos_similarity < 0.75:
       #print(sheet.cell(i+65,1).value,"-",sheet.cell(i+66,1).value,"                ",cos_similarity)
'''
#-----------------------------------------------------------------------------------
#конец добиваем нулями
print("")
#обрезаем длинный
for i in range(0, len(prop_vect_arr)-1):
    prop_v1=prop_vect_arr[i]
    prop_v2=prop_vect_arr[i+1]
    if len(prop_v1)>=len(prop_v2):
        tmp_prop_v=np.zeros(len(prop_v2))
        tmp_prop_v[0:len(prop_v2)]=prop_v1[0:len(prop_v2)]
        cos_similarity = getCOSSimilarity(prop_v2, tmp_prop_v)
    else:
        tmp_prop_v=np.zeros(len(prop_v1))
        tmp_prop_v[0:len(prop_v1)]=prop_v2[0:len(prop_v1)]
        cos_similarity = getCOSSimilarity(prop_v1, tmp_prop_v)
    #print(cos_similarity)
    if cos_similarity < 0.90:
       print(sheet.cell(i+65,1).value,"-",sheet.cell(i+66,1).value,"                ",cos_similarity)
#-----------------------------------------------------------------------------------
#конец обрезаем длинный
'''



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
