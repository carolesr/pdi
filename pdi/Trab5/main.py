
#===============================================================================
# Trabalho 5 - Chroma Key
# Alunos: Caroline Rosa & Leonardo Trevisan
#===============================================================================

from os import altsep
import sys
import numpy as np
import cv2
import sys
import timeit
from math import sqrt

#===============================================================================

INPUT_IMAGE =  'img\\7.bmp'
PNORM = 0.01
MMGFACTOR = 0.01

def greenmask(img):
    img_out = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            (r, g, b) = img[row][col]
            ngf = (r * r + b * b + (1.0 - g) * (1.0 - g)) / 3
            img_out[row][col] = ngf
    return img_out

def normalize(img):
    histograma = [0 for x in range(256)]
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            i = int(255 * img[row][col])
            histograma[i] += 1
    
    mincount = img.shape[0] * img.shape[1] * PNORM
    minimo = 0
    for i in range(len(histograma)):
        mincount -= histograma[i]
        if mincount < 0:
            minimo = i / 255.0
            break
    
    maxcount = img.shape[0] * img.shape[1] * PNORM
    maximo = 0
    for i in range(len(histograma)):
        maxcount -= histograma[255 - i]
        if maxcount < 0:
            maximo = (255 - i) / 255.0
            break
    
    img_out = img.copy()
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x = ((img[row][col] - minimo) / (maximo - minimo))
            if x > 1.0:
                x = 1.0
            elif x < 0.0:
                x = 0.0
            img_out[row][col] = x

    return img_out

def otsu(histograma, size):
    divi = 1

    suma = histograma[0]
    sumb = 0
    index = 2
    for value in histograma[1:size]:
        sumb += index * value
        index += 1

    counta = histograma[0]
    countb = sum(histograma) - histograma[0]

    meda = histograma[0]
    medb = sumb / countb
    
    vara = 0
    varb = 0

    index = 1
    for v in histograma[divi:size]:
        varb += index * (v - medb) * (v - medb)
        index += 1
    varb = sqrt(varb)
    
    minvar = varb
    result = divi
    while divi < size - 1:
        divi += 1
        suma += divi * histograma[divi - 1]
        sumb -= divi * histograma[divi - 1]
        counta += histograma[divi - 1]
        countb -= histograma[divi - 1]
        meda = suma / counta
        medb = sumb / countb

        vara = 0
        varb = 0

        index = 1
        for v in histograma[0:divi]:
            vara += v * (index - meda) * (index - meda)
            index += 1
        vara = sqrt(vara / counta)

        index = divi
        for v in histograma[divi:size]:
            varb += v * (index - medb) * (index - medb)
            index += 1
        varb = sqrt(varb / countb)
        
        if vara + varb < minvar:
            minvar = vara + varb
            result = divi
    return result

def findminmaxgreen(mask):
    histograma = [0 for x in range(256)]
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            i = int(255 * mask[row][col])
            histograma[i] += 1
    ideal = otsu(histograma, 256)
    count = mask.shape[0] * mask.shape[1] * MMGFACTOR
    i = 0
    j = 0
    while count > 0:
        if histograma[ideal + i] > histograma[ideal - j]:
            count -= histograma[ideal - j]
            j += 1
        else:
            count -= histograma[ideal + i]
            i += 1

    return ((ideal - j) / 255, (ideal + i) / 255)

def seletivebinarization(img, maxgreen, mingreen):
    img_out = img.copy()
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img_out[row][col] > maxgreen:
                img_out[row][col] = 1.0
            elif img_out[row][col] < mingreen:
                img_out[row][col] = 0.0
    return img_out

def getidealgreen(img, mask):
    rs = 0
    gs = 0
    bs = 0
    c = 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if mask[row][col] == 0.0:
                c += 1
                rs += img[row][col][0]
                gs += img[row][col][1]
                bs += img[row][col][2]
    return (rs / c, gs / c, bs / c)

def apply(map, img, bg, idealgreen):
    img_out = img.copy()
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            a = map[row][col]
            b = 1.0 - a
            if a == 1:
                continue
            elif a == 0:
                img_out[row][col][0] = bg[row][col][0]
                img_out[row][col][1] = bg[row][col][1]
                img_out[row][col][2] = bg[row][col][2]
            else:
                img_out[row][col][0] = img_out[row][col][0] + b * (bg[row][col][0] - idealgreen[0])
                img_out[row][col][1] = img_out[row][col][1] + b * (bg[row][col][1] - idealgreen[1])
                img_out[row][col][2] = img_out[row][col][2] + b * (bg[row][col][2] - idealgreen[2])
    return img_out

#===============================================================================

def main ():
    img = cv2.imread (INPUT_IMAGE)
    bg = cv2.imread('bg.png')
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], img.shape [2]))
    img = img.astype (np.float32) / 255

    bg = bg.reshape((bg.shape [0], bg.shape [1], bg.shape [2]))
    bg = bg.astype (np.float32) / 255

    start_time = timeit.default_timer()
    mask = greenmask(img)
    print ('Tempo mask: %f' % (timeit.default_timer() - start_time))
    cv2.imwrite (f'mask.png', mask * 255)

    start_time = timeit.default_timer()
    normmask = normalize(mask)
    print ('Tempo normalize: %f' % (timeit.default_timer() - start_time))
    cv2.imwrite (f'norm.png', normmask * 255)

    # O algortimo funciona bem, porém é necessário encontrar intervalos do fator do quão verde um
    # pixel é para que o resultado seja satisfatório, aplicando assim calculos na região correta.
    # Porém, nossa abordagem estatística para determinar esses parâmetros acabou por ser ineficaz
    # para algumas imagens. Se colocarmos os parâmetros manualmente o resultado acaba por ser bom.
    start_time = timeit.default_timer()
    mingreen, maxgreen = findminmaxgreen(normmask)
    print(mingreen, maxgreen)
    finalmask = seletivebinarization(normmask, maxgreen, mingreen)
    print ('Tempo seletivebinarization: %f' % (timeit.default_timer() - start_time))
    cv2.imwrite (f'seletivebinarization.png', finalmask * 255)

    start_time = timeit.default_timer()
    idealgreen = getidealgreen(img, finalmask)
    print ('Tempo getidealgreen: %f' % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    result = apply(finalmask, img, bg, idealgreen)
    print ('Tempo apply: %f' % (timeit.default_timer() - start_time))
    cv2.imwrite (f'teste.png', result * 255)

if __name__ == '__main__':
    main ()

#===============================================================================
