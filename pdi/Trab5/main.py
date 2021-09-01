
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

#===============================================================================

INPUT_IMAGE =  'img\\1.bmp'

def graymask(img):
    img_out = img.copy()
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            (r, g, b) = img[row][col]
            img_out[row][col] = (r * r + b * b + (1 - g) * (1 - g)) / 3
    return img_out

def normalize(img):
    maximo = img[..., 0].max()
    minimo = img[..., 0].min()
    print(maximo)
    print(minimo)
    img_out = img.copy()
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img_out[row][col][0] = 1 - ((img[row][col][0] - minimo) / (maximo - minimo))
    return img_out

def apply(map, img, bg):
    img_out = img.copy()
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img_out[row][col][0] = (map[row][col][0] * img_out[row][col][0]) + ((1.0 - map[row][col][0]) * bg[row][col][0])
            img_out[row][col][1] = (map[row][col][0] * img_out[row][col][1]) + ((1.0 - map[row][col][0]) * bg[row][col][1])
            img_out[row][col][2] = (map[row][col][0] * img_out[row][col][2]) + ((1.0 - map[row][col][0]) * bg[row][col][2])
    return img_out

#===============================================================================

def main ():
    img = cv2.imread (INPUT_IMAGE)
    bg = cv2.imread('bg.png')
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], img.shape [2]))
    img = img.astype (np.float32) / 255

    bg = bg.reshape((bg.shape [0], bg.shape [1], bg.shape [2]))
    bg = bg.astype (np.float32) / 255

    start_time = timeit.default_timer ()
    mask = graymask(img)
    print ('Tempo mask: %f' % (timeit.default_timer () - start_time))
    cv2.imwrite (f'mask.png', mask * 255)

    grayscale = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    grayscale = grayscale.reshape ((grayscale.shape [0], grayscale.shape [1], 1))
    grayscale = grayscale.astype (np.float32) / 255

    start_time = timeit.default_timer ()
    normalizada = normalize(grayscale)
    print ('Tempo normalize: %f' % (timeit.default_timer () - start_time))
    cv2.imwrite (f'norm.png', normalizada * 255)

    start_time = timeit.default_timer ()
    result = apply(normalizada, img, bg)
    print ('Tempo apply: %f' % (timeit.default_timer () - start_time))
    cv2.imwrite (f'teste.png', result * 255)
    

    # # Converte imagem RGB para HSL
    # imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # verde = grau_de_verde(imgHLS)
    # # Converte de volta pra RGB
    # teste = cv2.cvtColor(verde, cv2.COLOR_HLS2BGR)
    # cv2.imwrite (f'teste.png', teste*255)
    # print('done')

    # cv2.waitKey ()
    # cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================
