#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

#===============================================================================
# Alunos: Caroline Rosa & Leonardo Trevisan
#===============================================================================

import sys
import timeit
import numpy as np
import cv2
import sys
import math as m

#===============================================================================

INPUT_IMAGE =  'flores.bmp'
ALTURA = 101
LARGURA = 101

def filtro_media_separavel(img):

    img_out = img
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for color in range(img.shape[2]):

                soma = 0.0
                for col_janela in range(col - m.floor(LARGURA/2), col + m.floor(LARGURA/2)+1):
                    soma += img[row][col_janela][color] if check_bordas(img, row, col_janela) else 1.0
                img_out[row][col][color] = float(soma / LARGURA)

    img = img_out
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for color in range(img.shape[2]):

                soma = 0.0
                for row_janela in range(row - m.floor(ALTURA/2), row + m.floor(ALTURA/2)+1):
                    soma += img[row_janela][col][color] if check_bordas(img, row_janela, col) else 1.0
                img_out[row][col][color] = float(soma / ALTURA)

    return img_out


def filtro_media_ingenuo(img):

    img_out = img
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for color in range(img.shape[2]):

                soma = 0.0
                for row_janela in range(row - m.floor(ALTURA/2), row + m.floor(ALTURA/2)+1):
                    for col_janela in range(col - m.floor(LARGURA/2), col + m.floor(LARGURA/2)+1):

                        soma += img[row_janela][col_janela][color] if check_bordas(img, row_janela, col_janela) else 1.0

                img_out[row][col][color] = float(soma / (ALTURA*LARGURA))

    return img_out

def check_bordas(img, row, col):
    return row < img.shape[0] and row >= 0 and col < img.shape[1] and col >= 0

#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], img.shape [2]))
    img = img.astype (np.float32) / 255

    print(img.shape)

    start_time = timeit.default_timer ()
    # img2 = filtro_media_ingenuo(img)
    img2 = filtro_media_separavel(img)
    cv2.imwrite (f'janela_{ALTURA}x{LARGURA}.png', img2*255)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================
