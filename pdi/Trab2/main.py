#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

#===============================================================================
# Alunos: Caroline Rosa & Leonardo Trevisan
#===============================================================================

from os import altsep
import sys
import timeit
import numpy as np
import cv2
import sys
import math as m

#===============================================================================

INPUT_IMAGE =  'flores.bmp'
ALTURA = 3
LARGURA = 3

def filtro_imagens_integrais(img):
    
    img_aux = img
    for color in range(img_aux.shape[2]):
        for row in range(img_aux.shape[0]):
            for col in range(img_aux.shape[1]):
                img_aux[row][col][color] = img_aux[row][col][color] + (img_aux[row][col-1][color] if check_bordas(img_aux, row, col-1) else 0)

    for color in range(img_aux.shape[2]):
        for row in range(img_aux.shape[0]):
            for col in range(img_aux.shape[1]):
                img_aux[row][col][color] = img_aux[row][col][color] + (img_aux[row-1][col][color] if check_bordas(img_aux, row-1, col) else 0)

    # return img_aux

    img_out = img_aux
    for row in range(img_aux.shape[0]):
        for col in range(img_aux.shape[1]):
            for color in range(img_aux.shape[2]):

                if check_bordas(img_aux, row-ALTURA, col) and check_bordas(img_aux, row, col-LARGURA) and check_bordas(img_aux, row-ALTURA, col-LARGURA):
                    print(img_aux[row][col][color])
                    print(img_aux[row-ALTURA][col][color])
                    print(img_aux[row][col-LARGURA][color])
                    print(img_aux[row-ALTURA][col-LARGURA][color])
                    valor = img_aux[row][col][color] - img_aux[row-ALTURA][col][color] - img_aux[row][col-LARGURA][color] + img_aux[row-ALTURA][col-LARGURA][color]
                    valor = float(valor / (ALTURA * LARGURA))
                    print(f'pixel final: {valor}')
                    #esse valor ta passando de 1 oq eu faço??????????
                    img_out[row][col][color] = valor
                else: 
                    # print('out of range')
                    img_out[row][col][color] = img[row][col][color]

    return img_out


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
    # img2 = filtro_media_separavel(img)
    img2 = filtro_imagens_integrais(img)
    cv2.imwrite (f'integrais_janela_{ALTURA}x{LARGURA}.png', img2*255)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================
