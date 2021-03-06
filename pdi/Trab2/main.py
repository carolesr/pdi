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

INPUT_IMAGE =  'plantas.jpg'
ALTURA = 9
LARGURA = 9

def integral(img):
    img_aux = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for color in range(img.shape[2]):
        img_aux[0][0][color] = img[0][0][color]
        # Calcula primeira coluna
        for row in range(1, img.shape[0]):
            img_aux[row][0][color] = img[row][0][color] + img_aux[row - 1][0][color]
        # Calcula primeira linha
        for col in range(1, img.shape[1]):
            img_aux[0][col][color] = img[0][col][color] + img_aux[0][col - 1][color]
        
        # Calcula o resto da imagem integral reaproveitando somas anteriores
        # I(x, y) = img(x, y) + I(x - 1, y) + I(x, y - 1) - I(x - 1, y - 1)
        for row in range(1, img.shape[0]):
            for col in range(1, img.shape[1]):
                img_aux[row][col][color] = img[row][col][color] + img_aux[row][col - 1][color] + img_aux[row - 1][col][color] - img_aux[row - 1][col - 1][color]
    return img_aux

def filtro_media_imagens_integrais(img):
    
    img_aux = integral(img)

    alto2 = ALTURA // 2
    laro2 = LARGURA // 2
    for row in range(img_aux.shape[0]):
        for col in range(img_aux.shape[1]):
            for color in range(img_aux.shape[2]):
                #┌─────────┐
                #│         │
                #│   ┌───┬─┤ y1
                #│   │ K │ │
                #│   ├───┼─┤ y2
                #└───┼───┼─┘
                #    x1  x2
                x1 = col - laro2 - 1
                x2 = col + laro2
                y1 = row - alto2 - 1
                y2 = row + alto2
                if (x1 < 0):
                    x1 = 0
                if (y1 < 0):
                    y1 = 0
                if (x2 >= img.shape[1]):
                    x2 = img.shape[1] - 1
                if (y2 >= img.shape[0]):
                    y2 = img.shape[0] - 1

                valor = img_aux[y2][x2][color] - img_aux[y1][x2][color] - img_aux[y2][x1][color] + img_aux[y1][x1][color]
                area = float((x2 - x1) * (y2 - y1))
                valor = float(valor) / area
                img[row][col][color] = valor

    return img


def filtro_media_separavel(img):

    img_out = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
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

    img_out = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
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

    start_time = timeit.default_timer ()
    img_out = filtro_media_ingenuo(img)
    cv2.imwrite (f'ingeuno_janela_{ALTURA}x{LARGURA}.png', img_out*255)
    print ('Tempo ingênuo: %f' % (timeit.default_timer () - start_time))
    
    start_time = timeit.default_timer ()
    img_out = filtro_media_separavel(img)
    cv2.imwrite (f'separavel_janela_{ALTURA}x{LARGURA}.png', img_out*255)
    print ('Tempo separável: %f' % (timeit.default_timer () - start_time))
    
    start_time = timeit.default_timer ()
    img_out = filtro_media_imagens_integrais(img)
    cv2.imwrite (f'integrais_janela_{ALTURA}x{LARGURA}.png', img_out*255)
    print ('Tempo integrais: %f' % (timeit.default_timer () - start_time))

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================
