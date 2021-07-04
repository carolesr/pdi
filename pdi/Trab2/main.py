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

#===============================================================================

INPUT_IMAGE =  'flores.bmp'
JANELA = 3

def filtro_media_ingenuo(img, janela):

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            
            for color in range(3):

                soma = img[row][col][color]
                for camada in range(1, janela-1):
                    soma += get_valor_pixel(img, row-camada, col, color)
                    soma += get_valor_pixel(img, row-camada, col+camada, color)
                    soma += get_valor_pixel(img, row, col+camada, color)
                    soma += get_valor_pixel(img, row+camada, col+camada, color)
                    soma += get_valor_pixel(img, row+camada, col, color)
                    soma += get_valor_pixel(img, row+camada, col-camada, color)
                    soma += get_valor_pixel(img, row, col-camada, color)
                    soma += get_valor_pixel(img, row-camada, col-camada, color)

                img[row][col][color] = float(soma / (janela*janela))

    return img

def get_valor_pixel(img, row, col, color):
    if row < img.shape[0] and row >= 0 and col < img.shape[1] and col >= 0:
        return img[row][col][color]
    return 0.0                                                                                                     
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
    img2 = filtro_media_ingenuo(img, 3)
    cv2.imwrite ('janela_3.png', img2*255)
    print ('Tempo janela 3: %f' % (timeit.default_timer () - start_time))

    start_time = timeit.default_timer ()
    img2 = filtro_media_ingenuo(img, 5)
    cv2.imwrite ('janela_5.png', img2*255)
    print ('Tempo janela 5: %f' % (timeit.default_timer () - start_time))

    start_time = timeit.default_timer ()
    img2 = filtro_media_ingenuo(img, 7)
    cv2.imwrite ('janela_7.png', img2*255)
    print ('Tempo janela 7: %f' % (timeit.default_timer () - start_time))

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================
