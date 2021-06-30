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

INPUT_IMAGE =  'arroz.bmp'

NEGATIVO = False
BIT = float(not NEGATIVO)
THRESHOLD = 0.7
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 20

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    return np.where(img > threshold, BIT, not BIT)

#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''
    
    result = []
    label = 0.1
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
                        
            if img[row][col][0] == BIT:                
                blob = {
                    'label': label,
                    'pixels': []
                }
                
                blob, img = inunda(blob, img, row, col)
            
                blob = {
                    'label': label,
                    'n_pixels': len(blob['pixels']),
                    'T': min(blob['pixels'], key=lambda x:x['row'])['row'],
                    'L': min(blob['pixels'], key=lambda x:x['col'])['col'],
                    'B': max(blob['pixels'], key=lambda x:x['row'])['row'],
                    'R': max(blob['pixels'], key=lambda x:x['col'])['col']                    
                }
                if blob['n_pixels'] >= n_pixels_min and blob['B'] - blob['T'] >= altura_min and blob['R'] - blob['L'] >= largura_min:
                    result.append(blob)
                
                label += 0.1
                
    return result
    

def inunda (blob, img, row, col):
            
    label = blob['label']
    img[row][col][0] = label
    blob['pixels'].append({ 'row': row, 'col': col })
                
    if check_inunda(img, row, col-1):
        blob, img = inunda(blob, img, row, col-1)
        
    if check_inunda(img, row+1, col):
        blob, img = inunda(blob, img, row+1, col)
        
    if check_inunda(img, row-1, col):
        blob, img = inunda(blob, img, row-1, col)

    if check_inunda(img, row, col+1):
        blob, img = inunda(blob, img, row, col+1)
    
    return blob, img

def check_inunda(img, row, col):
    return range_permitido(img, row, col) and img[row][col][0] == BIT and pixel_nao_mapeado(img, row, col)

def pixel_nao_mapeado(img, row, col):
    return img[row][col][0] == BIT or img[row][col][0] == (not BIT)

def range_permitido(img, row, col):
    return row >= 0 and row < img.shape[0] and col >= 0 and col < img.shape[1]



#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    #cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)
    print(componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    #cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================








