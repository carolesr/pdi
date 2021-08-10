
#===============================================================================
# Trabalho 4 - Separar Arroz
# Alunos: Caroline Rosa & Leonardo Trevisan
#===============================================================================

from os import altsep
import sys
import numpy as np
import cv2
import sys
import timeit

#===============================================================================

INPUT_IMAGE =  '60.bmp'
ALTURA = 300
LARGURA = 300
FATOR = 1.3
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 20
    
def integral(img):
    img_aux = np.zeros((img.shape[0], img.shape[1]))
    # Calcula primeira coluna
    for row in range(1, img.shape[0]):
        img_aux[row][0] = img[row][0] + img_aux[row - 1][0]
    # Calcula primeira linha
    for col in range(1, img.shape[1]):
        img_aux[0][col] = img[0][col] + img_aux[0][col - 1]
    
    # Calcula o resto da imagem integral reaproveitando somas anteriores
    # I(x, y) = img(x, y) + I(x - 1, y) + I(x, y - 1) - I(x - 1, y - 1)
    for row in range(1, img.shape[0]):
        for col in range(1, img.shape[1]):
            img_aux[row][col] = img[row][col] + img_aux[row][col - 1] + img_aux[row - 1][col] - img_aux[row - 1][col - 1]
    return img_aux

def limiarizacao_adaptativa(img):
    
    img_aux = integral(img)

    alto2 = ALTURA // 2
    laro2 = LARGURA // 2
    for row in range(img_aux.shape[0]):
        for col in range(img_aux.shape[1]):
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

                valor = img_aux[y2][x2] - img_aux[y1][x2] - img_aux[y2][x1] + img_aux[y1][x1]
                area = float((x2 - x1) * (y2 - y1))
                media = float(valor) / area
                img[row][col] = 1.0 if img[row][col] > (media*FATOR) else 0.0

    return img

def check_bordas(img, row, col):
    return row < img.shape[0] and row >= 0 and col < img.shape[1] and col >= 0


def rotula (img, largura_min, altura_min, n_pixels_min):
    result = []
    label = 2.0
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
                        
            if img[row][col] == 1:                
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
                
                label += 1.0
                
    return result
    

def inunda (blob, img, row, col):
            
    label = blob['label']
    img[row][col] = label
    blob['pixels'].append({ 'row': row, 'col': col })
    # print({ 'label': label, 'row': row, 'col': col })
                
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
    return range_permitido(img, row, col) and img[row][col] == 1 and pixel_nao_mapeado(img, row, col)

def pixel_nao_mapeado(img, row, col):
    return img[row][col] == 1.0

def range_permitido(img, row, col):
    return row >= 0 and row < img.shape[0] and col >= 0 and col < img.shape[1]


def calcula_total_arroz(componentes):

    tamanhos = [c['n_pixels'] for c in componentes]
    media = np.mean(tamanhos)

    total_arroz = 0
    for blob in componentes:
        tam = blob['n_pixels']
        n = tam / media
        # print({'tam': tam, 'n':n, 'round n': round(n)})
        total_arroz += round(n)

    return total_arroz



#===============================================================================

def main ():
    images = [
        '60', 
        '82', 
        # '114', 
        # '150', 
        # '205'
        ]
    for input_image in images:

        img = cv2.imread (input_image + '.bmp')
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()

        # É uma boa prática manter o shape com 3 valores, independente da imagem ser
        # 0ida ou não. Também já convertemos para float32.
        img = img.reshape ((img.shape [0], img.shape [1], img.shape [2]))
        img = img.astype (np.float32) / 255

        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        start_time = timeit.default_timer ()
        binarizada = limiarizacao_adaptativa(gray_scale)
        print ('Tempo: %f' % (timeit.default_timer () - start_time))
        cv2.imwrite (f'binarizada_{input_image}.png', binarizada*255)
        print(f'binarizada {input_image}.bmp done')

        start_time = timeit.default_timer ()
        componentes = rotula (binarizada, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
        n_componentes = len (componentes)
        print ('Tempo: %f' % (timeit.default_timer () - start_time))
        print ('%d componentes detectados.' % n_componentes)

        # Mostra os objetos encontrados.
        for c in componentes:
            cv2.rectangle (img, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

        cv2.imwrite (f'arroz_{input_image}.png', img*255)
        print(f'arroz {input_image}.bmp done')

        total = calcula_total_arroz(componentes)
        print(f'O total de arroz da imagem {input_image}.bmp é {total}!\n')

        cv2.waitKey ()
        cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================
