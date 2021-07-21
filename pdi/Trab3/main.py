
#===============================================================================
# Trabalho 3 - Bloom lighting
# Alunos: Caroline Rosa & Leonardo Trevisan
#===============================================================================

from os import altsep
import sys
import numpy as np
import cv2
import sys

#===============================================================================

INPUT_IMAGE =  'carro.bmp'
THRESHOLD = 0.5
ALPHA = 1
BETA= 0.1
    
def bright_pass(img):
    img_out = img.copy()
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img_out[row][col] = img[row][col] if img[row][col][1] > THRESHOLD else 0

    return img_out

def bloom_box_blur(img, original):
    blur = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    # Borra 4 vezes, com janelas de 16, 25, 36 e 49
    for round in range(4,8):
        blur += cv2.blur(img, (round*round, round*round))

    return original*ALPHA + blur*BETA

def bloom_gaussian(img, original):
    blur = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    sigma = 1
    # Borra 4 vezes, com janelas de 17, 21, 25 e 29
    for round in range(4,8): 
        blur += cv2.GaussianBlur(img, ((round*4)+1, (round*4)+1), sigma, sigma)
        sigma += 1

    return original*ALPHA + blur*BETA


#===============================================================================

def main ():
    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], img.shape [2]))
    img = img.astype (np.float32) / 255

    # Converte imagem RGB para HSL para fazer o bright-pass
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    bright_pass_hsl = bright_pass(imgHLS)
    # Converte de volta pra RGB
    bright_pass_img = cv2.cvtColor(bright_pass_hsl, cv2.COLOR_HLS2BGR)
    cv2.imwrite (f'bright_pass_img.png', bright_pass_img*255)
    print('Bright Pass done')

    bloom_img = bloom_box_blur(bright_pass_img, img)
    cv2.imwrite (f'bloom-box-blur.png', bloom_img*255)
    print('Box-Blur Bloom done')
    
    bloom_img = bloom_gaussian(bright_pass_img, img)
    cv2.imwrite (f'bloom-gaussian.png', bloom_img*255)
    print('Gaussian Bloom done')

    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================
