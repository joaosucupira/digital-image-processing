#===============================================================================
# Chorma Key
# AutoreS: João Teixeira e João Teixeira
#-------------------------------------------------------------------------------
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import numpy as np
import cv2

#===============================================================================
INPUT_IMAGE = 'assets/7.bmp'


#===============================================================================
def nivelVerde(img):
    
    imgG = np.zeros(img.shape[:2], dtype=np.float32)
    # Converte a imagem para um tipo de dado maior (int16) para evitar overflow
    img_calc = img.astype(np.float32) / 255.0
    # nivelVerde = Green - vazamento(Blue + Red)
    nivelVerde = img_calc[:, :, 1] - (img_calc[:, :, 0] + img_calc[:, :, 2])/1.5
    # Garante que os valores estejam no intervalo [0, 1]
    imgG = np.clip(nivelVerde, 0, 1)
    imgG = cv2.normalize(imgG, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)

    return imgG

#===============================================================================

#===============================================================================
def main():

    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR_BGR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    
    cv2.imshow ('original', (img).astype(np.uint8))
    cv2.waitKey ()

    imgG = nivelVerde(img)
    cv2.imshow ('nivelVerde', (imgG*255).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()
#===============================================================================
