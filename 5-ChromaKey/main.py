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
INPUT_IMAGE = 'assets/5.bmp'


#===============================================================================
def nivelVerde(img):
    
    imgG = np.zeros(img.shape[:2], dtype=np.float32)
    img_calc = img.astype(np.float32) / 255.0
    
    #Gera o nivel de verde de cada pixel [0(muito verde),2(sem verde)]
    imgG = 1 + np.maximum(img_calc[:, :, 0], img_calc[:, :, 2]) - img_calc[:,:,1]
    
    cv2.imshow ('nivelVerde', (imgG*255/2.0).astype(np.uint8))
    cv2.waitKey ()

    return imgG

#===============================================================================
def bordas(imgG):

    buffer = np.clip(imgG, 0, 1)
    buffer = np.where(buffer < 0.9, 0, 1)
    cv2.imshow ('clip', (buffer*255).astype(np.uint8))
    cv2.waitKey ()
    canny = cv2.Canny((buffer*255).astype(np.uint8), 40, 80, L2gradient= 1)
    return canny
#===============================================================================
def main():

    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR_BGR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    
    cv2.imshow ('original', (img).astype(np.uint8))
    cv2.waitKey ()

    imgG = nivelVerde(img)

    borda = bordas(imgG)
    cv2.imshow('bordas', (borda).astype(np.uint8))
    cv2.waitKey ()

    
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()
#===============================================================================
