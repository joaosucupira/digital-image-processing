import sys
import time
import timeit
import numpy as np
import cv2
INPUT_IMAGE_1 = "assets/GT2.BMP"
INPUT_IMAGE_2 = "assets/Wind Waker GC.bmp"
LUMINOSO = 0.55
PRETO = 0
#===============================================================================
def main ():

    img = cv2.imread (INPUT_IMAGE_2).astype(np.float32) / 255.0
    
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    
    altura = img.shape[0]
    largura = img.shape[1]
    
    img_grey = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    for i in range (altura):
        for j in range (largura):
            if(img_grey[i][j] < LUMINOSO):
                img[i][j][0] = PRETO
                img[i][j][1] = PRETO
                img[i][j][2] = PRETO


    
    cv2.imshow ('03 - out', (img*255.0).astype(np.uint8))
    cv2.waitKey (0)
    cv2.imwrite ('out/03 - out.png', (img*255.0).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()