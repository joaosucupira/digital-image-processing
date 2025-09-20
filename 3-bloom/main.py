import sys
import time
import timeit
import numpy as np
import cv2
INPUT_IMAGE_1 = "assets/GT2.BMP"
INPUT_IMAGE_2 = "assets/Wind Waker GC.bmp"
LUMINOSO = 0.57
PRETO = 0
JANELA = 15
REPETICOES = 4
#===============================================================================


def bloxbloom(mascara, janela, reps):

    kernel = (15, 15)
    borrada_1 = cv2.blur(mascara, kernel)

    for i in range (4):
        borrada_1 = cv2.blur(borrada_1, kernel)

    cv2.imshow ('03 - out', (borrada_1*255.0).astype(np.uint8))
    cv2.waitKey (0)

    kernel = (30,30)
    borrada_2 = cv2.blur(mascara, kernel)

    for i in range (4):
        borrada_2 = cv2.blur(borrada_2, kernel)

    cv2.imshow ('03 - out', (borrada_2*255.0).astype(np.uint8))
    cv2.waitKey (0)

    kernel = (60, 60)
    borrada_3 = cv2.blur(mascara, kernel)

    for i in range (4):
        borrada_3 = cv2.blur(borrada_3, kernel)
    
    cv2.imshow ('03 - out', (borrada_3*255.0).astype(np.uint8))
    cv2.waitKey (0)

    kernel = (120, 120)
    borrada_4 = cv2.blur(mascara, kernel)

    for i in range (4):
        borrada_4 = cv2.blur(borrada_4, kernel)

    cv2.imshow ('03 - out', (borrada_4*255.0).astype(np.uint8))
    cv2.waitKey (0)
    return borrada_1 + borrada_2 + borrada_3 + borrada_4


#===============================================================================

def main ():

    img = cv2.imread (INPUT_IMAGE_2).astype(np.float32) / 255.0
    
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    
    altura = img.shape[0]
    largura = img.shape[1]

    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    img_mascara = img.copy()

    img_grey = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    for i in range (altura):
        for j in range (largura):
            if(img_grey[i][j] < LUMINOSO):
                img_mascara[i][j][0] = PRETO
                img_mascara[i][j][1] = PRETO
                img_mascara[i][j][2] = PRETO


    img_mascara = bloxbloom (img_mascara, JANELA, REPETICOES)
    
    cv2.imshow ('03 - out', (img_mascara*255.0).astype(np.uint8))
    cv2.waitKey (0)
    img += img_mascara

    cv2.imshow ('03 - out', (img*255.0).astype(np.uint8))
    cv2.waitKey (0)
    cv2.imwrite ('out/03 - out.png', (img*255.0).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()