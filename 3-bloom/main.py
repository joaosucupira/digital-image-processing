import sys
import time
import timeit
import numpy as np
import cv2
INPUT_IMAGE_1 = "assets/GT2.BMP"
INPUT_IMAGE_2 = "assets/Wind Waker GC.bmp"
LUMINOSO = 0.77
PRETO = 0
JANELA = 3
REPETICOES = 3
#===============================================================================


def bloxbloom(mascara, janela, reps):

    borrada = np.zeros_like(mascara)
    kernel = (0,0)
    sum =  np.zeros_like(mascara)
    for j in range (4):

        kernel = (janela,janela)
        borrada = cv2.blur(mascara, kernel)

        for i in range (reps):
            borrada = cv2.blur(borrada, kernel)

        cv2.imshow ('03 - out', (borrada*255.0).astype(np.uint8))
        cv2.waitKey (0)
        janela = janela*2 + 1
        sum += borrada

    return np.clip(sum, 0, 1)


#===============================================================================

def main ():

    img = cv2.imread (INPUT_IMAGE_1).astype(np.float32) / 255.0
    
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
            if(img[i][j][0] < LUMINOSO and img[i][j][1] < LUMINOSO and img[i][j][2] < LUMINOSO):
                img_mascara[i][j][0] = PRETO
                img_mascara[i][j][1] = PRETO
                img_mascara[i][j][2] = PRETO


    img_mascara = bloxbloom (img_mascara, JANELA, REPETICOES)
    
    cv2.imshow ('03 - out', (img_mascara*255.0).astype(np.uint8))
    cv2.waitKey (0)

    img += img_mascara
    img = np.clip(img, 0, 1)

    cv2.imshow ('03 - out', (img*255.0).astype(np.uint8))
    cv2.waitKey (0)
    cv2.imwrite ('out/03 - out.png', (img*255.0).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()