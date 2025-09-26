#===============================================================================
# Bloom
#-------------------------------------------------------------------------------
# AutoreS: João Teixeira e João Teixeira
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import time
import timeit
import numpy as np
import cv2
INPUT_IMAGE_1 = "assets/GT2.BMP"
INPUT_IMAGE_2 = "assets/Wind Waker GC.bmp"
LUMINOSO = 0.6
INTENSIDADE_BLOOM = 0.9
PRETO = 0
JANELA = 11
REPETICOES = 5
#===============================================================================


def bloxbloom(mascara, janela = JANELA, reps = REPETICOES, intensity = INTENSIDADE_BLOOM):

    sum =  np.zeros_like(mascara)
    for j in range (4):

        kernel = (janela,janela)
        borrada = cv2.blur(mascara, kernel)

        for i in range (reps):
            borrada = cv2.blur(borrada, kernel)

        janela = janela*2 + 1
        sum += borrada

    sum = cv2.normalize(sum, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return sum * intensity


#===============================================================================

def gaussian_bloom(img, intensity=INTENSIDADE_BLOOM):
    k_size_1 = (51, 51)
    k_size_2 = (121, 121)
    k_size_3 = (181, 181)
    k_size_4 = (291, 291)
    img_blur_1 = cv2.GaussianBlur(img.copy(), k_size_1, 0)
    img_blur_2 = cv2.GaussianBlur(img.copy(), k_size_2, 0)
    img_blur_3 = cv2.GaussianBlur(img.copy(), k_size_3, 0)
    img_blur_4 = cv2.GaussianBlur(img.copy(), k_size_4, 0)
    
    img_blur = img_blur_1 + img_blur_2 + img_blur_3 + img_blur_4
    img_blur = cv2.normalize(img_blur, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_blur * intensity

def main ():

    img = cv2.imread (INPUT_IMAGE_2).astype(np.float32) / 255.0
    img_original = img.copy()
    
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    
    altura = img.shape[0]
    largura = img.shape[1]

    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for i in range (altura):
        for j in range (largura):
            if(img_hsv[i][j][2] < LUMINOSO):
                img[i][j][0] = PRETO
                img[i][j][1] = PRETO
                img[i][j][2] = PRETO
    
    bloom_gaussian = gaussian_bloom(img)
    bloom_blox = bloxbloom(img)
    
    img_final_gaussian = img_original + bloom_gaussian
    img_final_blox = img_original + bloom_blox
    
    img_final_gaussian = (np.clip(img_final_gaussian, 0, 1)*255.0).astype(np.uint8)
    img_final_blox = (np.clip(img_final_blox, 0, 1)*255.0).astype(np.uint8)
    
    cv2.imwrite ('out/03 - out-gaussian.png', img_final_gaussian)
    cv2.imwrite ('out/03 - out-blox.png', img_final_blox)

    cv2.imshow ('03 - out-gaussian', img_final_gaussian)
    cv2.imshow ('03 - out-blox',img_final_blox)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()