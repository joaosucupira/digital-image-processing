import sys
import time
import timeit
import numpy as np
import cv2
INPUT_IMAGE_1 = "assets/GT2.BMP"
INPUT_IMAGE_2 = "assets/Wind Waker GC.bmp"
LUMINOSO = 0.77
INTENSIDADE_BLOOM = 1.2
PRETO = 0
#===============================================================================
def main ():

    img = cv2.imread (INPUT_IMAGE_2).astype(np.float32) / 255.0
    img_original = img.copy()
    
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    
    altura = img.shape[0]
    largura = img.shape[1]

    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # img_grey = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    canal_brilho = img_hsv[:,:,2] 

    for i in range (altura):
        for j in range (largura):
            if(canal_brilho[i][j] < LUMINOSO):
                img[i][j][0] = PRETO
                img[i][j][1] = PRETO
                img[i][j][2] = PRETO


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
    
    bloom = gaussian_bloom(img)
    
    img_final = img_original + bloom
    
    # em vez de normalizar usamos a funcao de clipping do numpy p permitir que os valores acima de um sejam interpretados como queremos
    # ela corta os valores acima de um (branco puro) e abaixo de zero (escuridao eterna)
    img_final = np.clip(img_final, 0, 1)
    

    cv2.imshow ('03 - out', (img_final*255.0).astype(np.uint8))
    cv2.waitKey (0)
    cv2.imwrite ('out/03 - out.png', (img_final*255.0).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()