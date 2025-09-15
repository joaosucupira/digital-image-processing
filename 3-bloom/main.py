import sys
import time
import timeit
import numpy as np
import cv2
INPUT_IMAGE_1 = "assets/GT2.BMP"
INPUT_IMAGE_2 = "assets/Wind Waker GC.bmp"

#===============================================================================
def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE_1)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    #Convertendo para float.
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    # img_out = filtro_media_separavel(img,JANELA)
    # img_out = filtro_media_integral(img, JANELA)

    cv2.imshow ('03 - out', img)
    cv2.imwrite ('out/03 - out.png', (img*255).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()