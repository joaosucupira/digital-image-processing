import sys
import timeit
import numpy as np
import cv2
INPUT_IMAGE = "assets/a01 - Original.bmp"

def main ():

     # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = img

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', (img_out*255).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()