import sys
import time
import timeit
import numpy as np
import cv2
INPUT_IMAGE_1 = "assets/GT2.BMP"
INPUT_IMAGE_2 = "assets/Wind Waker GC.bmp"
LUMINOSO = 170
#===============================================================================
def main ():

    img = cv2.imread (INPUT_IMAGE_1)
    
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    
    altura = img.shape[0]
    largura = img.shape[1]
    
    brigth_pass = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brigth_pass[:,:,2] = np.where(brigth_pass[:,:,2] < LUMINOSO, 0, brigth_pass[:,:,2])

    img_out = cv2.cvtColor(brigth_pass, cv2.COLOR_HSV2BGR)

    cv2.imshow ('03 - out', img_out)
    cv2.imwrite ('out/03 - out.png', img_out)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()