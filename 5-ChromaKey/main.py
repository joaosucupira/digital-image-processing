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
INPUT_IMAGE = 'assets/1.bmp'

#===============================================================================
def main():

    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR_BGR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    cv2.imshow ('0 - img', (img).astype(np.uint8))
    cv2.imwrite ('out/0 - img.png', (img).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()
#===============================================================================