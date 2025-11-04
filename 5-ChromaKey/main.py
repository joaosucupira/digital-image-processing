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
INPUT_IMAGE = [
    'assets/0.bmp',
    'assets/1.bmp',
    'assets/2.bmp',
    'assets/3.bmp',
    'assets/4.bmp',
    'assets/5.bmp',
    'assets/6.bmp',
    'assets/7.bmp',
    'assets/8.bmp',
]

BACKGROUND_IMAGE = 'assets/cachorro_boboca.jpg'
VERDE = 120.0

#===============================================================================

def geraNivelVerde(img):
    
    imgG = np.zeros(img.shape[:2], dtype=np.float32)
    
    #Gera o nivel de verde de cada pixel [0(muito verde),2(sem verde)]
    imgG = 1 + np.maximum(img[:, :, 0], img[:, :, 2]) - img[:,:,1]
    #Volta para o range [0,1]
    imgG = cv2.normalize(imgG, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imshow ('NivelVerde', (imgG*255.0).astype(np.uint8))
    cv2.waitKey ()

    return imgG

#===============================================================================


def aniquilaVerde(img, alpha):

    margem = 0.08

    threshold = cv2.threshold((alpha * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0

    limearFundo  = np.clip(threshold - margem, 0.0, 1.0)   # fundo
    limearFrente  = np.clip(threshold + margem, 0.0, 1.0)   # frente

    matte = np.clip(alpha, limearFundo, limearFrente).astype(np.float32)
    matte = cv2.normalize(matte, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    matte = np.clip(np.nan_to_num(matte, nan=0.0), 0.0, 1.0).astype(np.float32)
    matte = np.power(matte, 0.8)

    img_frente = img * matte[:, :, None]

    # # debug:
    print("w[mean,max]=", float(matte.mean()), float(matte.max()))
    cv2.imshow('matte', (matte*255).astype(np.uint8))
    cv2.imshow('frente', (img_frente*255).astype(np.uint8)); 
    cv2.waitKey(1)

    return img_frente, matte

#===============================================================================

def _trataFundo(img):
    
    # Fundo que vamos usar para preencher o verde
    fundo = cv2.imread(BACKGROUND_IMAGE, cv2.IMREAD_COLOR)
    
    if (fundo is None):
        print('ERRO AO ABRIR IMAGEM...')
        sys.exit()
        
    fundo = fundo.astype(np.float32) / 255.0
    
    altura_img = img.shape[0]
    largura_img = img.shape[1]
    
    altura_fundo = fundo.shape[0]
    largura_fundo = fundo.shape[1]
        
    margem_inferior = altura_img - altura_fundo if (altura_img > altura_fundo) else 0
    margem_direita = largura_img - largura_fundo if (largura_img > largura_fundo) else 0
    
    fundo_t = cv2.copyMakeBorder(
        fundo,
        0,
        margem_inferior,
        0,
        margem_direita,
        cv2.BORDER_REFLECT
    )
    
    return fundo_t[:altura_img, :largura_img]
    

#===============================================================================


def chroma(frente, fundo, alpha):

    chroma_key = frente + (fundo * (1 - alpha[:,:,None]))

    return chroma_key
    

#===============================================================================
def main():

    for i in range(len(INPUT_IMAGE)):
        
        img = cv2.imread(INPUT_IMAGE[i], cv2.IMREAD_COLOR) 
        
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        
        img = img.astype(np.float32) / 255.0 
        
        fundo = _trataFundo(img)
        alpha = geraNivelVerde(img)
        frente,alpha = aniquilaVerde(img, alpha)
        
        chroma_key = chroma(frente, fundo, alpha) 
        
        cv2.imwrite ('out/chromed_%d.png' % i, (chroma_key * 255).astype(np.uint8))
        cv2.imshow ('chroma_key', (chroma_key * 255).astype(np.uint8)) 
        
        cv2.waitKey ()
        cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()
#===============================================================================
