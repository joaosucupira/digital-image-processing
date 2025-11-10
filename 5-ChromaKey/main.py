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

#===============================================================================

def geraNivelVerde(img):
    
    img = img.astype(np.float32)
    b, g, r = img[...,0], img[...,1], img[...,2]

    verdice = 1 + np.maximum(b,r) - g
    verdice = np.clip(verdice, 0.0, 1.0)

    return verdice

#===============================================================================

def aniquilaVerde(img, verdice):

    alpha = cv2.normalize(verdice, None, 0, 1, cv2.NORM_MINMAX)
    alpha = np.where(alpha < 1e-5, 0, alpha)
    alpha = np.clip(alpha*1.5 - 0.5, 0, 1)#aumenta contraste

    aniquilado = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)

    #Saturacao
    aniquilado[:,:,2] *= alpha
    aniquilado = cv2.cvtColor(aniquilado, cv2.COLOR_HLS2BGR)
    
    alpha2 = geraNivelVerde(aniquilado.astype(np.float32))
    aniquilado[:,:,1] = np.where(alpha2 < 1, aniquilado[:,:,1] - (1-alpha2),  aniquilado[:,:,1])

    alpha = np.where((alpha2 < 1), np.clip(alpha + (1 - alpha2), 0, 1), alpha)
    alpha = np.clip(alpha*1.5 - 0.2, 0, 1) #aumenta contraste

    cv2.imshow('alpha', (alpha* 255).astype(np.uint8))
    #cv2.imshow('aniquilado', (aniquilado * 255).astype(np.uint8))

    return aniquilado, alpha

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


def chroma(frente, fundo, verdice):

    chroma_key = frente*verdice[:,:,None] + (fundo * (1 - verdice[:,:,None]))

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
        verdice = geraNivelVerde(img)
        frente,verdice = aniquilaVerde(img, verdice)
        
        chroma_key = chroma(frente, fundo, verdice) 
        
        cv2.imwrite ('out/chromed_%d.png' % i, (chroma_key * 255).astype(np.uint8))
        cv2.imshow ('chroma_key', (chroma_key * 255).astype(np.uint8)) 
        
        cv2.waitKey ()
        cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()
#===============================================================================
