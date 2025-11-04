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

    cv2.imshow ('NivelVerde', (verdice*255.0).astype(np.uint8))
    cv2.waitKey ()

    return verdice

#===============================================================================

def aniquilaVerde(img, verdice):
    
    margem = 0.1
    sigma = 1

    #Borra
    verdice_borrada = cv2.GaussianBlur(verdice, (0, 0), sigma)

    otsu = cv2.threshold((verdice_borrada * 255.0).astype(np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]/ 255.0

    #Gera curva suavizada de verdice
    limearFundo = float(np.clip(otsu - margem, 0.0, 1.0))
    limearFrente = float(np.clip(otsu + margem, 0.0, 1.0))
    diff = max(limearFrente - limearFundo, 1e-6)
    x = np.clip((verdice_borrada - limearFundo) / diff, 0.0, 1.0)
    verdice = x * x * (3.0 - 2.0 * x)

    #Tenta diminuir a verdice das bordas
    verdice = cv2.GaussianBlur(verdice, (0, 0), sigma)
    
    img_frente = img.astype(np.float32) * verdice[:, :, None]

    #Debug
    geraNivelVerde(img_frente)
    print(f"t(otsu)={otsu:.4f}  verdice[mean,max]={float(verdice.mean()):.4f},{float(verdice.max()):.4f}")
    cv2.imshow('verdice', (verdice*255).astype(np.uint8))
    cv2.imshow('frente', np.clip(img_frente*255,0,255).astype(np.uint8))
    cv2.waitKey(1)

    return img_frente, verdice


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

    chroma_key = frente + (fundo * (1 - verdice[:,:,None]))

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
