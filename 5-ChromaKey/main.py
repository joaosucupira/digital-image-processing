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
    
    img = img.astype(np.float32)
    b, g, r = img[...,0], img[...,1], img[...,2]

    verdice = 1 + np.maximum(b,r) - g
    verdice = np.clip(verdice, 0.0, 1.0)

    cv2.imshow ('NivelVerde', (verdice*255.0).astype(np.uint8))
    cv2.waitKey ()

    return verdice

#===============================================================================


def aniquilaVerde(img, alpha):
    
    margem = 0.1
    sigma = 0.5

    # 2) Suavizar
    alpha_suave = cv2.GaussianBlur(alpha, (0, 0), sigma)

    # 3) Otsu precisa de 8 bits; converte só para o cálculo do limiar
    t = cv2.threshold((alpha_suave * 255.0).astype(np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]/ 255.0

    # 4) Smoothstep
    t_bg = float(np.clip(t - margem, 0.0, 1.0))
    t_fg = float(np.clip(t + margem, 0.0, 1.0))

    den  = max(t_fg - t_bg, 1e-6)
    x = np.clip((alpha_suave - t_bg) / den, 0.0, 1.0)
    matte = x * x * (3.0 - 2.0 * x)

    # 5) Feather leve
    matte = cv2.GaussianBlur(matte, (0, 0), sigma)

    img_frente = img.astype(np.float32) * matte[:, :, None]

    geraNivelVerde(img_frente)

    print(f"t(otsu)={t:.4f}  matte[mean,max]={float(matte.mean()):.4f},{float(matte.max()):.4f}")
    cv2.imshow('matte', (matte*255).astype(np.uint8))
    cv2.imshow('frente', np.clip(img_frente*255,0,255).astype(np.uint8))
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
