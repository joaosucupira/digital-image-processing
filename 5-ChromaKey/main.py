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
def nivelVerde(img):
    
    imgG = np.zeros(img.shape[:2], dtype=np.float32)
    
    #Gera o nivel de verde de cada pixel [0(muito verde),2(sem verde)]
    imgG = 1 + np.maximum(img[:, :, 0], img[:, :, 2]) - img[:,:,1]
    
    imgG = cv2.normalize(imgG, None, 0, 1, cv2.NORM_MINMAX)
    cv2.imshow ('nivelVerde', (imgG*255).astype(np.uint8))
    cv2.waitKey ()

    return imgG

#===============================================================================

# Aplica a máscara criada a partir de dado limiar

def aniquilaVerde(img, mascara):
    LIMIAR = 0.7

    aniquilado = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    aniquilado = aniquilado.astype(np.float32) / 255.0

    # Aplica a mascara no canal de luminosidade
    aniquilado[:, :, 1] *= mascara

    # Escurece o canal de saturação
    aniquilado[:, :, 2] = 0

    aniquilado = cv2.cvtColor(aniquilado, cv2.COLOR_HLS2BGR)

    # Cria uma máscara booleana para os pixels não-verdes
    mascara_nao_verde = mascara >= LIMIAR
    # Usa a máscara para copiar os pixels da imagem original para a imagem aniquilada de forma vetorizada
    aniquilado[mascara_nao_verde] = img[mascara_nao_verde]

    cv2.imshow ('semVerde', (aniquilado*255).astype(np.uint8))
    cv2.waitKey ()

    return aniquilado

#===============================================================================

# 
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
    
    # Cria nova para ajudar do fundo à imagem
    
    fundo_t = cv2.copyMakeBorder(
        fundo,
        0,
        margem_inferior,
        0,
        margem_direita,
        cv2.BORDER_REFLECT
    )
    
    return fundo_t
    

#===============================================================================


def chromaVerde(img, mascara):
    
    altura = img.shape[0]
    largura = img.shape[1]
    
    chroma_key = np.zeros_like(img) 
    
    fundo = _trataFundo(img)
    
    for y in range(altura):
        for x in range(largura):
             chroma_key[y, x] = (img[y, x] * mascara[y, x]) + (fundo[y, x] * (1 - mascara[y, x]))
             
    return chroma_key
    

#===============================================================================
def main():

    for i in range(1):
        
        img = cv2.imread(INPUT_IMAGE[i], cv2.IMREAD_COLOR) 
        
        if img is None:
            print ('Erro abrindo a imagem.\n')
            sys.exit ()
        
        img = img.astype(np.float32) / 255.0 

        verde = nivelVerde(img)
        
        dessaturado = aniquilaVerde(img, verde) 
        
        chroma_key = chromaVerde(dessaturado, verde) 
        
        img_display = chroma_key
        cv2.imwrite ('out/chromed_%d.png' % i, (chroma_key * 255).astype(np.uint8))
        cv2.imshow ('chroma_key', (img_display * 255).astype(np.uint8)) 
        
        cv2.waitKey ()
        cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()
#===============================================================================
