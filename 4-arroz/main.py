#===============================================================================
# Contagem de arroz
# AutoreS: João Teixeira e João Teixeira
#-------------------------------------------------------------------------------
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE = 'assets/60.bmp'
ALTURA_MIN = 5
LARGURA_MIN = 5
JANELA_PERCENT = 0.15 #porcentagem do tamanho da imagem
ARROZ = 1
FUNDO = 0
#===============================================================================

def binariza (img):

    janela = int(min(img.shape[0], img.shape[1])*JANELA_PERCENT)
    if janela % 2 == 0:
        janela += 1
    
    buffer = cv2.GaussianBlur(img, (janela, janela),0)

    buffer = img - buffer
    
    buffer = cv2.normalize(buffer, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img_binarizada = cv2.threshold((buffer * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
   
    buffer = (img_binarizada / 255.0).astype(np.float32)

    kernel = np.ones((3,3),np.float32)
    
    buffer = cv2.erode(buffer, kernel, iterations = 1)
    buffer = cv2.dilate(buffer, kernel, iterations = 1)

    return buffer
#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min):

    altura = img.shape[0]
    largura = img.shape[1]
    label = 1.01
    componentes = []

    for y in range(altura):
        for x in range(largura):
            # Se achou um pixel de um objeto que ainda não foi rotulado.
            if img[y][x] == ARROZ:
                # Inicia o flood fill para encontrar o componente inteiro.
                componente_achado = {'label': label, 'n_pixels': 0, 'T': y, 'L': x, 'B': y, 'R': x}
                flood_fill(img, label, x, y, componente_achado)

                # Verifica se o componente atende aos critérios de tamanho.
                if verficar_componente(componente_achado, largura_min, altura_min):
                    componentes.append(componente_achado)

                label += 0.01
    return componentes
#-------------------------------------------------------------------------------

def verficar_componente(componente, largura_min, altura_min):

    altura_obj = componente['B'] - componente['T'] + 1
    largura_obj = componente['R'] - componente['L'] + 1

    if(altura_obj < altura_min):
        return False
    if(largura_obj < largura_min):
        return False

    return True
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def flood_fill(img, label, x, y, componente):

    altura = img.shape[0]
    largura = img.shape[1]

    pilha = [(x,y)]
    img[y][x] = label
    
    while(pilha):
        
        componente['n_pixels'] += 1
        x, y = pilha.pop()

        if(x > 0 and img[y][x-1] == ARROZ): # Vizinho da ESQUERDA
            pilha.append((x-1, y))
            img[y][x-1] = label
            componente['L'] = min(componente['L'], x-1)
        if(y > 0 and img[y-1][x] == ARROZ): # Vizinho de CIMA
            pilha.append((x, y-1))
            img[y-1][x] = label
            componente['T'] = min(componente['T'], y-1)
        if(x < largura-1 and img[y][x+1] == ARROZ): # Vizinho da DIREITA
            pilha.append((x+1, y))
            img[y][x+1] = label
            componente['R'] = max(componente['R'], x+1)
        if(y < altura-1 and img[y+1][x] == ARROZ): # Vizinho de BAIXO
            pilha.append((x, y+1))
            img[y+1][x] = label
            componente['B'] = max(componente['B'], y+1)
#------------------------------------------------------------------------------- 

def main ():

     # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.astype (np.float32) / 255.0

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    img = binariza (img)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('out/01 - binarizada.png', (img*255).astype(np.uint8))

    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN)
    n_componentes = len (componentes)
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('out/02 - out.png', (img_out*255).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
