#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'assets/60.bmp'
THRESHOLD = 0.7
ALTURA_MIN = 5
LARGURA_MIN = 5
N_PIXELS_MIN = 5
THRESHOLD = 0.05
JANELA = 20
SIGMA = 4
ARROZ = 1
FUNDO = 0
#===============================================================================

def binariza (img, threshold):

    diff = img - cv2.GaussianBlur(img, (0, 0), 3.5)

    return np.where(diff > threshold, ARROZ, FUNDO).astype(np.float32)
#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):

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
                if verficar_componente(componente_achado, largura_min, altura_min, n_pixels_min):
                    componentes.append(componente_achado)

                label += 0.01
    return componentes
#-------------------------------------------------------------------------------

def verficar_componente(componente, largura_min, altura_min, n_pixels_min):

    altura_obj = componente['B'] - componente['T'] + 1
    largura_obj = componente['R'] - componente['L'] + 1

    if(componente['n_pixels'] < n_pixels_min):
        return False
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

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.astype (np.float32) / 255.0

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('out/01 - binarizada.png', (img*255).astype(np.uint8))

    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
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
