#===============================================================================
# Contagem de arroz
# AutoreS: João Teixeira e João Teixeira
#-------------------------------------------------------------------------------
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import numpy as np
import cv2
from operator import itemgetter

#===============================================================================

INPUT_IMAGE = 'assets/205.bmp'
AREA_MIN = 25
ALPHA = 3.0
MAX_IT = 20
JANELA_PERCENT = 0.15 #porcentagem em relacao ao tamanho da imagem
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

def rotula (img, area_min):

    altura = img.shape[0]
    largura = img.shape[1]
    label = 1.01
    componentes = []

    for y in range(altura):
        for x in range(largura):
            # Se achou um pixel de um objeto que ainda não foi rotulado.
            if img[y][x] == ARROZ:
                # Inicia o flood fill para encontrar o componente inteiro.
                componente_achado = {'label': label, 'n_pixels': 0, 'area':0, 'isolado':False, 'T': y, 'L': x, 'B': y, 'R': x}
                flood_fill(img, label, x, y, componente_achado)

                # Verifica se o componente atende aos critérios de tamanho.
                if verficar_componente(componente_achado, area_min):
                    componentes.append(componente_achado)

                label += 0.01
    return componentes
#-------------------------------------------------------------------------------
    
def definir_isolados(componentes, alpha, max_it):

    gaus = 1.4826
    areas = np.sort(np.asarray([c['area'] for c in componentes], dtype=np.uint32))
    isolados = areas.copy()

    for _ in range(max_it):

        mediana = np.median(isolados)
        mad = np.median(np.abs(isolados - mediana))

        if mad == 0:
            break

        limite = mediana + alpha * mad * gaus
        cortados = isolados <= limite

        if cortados.all():
            break

        isolados = isolados[cortados]

    for c in componentes:
        if c['area'] in isolados:
            c['isolado'] = True

    return componentes
#-------------------------------------------------------------------------------

def total_graos(componentes):

    total = 0
    n_pixels_medio = 0

    grudados = []

    for c in componentes:
        if(c['isolado']):
            n_pixels_medio += c['n_pixels']
            total += 1
        else:
            grudados.append(c['n_pixels'])

    n_pixels_medio /= total

    for pixels in grudados:
        total += round (pixels / n_pixels_medio)

    return total
#-------------------------------------------------------------------------------   

def verficar_componente(componente, area_min):

    altura_obj = componente['B'] - componente['T'] + 1
    largura_obj = componente['R'] - componente['L'] + 1

    area_obj = altura_obj * largura_obj

    if(area_obj < area_min):
        return False
    
    componente['area'] = area_obj

    return True
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

    componentes = rotula (img, AREA_MIN)
    n_componentes = len (componentes)
    print ('%d componentes detectados.' % n_componentes)

    componentes = definir_isolados(componentes, ALPHA, MAX_IT)
    
    total_de_graos = total_graos(componentes)
    print(f'Total de grãos estimado: {total_de_graos}')

    # Mostra os objetos encontrados.
    for c in componentes:
        cor = (0,1,0) if c['isolado'] else (0,0,1) # Verde para isolados, Vermelho para grudados
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), cor, 1)

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('out/02 - out.png', (img_out*255).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
