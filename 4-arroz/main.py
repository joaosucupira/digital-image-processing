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

INPUT_IMAGE = 'assets/150.bmp'
AREA_MIN = 25
ALPHA = 3.0
MAX_IT = 20
JANELA_PERCENT = 0.15 #porcentagem em relacao ao tamanho da imagem
ARROZ = 1
FUNDO = 0
LIMITE_ESCURO = 0.68
LIMITE_APROX = 0.3
#===============================================================================

class Grao:
    def __init__(self, label, x, y):
        self.label = label
        self.n_pixels = 0
        self.area = 0
        self.isolado = False
        self.T = y
        self.L = x
        self.B = y
        self.R = x


def binariza (img):

    janela = int(min(img.shape[0], img.shape[1])*JANELA_PERCENT)
    if janela % 2 == 0:
        janela += 1
    
    buffer = cv2.GaussianBlur(img, (janela, janela),0)

    buffer = img - buffer
    
    buffer = cv2.normalize(buffer, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # cv2.imshow ('03 - diff', buffer)
    # cv2.imwrite ('out/03 - diff.png', (buffer*255).astype(np.uint8))
    
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
    graos = []

    for y in range(altura):
        for x in range(largura):
            # Se achou um pixel de um objeto que ainda não foi rotulado.
            if img[y][x] == ARROZ:
                # Inicia o flood fill para encontrar o componente inteiro.
                grao_achado = Grao(label, x, y)
                flood_fill(img, grao_achado)

                # Verifica se o componente atende aos critérios de tamanho.
                if verficar_grao(grao_achado, area_min):
                    graos.append(grao_achado)

                label += 0.01
    return graos
#-------------------------------------------------------------------------------
    
def definir_isolados(graos, alpha, max_it):

    gaus = 1.4826
    areas = np.sort(np.asarray([g.area for g in graos], dtype=np.uint32))
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

    for g in graos:
        if g.area in isolados:
            g.isolado = True

    return graos
#-------------------------------------------------------------------------------

def estimar_grudado(grudado, media):
    
    # parte inteira da estimativa baseada na média simples
    estimativa = int(grudado.n_pixels / media)
    
    # porcentagem da zona vazia daquela sessão local de arroz grudado
    escuro = grudado.area - grudado.n_pixels
    escuro_pctg = round(escuro/grudado.area, 2) 
    
    # parte decimal da estimativa
    e_truncada = int(grudado.n_pixels / media)
    e_exata = (grudado.n_pixels / media)
    parte_decimal = e_exata - e_truncada

    if (parte_decimal > LIMITE_APROX):
        estimativa += 1
        
    if (escuro_pctg > LIMITE_ESCURO):
        estimativa -= 1
    
    # if (escuro_pctg > 0.7):
    #     estimativa -= 1
    # print(e_exata - e_truncada)
    
    return estimativa

#-------------------------------------------------------------------------------

def estimar_total(graos):

    total = 0
    n_pixels_medio = 0
    
    isolados = 0
    clusters = 0

    grudados = []

    for g in graos:
        if(g.isolado):
            n_pixels_medio += g.n_pixels
            total += 1
        else:
            grudados.append(g)
            clusters += 1
    
    isolados = total
    media = n_pixels_medio / total
    
    for g in grudados:
        total += estimar_grudado(g, media)

    # n_pixels_medio /= total
    
    print(f'isolados = {isolados}; clusters = {clusters}')
    # for pixels in grudados:
    #     total += round (pixels / n_pixels_medio)

    return total
#-------------------------------------------------------------------------------   

def verficar_grao(grao, area_min):

    altura_obj = grao.B - grao.T + 1
    largura_obj = grao.R - grao.L + 1

    area_obj = altura_obj * largura_obj

    if(area_obj < area_min):
        return False
    
    grao.area = area_obj

    return True
#-------------------------------------------------------------------------------

def flood_fill(img, grao):

    altura = img.shape[0]
    largura = img.shape[1]
    label = grao.label
    x = grao.L
    y = grao.T
    
    pilha = [(x,y)]
    img[y][x] = label
    
    while(pilha):
        
        grao.n_pixels += 1
        x, y = pilha.pop()

        if(x > 0 and img[y][x-1] == ARROZ): # Vizinho da ESQUERDA
            pilha.append((x-1, y))
            img[y][x-1] = label
            grao.L = min(grao.L, x-1)
        if(y > 0 and img[y-1][x] == ARROZ): # Vizinho de CIMA
            pilha.append((x, y-1))
            img[y-1][x] = label
            grao.T = min(grao.T, y-1)
        if(x < largura-1 and img[y][x+1] == ARROZ): # Vizinho da DIREITA
            pilha.append((x+1, y))
            img[y][x+1] = label
            grao.R = max(grao.R, x+1)
        if(y < altura-1 and img[y+1][x] == ARROZ): # Vizinho de BAIXO
            pilha.append((x, y+1))
            img[y+1][x] = label
            grao.B = max(grao.B, y+1)
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
    # cv2.imshow ('01 - binarizada', img)
    # cv2.imwrite ('out/01 - binarizada.png', (img*255).astype(np.uint8))

    graos = rotula (img, AREA_MIN)
    n_graos = len (graos)
    print ('%d componentes detectados.' % n_graos)

    graos = definir_isolados(graos, ALPHA, MAX_IT)
    
    total = estimar_total(graos)
    print(f'Total de grãos estimado: {total}')

    # Mostra os objetos encontrados.
    for g in graos:
        cor = (0,1,0) if g.isolado else (0,0,1) # Verde para isolados, Vermelho para grudados
        cv2.rectangle (img_out, (g.L, g.T), (g.R, g.B), cor, 1)
        if g.n_pixels == 981 and g.area == 2312:
            cv2.rectangle (img_out, (g.L, g.T), (g.R, g.B), (1,0,0), 1)

    # cv2.imshow ('02 - out', img_out)
    # cv2.imwrite ('out/02 - out.png', (img_out*255).astype(np.uint8))
    # cv2.waitKey ()
    # cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
