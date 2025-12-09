import numpy as np
import cv2
import math

self.n_bins = 8

def calcular_matriz_custo():

    tam_bins = 256 // self.n_bins  
    offset = tam_bins / 2     
    
    # Lista para guardar os "endereços" das 512 cores
    coordenadas = []

    # A ordem dos loops (B fora, R dentro) simula o comportamento
    # do ravel() em arrays criados com indexing='ij'

    for b in range(self.n_bins):
        for g in range(self.n_bins):     
            for r in range(self.n_bins): 
                
                # Transforma índice em valor de cor
                val_r = (r * tam_bins) + offset
                val_g = (g * tam_bins) + offset
                val_b = (b * tam_bins) + offset
                
                # Guarda a trinca (R, G, B) na lista
                coordenadas.append([val_r, val_g, val_b])

    n_cores = len(coordenadas)
    
    # Cria uma matriz 512x512 preenchida com zeros
    matriz_custos = [[0.0] * n_cores for _ in range(n_cores)]
    
    # Loop para a LINHA da matriz (Cor A)
    for i in range(n_cores):
        cor_A = coordenadas[i]
        
        # Loop para a COLUNA da matriz (Cor B)
        for j in range(n_cores):
            cor_B = coordenadas[j]
            
            # Cálculo da Distância Euclidiana (Pitágoras 3D)
            # d = raiz( (r2-r1)² + (g2-g1)² + (b2-b1)² )
            diferenca_r = (cor_A[0] - cor_B[0]) ** 2
            diferenca_g = (cor_A[1] - cor_B[1]) ** 2
            diferenca_b = (cor_A[2] - cor_B[2]) ** 2
            
            distancia = math.sqrt(diferenca_r + diferenca_g + diferenca_b)
            
            # Armazena na matriz
            matriz_custos[i][j] = distancia
            
    return matriz_custos


def histograma_cor():

    n_cores = self.n_bins **3
    tam_bins = 256 // self.n_bins

    histograma =  np.zeros(n_cores)
    
    img_quantizada = self.img // tam_bins

    b,g,r = img_quantizada[:,:,0].astype(np.int32),  img_quantizada[:,:,1].astype(np.int32) , img_quantizada[:,:,2].astype(np.int32)

    #Mistura o r,g,b em um único canal
    img_planificada = (b * (self.n_bins ** 2) + g*self.n_bins + r) 

    for i in range (img_planificada.shape[0]):
        for j in range (img_planificada.shape[1]):
            histograma[img_planificada[i][j]] += 1

    #Normaliza o histograma
    histograma = histograma.astype(np.float32)
    histograma /= histograma.sum()

    return histograma

def earth_movers_distancia(hist_base, hist_entrada, matriz_custos):

    distancia = 0
    n_cores = self.n_bins**3
    hist_base = hist_base.astype(np.float32).reshape(n_cores,1)
    hist_entrada = hist_entrada.astype(np.float32).reshape(n_cores,1)
    matriz_custos = np.array(matriz_custos).astype(np.float32)

    distancia = cv2.EMD(hist_base,hist_entrada,cv2.DIST_USER, matriz_custos)

    return distancia

