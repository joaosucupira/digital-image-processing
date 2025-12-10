from stdafx import *
from Descriptor import Descriptor

class ColorHistogram(Descriptor):

    # 1. ATRIBUTO DE CLASSE ESTATICO: Armazenará a matriz de custos (inicialmente None)
    matriz_custo = None
    
    # 2. MÉTODO DE CLASSE para calcular a matriz (agora estático)
    @classmethod
    def calcular_matriz_custo(cls, n_bins: int):
        
        tam_bins = 256 // n_bins  
        offset = tam_bins / 2 
        
        coordenadas = []

        # Simplificando a criação das coordenadas
        for b in range(n_bins):
            for g in range(n_bins):  
                for r in range(n_bins): 
                    val_r = (r * tam_bins) + offset
                    val_g = (g * tam_bins) + offset
                    val_b = (b * tam_bins) + offset
                    coordenadas.append([val_r, val_g, val_b])

        n_cores = len(coordenadas)
        # Usando NumPy para matrizes grandes é mais eficiente (opcional, mas recomendado)
        matriz_custos = np.zeros((n_cores, n_cores), dtype=np.float32)
        
        # Otimização: A matriz é simétrica, calculamos apenas metade
        for i in range(n_cores):
            cor_A = coordenadas[i]
            for j in range(i, n_cores): # Começa de 'i' para calcular metade superior
                cor_B = coordenadas[j]
                
                # Cálculo da Distância Euclidiana (Pitágoras 3D)
                diferenca_r = (cor_A[0] - cor_B[0]) ** 2
                diferenca_g = (cor_A[1] - cor_B[1]) ** 2
                diferenca_b = (cor_A[2] - cor_B[2]) ** 2
                
                distancia = math.sqrt(diferenca_r + diferenca_g + diferenca_b)
                
                # Armazena e usa a simetria
                matriz_custos[i][j] = distancia
                matriz_custos[j][i] = distancia # Garante a simetria
                
        return matriz_custos
    
    # 3. Bloco Estático de Inicialização da Matriz de Custo
    # Este bloco garante que a matriz seja calculada e armazenada APENAS UMA VEZ.
    # if matriz_custo is None:
    #     matriz_custo = calcular_matriz_custo(N_BINS)



    def __init__(self, img_path:str, bins:int=N_BINS):
        self.n_bins = bins
        super().__init__(img_path)
        self.descriptor = self.histograma_cor()
        
    
    def get_similarity(self, des:'ColorHistogram') -> float:
        return self.earth_movers_distancia(self.descriptor, des.descriptor)

    def fill_descriptor(self) -> None:
        pass

    def histograma_cor(self):

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

    def earth_movers_distancia(self, hist_base, hist_entrada):

        distancia = 0
        n_cores = self.n_bins**3
        hist_base = hist_base.astype(np.float32).reshape(n_cores,1)
        hist_entrada = hist_entrada.astype(np.float32).reshape(n_cores,1)
        matriz_custos = np.array(self.matriz_custo).astype(np.float32)

        distancia = cv2.EMD(hist_base,hist_entrada,cv2.DIST_USER, matriz_custos)

        return distancia


    