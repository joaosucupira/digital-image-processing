from stdafx import *
from Descriptor import Descriptor

class ColorHistogram(Descriptor):

    matriz_custo = None
    
    def __init__(self, img_path: str, bins: int = COLOR_N_BINS, retrieve_desc: bool = False):
        self.n_bins = bins
        self.n_cores = bins ** 3
        self.tam_bins = 256 // bins
        
        super().__init__(img_path, retrieve_desc)
        
        if ColorHistogram.matriz_custo is None:
            ColorHistogram.matriz_custo = self.calcular_matriz_custo(self.n_bins)
        
        if(not self.retrieve):
            self.descriptor = self.histograma_cor()
        
    def get_similarity(self, des: 'ColorHistogram') -> float:
        return self.earth_movers_distancia(
            self.descriptor, 
            des.descriptor, 
            ColorHistogram.matriz_custo
        )

    def histograma_cor(self):
        img_quantizada = (self.img // self.tam_bins).astype(np.int32)

        b = img_quantizada[:,:,0]
        g = img_quantizada[:,:,1]
        r = img_quantizada[:,:,2]

        indices = (b * (self.n_bins ** 2) + g * self.n_bins + r)

        histograma = np.bincount(indices.flatten(), minlength=self.n_cores)

        histograma = histograma.astype(np.float32)
        soma = histograma.sum()
        if soma > 0:
            histograma /= soma

        return histograma

    def earth_movers_distancia(self, hist_base, hist_entrada, matriz_custo):
        hist_base = hist_base.astype(np.float32).reshape(self.n_cores, 1)
        hist_entrada = hist_entrada.astype(np.float32).reshape(self.n_cores, 1)
        
        resultado = cv2.EMD(hist_base, hist_entrada, cv2.DIST_USER, matriz_custo)
        
        return resultado[0] 
    
    @classmethod
    def calcular_matriz_custo(cls, n_bins: int):
        tam_bins = 256 // n_bins  
        offset = tam_bins / 2 
        
        coordenadas = []
        for b in range(n_bins):
            for g in range(n_bins):
                for r in range(n_bins):
                    val_b = (b * tam_bins) + offset
                    val_g = (g * tam_bins) + offset
                    val_r = (r * tam_bins) + offset
                    coordenadas.append([val_r, val_g, val_b])

        coordenadas = np.array(coordenadas, dtype=np.float32)
        
        diff = coordenadas[:, np.newaxis, :] - coordenadas[np.newaxis, :, :]
        
        matriz_custos = np.sqrt(np.sum(diff ** 2, axis=-1))
                
        return matriz_custos.astype(np.float32)