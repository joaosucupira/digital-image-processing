import sys
import time
import timeit
import numpy as np
import cv2
# INPUT_IMAGE = "assets/a01 - Original.bmp"
INPUT_IMAGE = "assets/b01 - Original.bmp"
JANELA = 7
FILTRO = 'ingenuo'
# FILTRO = 'separavel'
# FILTRO = 'integral'

def filtro_media_ingenuo(img, janela):
    """
    Aplica um filtro de média usando uma abordagem ingênua (força bruta).
    Para cada pixel, recalcula a soma de todos os pixels em sua vizinhança.

    Parâmetros:
        img: imagem de entrada (NumPy array de 3 canais).
        janela: tamanho da janela do filtro (deve ser um número ímpar).

    Valor de retorno:
        imagem filtrada (NumPy array).
    """
    start = time.perf_counter()

    # Obtém as dimensões da imagem.
    altura = img.shape[0]
    largura = img.shape[1]
    # Calcula o raio da janela para facilitar os cálculos de vizinhança.
    r_janela = janela // 2

    #Cria um buffer para calcular a media sem interferencia dos vizinhos ja calculados
    buffer = img.copy()

    # Itera sobre os 3 canais de cor da imagem (B, G, R).
    for z in range (3):
        # Itera sobre cada pixel da imagem, exceto as bordas que não comportam a janela.
        for y in range (r_janela, altura - r_janela):
            for x in range (r_janela, largura - r_janela):
                soma = 0
                # Itera sobre a janela de vizinhança do pixel (x, y).
                for i in range (y-r_janela,y+r_janela+1):
                    for j in range (x-r_janela,x+r_janela+1):
                        soma += buffer[i][j][z]
                # Calcula a média e atribui ao pixel correspondente na imagem de saída.
                img[y][x][z] = soma / (janela * janela)
    
    end = time.perf_counter()
    print(f'tempo = {end - start:.6f}s')

    return img

def filtro_media_separavel(img, janela):
    """
    Aplica um filtro de média usando um algoritmo separável e otimizado com janelas deslizantes.
    Primeiro, calcula a soma horizontal em um buffer e, em seguida, a soma vertical sobre esse buffer.

    Parâmetros:
        img: imagem de entrada (NumPy array).
        janela: tamanho da janela do filtro (deve ser um número ímpar).

    Valor de retorno:
        imagem filtrada (NumPy array).
    """
    start = time.perf_counter()

    # Obtém as dimensões da imagem.
    altura = img.shape[0]
    largura = img.shape[1]

    # Cria um buffer para armazenar os resultados intermediários (somas horizontais).
    # Utiliza a própria imagem de entrada (img) para armazenar o resultado.
    buffer = img.copy()

    # Calcula o raio da janela (metade do tamanho).
    r_janela = janela // 2

    # Itera sobre os 3 canais de cor da imagem (B, G, R).
    for z in range (3):
        # --- SOMA HORIZONTAL ---
        # Itera sobre cada linha da imagem.
        for y in range (altura):
            
            # Calcula a soma da primeira janela da linha.
            soma = 0
            for x in range (janela):
                soma += img[y][x][z]
            # Armazena a primeira soma na posição central da janela no buffer.
            buffer[y][r_janela][z] = soma
            
            # Desliza a janela pelo resto da linha.
            antigo = 0
            novo = janela
            for x in range (r_janela + 1, largura - r_janela):
                soma -= img[y][antigo][z]
                soma += img[y][novo][z]
                # Armazena a nova soma na posição atual no buffer.
                buffer [y][x][z] = soma
                antigo += 1
                novo += 1

        # ---SOMA VERTICAL E CÁLCULO DA MÉDIA ---
        # Itera sobre cada coluna da imagem.
        for x in range (r_janela,largura-r_janela):

            # Calcula a soma da primeira janela da coluna, usando os valores do buffer.
            soma = 0
            for y in range (janela):
                soma += buffer[y][x][z]
            # Calcula a média final e armazena na imagem de saída.
            img[r_janela][x][z] = soma / (janela * janela)

            antigo = 0
            novo = janela
            # Desliza a janela pelo resto da coluna.
            for y in range (r_janela + 1, altura - r_janela):
                soma -= buffer[antigo][x][z]
                soma += buffer[novo][x][z]
                # Calcula a média final e armazena na imagem de saída.
                img[y][x][z] = soma / (janela * janela)
                antigo += 1
                novo += 1
    end = time.perf_counter()
    print(f'tempo = {end - start:.6f}s')
    # Retorna a imagem com o filtro de média aplicado.
    return img

def get_integral(img):
    buffer = img.copy().astype(np.float32)
    
    width = buffer.shape[1]
    height = buffer.shape[0]
    
    for c in range(3):
        for y in range(height):
            for x in range(1, width):
                buffer[y, x][c] += buffer[y, x-1][c]
                
        for y in range(1, height):
            for x in range(width):
                buffer[y, x][c] += buffer[y-1, x][c]
                
    return buffer       

def filtro_media_integral(img, janela):
    
    buffer = get_integral(img)
    
    start = time.perf_counter()

    altura = img.shape[0]
    largura = img.shape[1]
    r_janela = janela // 2
    
    for c in range(3):

        for y in range (altura):
            for x in range (largura):
                
                superior = max (0 ,y - r_janela) - 1
                inferior = min (altura - 1, y + r_janela)
                esquerdo = max (0,x - r_janela) - 1
                direito = min (largura -1, x + r_janela)

                altura_janela = (inferior - (superior + 1) + 1)
                largura_janela = (direito - (esquerdo + 1) + 1)
                area = altura_janela * largura_janela
                
                A = buffer[inferior, direito][c] # soma canto inferior direito (dentro)
                
                B,C,D = 0,0,0
                if superior >= 0:
                    B = buffer[superior, direito][c] # reduz canto superior direito (+1 p cima)
                
                if esquerdo >= 0:
                    C = buffer[inferior, esquerdo][c]# reduz canto inferior esquerdo (+1 p esquerda)
                
                if superior >= 0 and esquerdo >= 0:
                    D = buffer[superior, esquerdo][c] # soma canto superior esquerdo (+1 p cima e p esquerda)
                
                media = (A - B - C + D) / area
                img[y, x][c] = media

    end = time.perf_counter()
    print(f'tempo = {end - start:.6f}s')
    return img

#===============================================================================
def filtro(img, janela):
    if (FILTRO == 'ingenuo'):
        return filtro_media_ingenuo(img, janela)
    elif (FILTRO == 'separavel'):
        return filtro_media_separavel(img, janela)
    elif (FILTRO == 'integral'):
        return filtro_media_integral(img, janela)
    else:
        raise ValueError("Erro de Valor da Macro: Selecione um valor valido para a macro")
def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    #Convertendo para float.
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    # img_out = filtro_media_separavel(img,JANELA)
    # img_out = filtro_media_integral(img, JANELA)
    
    img_out = filtro(img, JANELA)

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('out/02 - out.png', (img_out*255).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()