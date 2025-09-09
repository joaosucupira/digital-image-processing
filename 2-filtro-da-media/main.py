import sys
import timeit
import numpy as np
import cv2
INPUT_IMAGE = "assets/a01 - Original.bmp"
JANELA = 51

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

    # Retorna a imagem com o filtro de média aplicado.
    return img
#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    #Convertendo para float.
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = filtro_media_separavel(img,JANELA)

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', (img_out*255).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()