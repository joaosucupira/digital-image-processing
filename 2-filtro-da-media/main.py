import sys
import timeit
import numpy as np
import cv2
INPUT_IMAGE = "assets/b01 - Original.bmp"

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

    # --- PASSE 1: SOMA HORIZONTAL ---
    # Itera sobre cada linha da imagem.
    for y in range (altura):
        
        # Calcula a soma da primeira janela da linha.
        soma = 0
        for x in range (janela):
            soma += img[y][x]
        # Armazena a primeira soma na posição central da janela no buffer.
        buffer[y][r_janela] = soma
        
        # Desliza a janela pelo resto da linha.
        for x in range (r_janela + 1, largura - r_janela):
            primeiro = x - r_janela - 1
            ultimo = x + r_janela
            soma -= img[y][primeiro]
            soma += img[y][ultimo]
            # Armazena a nova soma na posição atual no buffer.
            buffer [y][x] = soma


    # --- PASSE 2: SOMA VERTICAL E CÁLCULO DA MÉDIA ---
    # Itera sobre cada coluna da imagem.
    for x in range (largura):

        # Calcula a soma da primeira janela da coluna, usando os valores do buffer.
        soma = 0
        for y in range (janela):
            soma += buffer[y][x]
        # Calcula a média final e armazena na imagem de saída.
        img[r_janela][x] = soma / (janela * janela)

        # Desliza a janela pelo resto da coluna.
        for y in range (r_janela + 1, altura - r_janela):
            primeiro = y - r_janela - 1
            ultimo = y + r_janela
            soma -= buffer[primeiro][x]
            soma += buffer[ultimo][x]
            # Calcula a média final e armazena na imagem de saída.
            img[y][x] = soma / (janela * janela)

    # Retorna a imagem com o filtro de média aplicado.
    return img
#===============================================================================

def main ():

     # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = filtro_media_separavel(img,11)

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', (img_out*255).astype(np.uint8))
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()