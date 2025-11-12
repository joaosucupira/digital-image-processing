# =============================================================================
# Chroma Key
# Autores: João Teixeira e João Teixeira
# UTFPR
# =============================================================================

import sys
import os
import numpy as np
import cv2
from typing import Tuple

# -----------------------------------------------------------------------------
INPUT_IMAGE = [
    "assets/0.bmp",
    "assets/1.bmp",
    "assets/2.bmp",
    "assets/3.bmp",
    "assets/4.bmp",
    "assets/5.bmp",
    "assets/6.bmp",
    "assets/7.bmp",
    "assets/8.bmp",
]

BACKGROUND_IMAGE = "assets/cachorro_boboca.jpg"
OUT_DIR = "out"
# -----------------------------------------------------------------------------

def geraNivelVerde(img: np.ndarray) -> np.ndarray:
    """Calcula o 'verdice' (nível de verde) por pixel no intervalo [0, 1]."""
    
    img = img.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    verdice = 1 + np.maximum(b, r) - g

    return np.clip(verdice, 0.0, 1.0)
# -----------------------------------------------------------------------------

def aniquilaVerde(img: np.ndarray, verdice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    #Aumenta contraste e borra um pouco (mantendo borda)
    alpha = cv2.normalize(verdice, None, 0, 1, cv2.NORM_MINMAX)
    alpha = np.where(alpha < 1e-5, 0, alpha)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.5)
    alpha = np.clip(alpha * 2 - 0.6, 0, 1)
    alpha = np.power(alpha, 2)
    
    #Morfologia para melhorar bordas
    kernel = np.ones((3, 3), np.uint8)
    alpha_u8 = (alpha * 255).astype(np.uint8)
    alpha_u8 = cv2.dilate(alpha_u8, kernel, iterations=1)
    alpha_u8 = cv2.erode(alpha_u8, kernel, iterations=1)
    alpha = alpha_u8.astype(np.float32) / 255.0

    #Retira verde (ou quase)
    aniquilado = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    aniquilado[:, :, 2] *= alpha #Luminancia
    aniquilado[:, :, 1] *= alpha #Saturação
    aniquilado = cv2.cvtColor(aniquilado, cv2.COLOR_HLS2BGR)

    #Subitrai o verde que ficou
    spill = 1 - geraNivelVerde(aniquilado.astype(np.float32))
    aniquilado[:, :, 1] -= spill

    #Melhora o alpha para evitar transparencia
    alpha -= spill
    alpha = np.clip(alpha * 1.2 - 0.1, 0, 1)

    return aniquilado, alpha
# -----------------------------------------------------------------------------

def _trataFundo(img: np.ndarray) -> np.ndarray:
    """Carrega e adapta o fundo ao tamanho da imagem de frente."""
    
    fundo = cv2.imread(BACKGROUND_IMAGE, cv2.IMREAD_COLOR)
    if fundo is None:
        print("ERRO AO ABRIR IMAGEM...")
        sys.exit()

    fundo = fundo.astype(np.float32) / 255.0
    h_img, w_img = img.shape[:2]
    h_bg, w_bg = fundo.shape[:2]

    margem_inf = max(h_img - h_bg, 0)
    margem_dir = max(w_img - w_bg, 0)

    fundo_t = cv2.copyMakeBorder(
        fundo, 0, margem_inf, 0, margem_dir, borderType=cv2.BORDER_REFLECT
    )
    
    return fundo_t[:h_img, :w_img]
# -----------------------------------------------------------------------------

def chroma(frente: np.ndarray, fundo: np.ndarray, verdice: np.ndarray) -> np.ndarray:
    """Composição linear usando o 'verdice' como máscara."""
    
    return frente * verdice[:, :, None] + (fundo * (1 - verdice[:, :, None]))
# -----------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, path in enumerate(INPUT_IMAGE):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print("Erro abrindo a imagem.")
            sys.exit()

        img = img.astype(np.float32) / 255.0
        fundo = _trataFundo(img)
        verdice = geraNivelVerde(img)
        frente, verdice = aniquilaVerde(img, verdice)

        chroma_key = chroma(frente, fundo, verdice)

        cv2.imwrite(os.path.join(OUT_DIR, f"chromed_{i}.png"), (chroma_key * 255).astype(np.uint8))
        cv2.imshow("chroma_key", (chroma_key * 255).astype(np.uint8))

        cv2.waitKey()
        cv2.destroyAllWindows()
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
