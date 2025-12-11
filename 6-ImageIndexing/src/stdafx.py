# MODULES
import os
import json
import math
import cv2 
import numpy as np 
from pathlib import Path
from abc import ABC, abstractmethod


# CLASSES
from Descriptor import Descriptor

# MACROS

DATASET_PATH = '../img'
DESCRIPTORS_PATH = '../descriptors'
INPUT_PATH = '../input'
FILE_HOG = 'hog.txt'
FILE_LBP = 'lbp.txt'
N_BINS = 8

IMAGE_SIZE = (500, 500) # (Largura, Altura)

# Parâmetros HOG ajustados para imagens grandes (500x500)
# Vamos usar células maiores para manter o vetor de features pequeno.
HOG_CELL_SIZE = (20, 20)  # Célula de 20x20 pixels (500 é divisível por 20)
HOG_BLOCK_SIZE = (40, 40) # Bloco de 2x2 células (40x40 pixels)
HOG_BLOCK_STRIDE = (20, 20) # Passo de 1 célula (20x20 pixels)
HOG_N_BINS = 9 # Número padrão de bins de orientação

