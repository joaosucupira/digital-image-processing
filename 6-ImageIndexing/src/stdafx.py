# MODULES
import os
import json
import cv2 #type:ignore
import numpy as np #type:ignore
from pathlib import Path
from abc import ABC, abstractmethod


# CLASSES
from Descriptor import Descriptor

# MACROS
DATASET_PATH = 'C:/Users/jasuc/Desktop/Projects/PDI/6-ImageIndexing/img'
DESCRIPTORS_PATH = 'C:/Users/jasuc/Desktop/Projects/PDI/6-ImageIndexing/descriptors'
INPUT_PATH = 'C:/Users/jasuc/Desktop/Projects/PDI/6-ImageIndexing/input'
FILENAME = 'descriptors.txt'