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
DATASET_PATH = '../img'
DESCRIPTORS_PATH = '../descriptors'
INPUT_PATH = '../input'
FILENAME = 'descriptors.txt'