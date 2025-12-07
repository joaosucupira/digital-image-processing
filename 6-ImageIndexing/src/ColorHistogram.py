from stdafx import *
from Descriptor import Descriptor

class ColorHistogram(Descriptor):
    def __init__(self, img_path:str, color_range:int=128):
        super.__init__(img_path)

    def save_info(self, dest: str, filename: str) -> None:
        pass

    def get_similarity(self, des:'ColorHistogram') -> float:
        pass

    def fill_descriptor(self) -> None:
        pass
    