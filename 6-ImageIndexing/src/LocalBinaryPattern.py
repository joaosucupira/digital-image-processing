from stdafx import *
from Descriptor import Descriptor

class LocalBinaryPattern(Descriptor):
    def __init__(self, img_path:str, radius:int=1, points:int=8, method:str='uniform'):
        super.__init__(img_path)
        self.descriptor = np.zeros(points * (points - 1) + 3, dtype=np.float64)
        
    def save_info(self, dest: str, filename: str) -> None:
        pass

    def get_similarity(self, des:'LocalBinaryPattern') -> float:
        pass

    def fill_descriptor(self) -> None:
        pass