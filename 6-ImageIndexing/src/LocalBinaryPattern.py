from stdafx import *
from Descriptor import Descriptor

class LocalBinaryPattern(Descriptor):
    def __init__(self, img_path:str, kernel: int = 3):
        super().__init__(img_path)
        points = kernel * kernel - 1
        self.descriptor = np.zeros(points * (points - 1) + 3, dtype=np.float64)

    def get_similarity(self, des:'LocalBinaryPattern') -> float:
        pass

    def fill_descriptor(self) -> None:
        pass