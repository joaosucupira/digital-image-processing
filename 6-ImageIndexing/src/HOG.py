from stdafx import *
from Descriptor import Descriptor

class HOG(Descriptor):
    def __init__(self, img_path:str):
        super().__init__(img_path)
        self.descriptor = cv2.HOGDescriptor()
        self.features = self.descriptor.compute(self.img)

    def save_info(self, dest: str, filename: str) -> None:
        super().save_info(dest, filename)
        pass



    def get_similarity(self, des:'HOG') -> float:
        pass

    def fill_descriptor(self) -> None:
        pass
    