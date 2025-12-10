from stdafx import *
from Descriptor import Descriptor

class HOG(Descriptor):
    def __init__(self, img_path:str ,  retrieve_desc: bool = False):
        super().__init__(img_path, retrieve_desc)
        if(not self.retrieve):
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            self.img = cv2.resize(self.img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            self.hog = cv2.HOGDescriptor(
                _winSize=IMAGE_SIZE,         # Tamanho da janela (deve ser o tamanho da imagem redimensionada)
                _blockSize=HOG_BLOCK_SIZE,
                _blockStride=HOG_BLOCK_STRIDE,
                _cellSize=HOG_CELL_SIZE,
                _nbins=HOG_N_BINS
            )
            self.descriptor = (self.hog).compute(self.img)        #self.features = self.descriptor.compute(self.img)

    def get_similarity(self, des:'HOG') -> float:
        pass

    def fill_descriptor(self) -> None:
        pass
    