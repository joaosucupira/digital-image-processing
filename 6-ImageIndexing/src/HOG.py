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
            self.descriptor = (self.hog).compute(self.img)

    def get_similarity(self, des:'HOG') -> float:
        return np.linalg.norm(self.descriptor - des.descriptor) #soma as distancias euclidianas de cada célula

    def fill_descriptor(self) -> None:
        pass

    def show_img(self, ranking, score):
        
        msg = "TOP " + str(ranking+1) + " HOG - " + str(score)
        if self.img is not None:
            cv2.imshow(msg, self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('Descriptor::show_img -> img is None')
    