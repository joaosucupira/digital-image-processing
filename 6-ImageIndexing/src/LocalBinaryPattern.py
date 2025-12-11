from stdafx import *
from Descriptor import Descriptor

class LocalBinaryPattern(Descriptor):
    def __init__(self, img_path:str, kernel: int = 3, retrieve_desc: bool = False):
        super().__init__(img_path, retrieve_desc)
        if not self.retrieve:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            self.img = cv2.equalizeHist(self.img)
            self.img = cv2.resize(self.img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            self.img = cv2.GaussianBlur(self.img, (3, 3), 0)
            self.radius = kernel // 2
            self.n_points = kernel * kernel - 1
            self.descriptor = np.zeros(256, dtype=np.float64)
            self.fill_descriptor()

    def fill_descriptor(self) -> None:
        height, width = self.img.shape
        r = self.radius
        
        center = self.img[r:height-r, r:width-r]
        lbp_image = np.zeros_like(center, dtype=np.uint8)
        
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                
                neighbor = self.img[r+dy:height-r+dy, r+dx:width-r+dx]
                lbp_image = (lbp_image << 1) | (neighbor >= center).astype(np.uint8)
        
        hist = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
        self.descriptor = hist.flatten()
        self.descriptor = self.descriptor / (np.sum(self.descriptor) + 1e-7)

    def get_similarity(self, des:'LocalBinaryPattern') -> float:
        dist = np.linalg.norm(self.descriptor - des.descriptor)
        return 1.0 / (1.0 + dist)