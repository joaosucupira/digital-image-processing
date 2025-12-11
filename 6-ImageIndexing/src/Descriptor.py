from stdafx import *

# COLOCAR ABC P FICAR POLIMORFICA
class Descriptor():
    def __init__(self, img_path: str, retrieve_desc: bool = False):
        
        self.image_path = img_path
        self.descriptor = []
        self.last = False
        self.retrieve = retrieve_desc
        self.img = cv2.imread(self.image_path)

    # cria nome de arquivo para guardar em txt
    def get_filename_desc(img_path: str) -> str:
        base = os.path.basename(img_path)
        return os.path.splitext(base)[0] + '.txt'
    

    def show_img(self):
        # img = cv2.imread(self.image_path)
        if self.img is not None:
            cv2.imshow('descriptor associated img', self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('Descriptor::show_img -> img is None')

    def save_info(self, dest: str, filename: str) -> None:
        feature_list = self.descriptor.flatten().tolist()
        descriptor_str = " ".join(["%.8f" % x for x in feature_list])
        
        with open(os.path.join(dest, filename), 'a', encoding='utf-8') as f:

            f.write(f'{self.image_path}; {descriptor_str}\n')

    def new_save_info(self, dest: str, filename: str) -> None:
        pass

    @abstractmethod
    def fill_descriptor(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_similarity(self, des: 'Descriptor') -> float:
        raise NotImplementedError

    def execute(self) -> None:
        self.fill_descriptor()
        
