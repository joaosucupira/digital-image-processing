from stdafx import *

# COLOCAR ABC P FICAR POLIMORFICA
class Descriptor():
    cont_save = 1
    def __init__(self, img_path: str, retrieve_desc: bool = False):
        
        self.image_path = img_path
        self.descriptor = []
        self.retrieve = retrieve_desc
        self.img = cv2.imread(self.image_path)
    
    @abstractmethod
    def show_img(self, ranking, score)-> None:
        raise NotImplementedError

    def save_info(self, dest: str, filename: str) -> None:
        feature_list = self.descriptor.flatten().tolist()
        descriptor_str = " ".join(["%.8f" % x for x in feature_list])
        
        with open(os.path.join(dest, filename), 'a', encoding='utf-8') as f:

            f.write(f'{self.image_path}; {descriptor_str}\n')
        
        print(Descriptor.cont_save)
        Descriptor.cont_save += 1
            
    def new_save_info(self, dest: str, filename: str) -> None:
        pass

    @abstractmethod
    def get_similarity(self, des: 'Descriptor') -> float:
        raise NotImplementedError

    def execute(self) -> None:
        self.fill_descriptor()
        
