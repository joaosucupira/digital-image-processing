from stdafx import *

class Descriptor:
    def __init__(self, img_path: str):
        # fix 1: removing line break so code wont break (pun)
        self.image_path = img_path.replace('\n','')
        self.descriptor = []
        self.last = False

    def show_img(self):
        img = cv2.imread(self.image_path)
        if img is not None:
            cv2.imshow('descriptor associated img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('Descriptor::show_img -> img is None')
    
    def save_info(self, dest: str, filename: str) -> None:
        try:
            with open(os.path.join(dest, filename), 'a', encoding='utf-8') as f:
                if self.last:
                    # fix 2: removing last error causing comma
                    f.write(f'{self.image_path}\n')
                else:
                    f.write(f'{self.image_path},\n')


        except FileNotFoundError:
            print('Descriptor::save_info -> dir does not exist')

    # @abstractmethod
    def fill_descriptor(self) -> None:
        pass

    # @abstractmethod
    def get_similarity(self, des: 'Descriptor') -> float:
        pass

    def execute(self) -> None:
        self.fill_descriptor()