from stdafx import *

class Descriptor:
    def __init__(self, img_path: str):
        self.image_path = img_path
        self.descriptor = []
    
    def save_info(self, dest: str, filename: str) -> None:
        try:
            with open(os.path.join(dest, filename), 'a', encoding='utf-8') as f:
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