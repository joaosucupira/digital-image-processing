from stdafx import *

# COLOCAR ABC P FICAR POLIMORFICA
class Descriptor():
    def __init__(self, img_path: str):
        # fix 1: removing line break so code wont break (pun)
        self.image_path = img_path.replace('\n','')
        self.descriptor = []
        self.last = False
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

    @abstractmethod
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
    # def save_info(self, dest:str, filename: str, desc_type: str = None):
    #     desc_filename = self.get_filename_desc(self.image_path)
    #     file_path = os.path.join(dest, desc_filename)
        
    #     data_to_save = {
    #         "type": desc_type,
    #         "features": self.features.tolist() 
    #     }
    #     with open(file_path, 'a', encoding='utf-8') as f:
    #         f.write(json.dumps(data_to_save) + '\n')
    
    # fazer prototipo para salvar cada grupo de informacoes da imagem (de todos os descritores) em um arquivo
    # numero de arquivos txt = numero de imagens no dataset
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
        
