from stdafx import *
from ColorHistogram import ColorHistogram
from LocalBinaryPattern import LocalBinaryPattern
from HOG import HOG

class App:
    def __init__(self, dataset=DATASET_PATH, input_image=INPUT_PATH, descriptors=DESCRIPTORS_PATH):
        self.descriptors = []

        self.data_path = dataset
        self.input_path = input_image
        self.desc_path = descriptors
        self.execute()

    def save_HOG_descriptors(self, source: str, dest: str) -> None:

        with open(os.path.join(dest, FILE_HOG), 'w') as f:
            f.write('')
        
        idx = 0
        for img in os.listdir(source):
            
            img_path = os.path.join(source, img)

            desc = HOG(img_path)

            desc.save_info(dest, FILE_HOG)
            idx+=1
    
    def save_COR_descriptors(self, source: str, dest: str) -> None:

        with open(os.path.join(dest, FILE_COR), 'w') as f:
            f.write('')
        
        idx = 0
        for img in os.listdir(source):
                
            img_path = os.path.join(source, img)

            desc = ColorHistogram(img_path)
    
            desc.save_info(dest, FILE_COR)
            idx+=1

    def save_LBP_descriptors(self, source: str, dest: str) -> None:

        with open(os.path.join(dest, FILE_LBP), 'w') as f:
            f.write('')
        
        idx = 0
        for img in os.listdir(source):
                
            img_path = os.path.join(source, img)

            desc = LocalBinaryPattern(img_path)

            desc.save_info(dest, FILE_LBP)
            idx+=1

    def retrieve_HOG_descriptors(self, source: str, filename: str) -> None:
        file_path = os.path.join(source, filename)
        
        self.descriptors = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                parts = line.split(';', 1)
                
                img_path = parts[0].strip()
                features_str = parts[1].strip()
                
                features_array = np.fromstring(features_str, sep=' ', dtype=np.float32)
                
                d = HOG(img_path, retrieve_desc=True)
                d.descriptor = features_array

                self.descriptors.append(d)
    
    def retrieve_COR_descriptors(self, source: str, filename: str) -> None:
        file_path = os.path.join(source, filename)
        
        self.descriptors = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                parts = line.split(';', 1)
                
                img_path = parts[0].strip()
                features_str = parts[1].strip()
                
                features_array = np.fromstring(features_str, sep=' ', dtype=np.float32)
                
                d = ColorHistogram(img_path, retrieve_desc=True)
                d.descriptor = features_array

                self.descriptors.append(d)

    def retrieve_LBP_descriptors(self, source: str, filename: str) -> None:
        file_path = os.path.join(source, filename)
        
        self.descriptors = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                parts = line.split(';', 1)
                
                img_path = parts[0].strip()
                features_str = parts[1].strip()
                
                features_array = np.fromstring(features_str, sep=' ', dtype=np.float32)
                
                d = LocalBinaryPattern(img_path, retrieve_desc =True)
                d.descriptor = features_array

                self.descriptors.append(d)
                

    # rank up similarities between each descriptor
    def compare_similarities(self):
        for desc in self.descriptors:
            desc.show_img()

    # show up dataset images related to input
    def get_results(self):
        pass

    # TESTS
    def test_retrieve(self):
        np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.8f}'.format})

        for desc in self.descriptors:
            print(desc.descriptor)

    def test_new_save(self, source: str, dest: str) -> None:
        for img_file in os.listdir(source):

            img_path = os.path.join(source, img_file)

            desc_CH = ColorHistogram(img_path)
            desc_LBP = LocalBinaryPattern(img_path)
            desc_HOG = HOG(img_path)

            desc_CH.save_info()
            desc_LBP.save_info()
            desc_HOG.save_info()

    def execute(self):
        
        #self.save_HOG_descriptors(self.data_path, self.desc_path)
        #self.save_COR_descriptors(self.data_path, self.desc_path)
        
        self.retrieve_HOG_descriptors(self.desc_path, FILE_HOG)
        self.retrieve_COR_descriptors(self.desc_path, FILE_COR)

        

