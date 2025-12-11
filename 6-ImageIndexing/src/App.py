from stdafx import *
from ColorHistogram import ColorHistogram
from LocalBinaryPattern import LocalBinaryPattern
from HOG import HOG

class App:
    def __init__(self, dataset=DATASET_PATH, input=INPUT_PATH_FILE, descriptors=DESCRIPTORS_PATH):
        self.datasetHogs = []
        self.datasetCores = []
        self.datasetLbps = []
        self.inputHog = HOG(input)
        self.inputCor = ColorHistogram(input)
        self.inputLbp = LocalBinaryPattern(input)
        self.data_path = dataset
        self.input_path_file = input
        self.desc_path = descriptors

        #Dicionarios de Ranking com o objeto e a pontuaçao (ordenar de menor para maior)
        self.rankingHog = []
        self.rankingCor = []
        self.rankingLbp = []
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
        
        self.datasetHogs = []
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

                self.datasetHogs.append(d)
    
    def retrieve_COR_descriptors(self, source: str, filename: str) -> None:
        file_path = os.path.join(source, filename)
        
        self.datasetCores = []
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

                self.datasetCores.append(d)

    def retrieve_LBP_descriptors(self, source: str, filename: str) -> None:
        file_path = os.path.join(source, filename)
        
        self.datasetLbps = []
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

                self.datasetLbps.append(d)
    
    def save (self):
        # self.save_HOG_descriptors(self.data_path, self.desc_path)
        # self.save_COR_descriptors(self.data_path, self.desc_path)
        self.save_LBP_descriptors(self.data_path, self.desc_path)
    
    def retrieve(self):
        # self.retrieve_HOG_descriptors(self.desc_path, FILE_HOG)
        # self.retrieve_COR_descriptors(self.desc_path, FILE_COR)
        self.retrieve_LBP_descriptors(self.desc_path, FILE_LBP)

    def gerarHankings(self):
        
        self.rankingHog = []
        self.rankingCor = []
        self.rankingLbp = []
        
        for hog in self.datasetHogs:
            dist = hog.get_similarity(self.inputHog)
            self.rankingHog.append([dist, hog])

        for cor in self.datasetCores:
            dist = cor.get_similarity(self.inputCor)
            self.rankingCor.append([dist, cor])

        for lbp in self.datasetLbps:
            dist = lbp.get_similarity(self.inputLbp)
            self.rankingLbp.append([dist, lbp])

        self.rankingHog.sort(key=lambda x: x[0])
        self.rankingCor.sort(key=lambda x: x[0])
        self.rankingLbp.sort(key=lambda x: x[0])            

    def get_results(self, top_n = 3):

        self.gerarHankings()

        def exibir_ranking(lista_ranking):

            for i, (score, descritor) in enumerate(lista_ranking[:top_n]):
                descritor.show_img(i,score)

        exibir_ranking(self.rankingHog)
        exibir_ranking(self.rankingCor)
        exibir_ranking(self.rankingLbp)

        return

    def test_retrieve(self):
        np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.8f}'.format})

        for desc in self.datasetHogs:
            print(desc.descriptor)


    def execute(self):
        
        # self.save()
        self.retrieve()
        self.get_results(6)


        

