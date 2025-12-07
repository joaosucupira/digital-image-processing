from stdafx import *
from Descriptor import Descriptor

class App:
    def __init__(self, dataset=DATASET_PATH, input_image=INPUT_PATH, descriptors=DESCRIPTORS_PATH):
        self.descriptors = []
        self.similarities = []
        self.data_path = dataset
        self.input_path = input_image
        self.desc_path = descriptors
        self.execute()


    # register in txt file the descriptor information for the next run
    def save_descriptors(self, source: str, dest: str) -> None:
        # clears previous content if there's already old registries

        with open(os.path.join(dest, FILENAME), 'w') as f:
            f.write('')
        
        idx = 0
        # acessing png from img folder
        for img in os.listdir(source):
                
            
            img_path = os.path.join(source, img)

            desc = Descriptor(img_path)
            if idx == len(os.listdir(source)) - 1:
                desc.last = True

            desc.save_info(dest, FILENAME)
            idx+=1


    # create previous Descriptor instances from the descriptors data folder 
    def retrieve_descriptors(self, source: str, filename:str) -> None:
        descriptors = os.path.join(source, filename)
        with open(descriptors, 'r', encoding='utf-8') as f:
            content = f.read()
            retrievals = content.split(',')
            for r in retrievals:
                d = Descriptor(img_path=r)
                self.descriptors.append(d)

    # rank up similarities between each descriptor
    def compare_similarities(self):
        for d in self.descriptors:
            # print(d.image_path)
            d.show_img()

    # show up dataset images related to input
    def get_results(self):
        pass

    # script orchestration
    def execute(self):
        # self.save_descriptors(self.data_path, self.desc_path)
        self.retrieve_descriptors(self.desc_path, FILENAME)
        self.compare_similarities()
        self.get_results()