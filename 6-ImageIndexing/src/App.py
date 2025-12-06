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


    # fill up the array of descriptors using img folder when they weren't stored yet
    def save_descriptors(self, source: str, dest: str) -> None:
        # clears previous content if there's already old registries

        with open(os.path.join(dest, FILENAME), 'w') as f:
            f.write('')
            
        for img in os.listdir(source):
            img_path = os.path.join(source, img)
            desc = Descriptor(img_path)
            desc.save_info(dest, FILENAME)


    # create Descriptor instances from the descriptors folder 
    def retrieve_descriptors(self, source: str) -> None:
        pass

    # rank up similarities between each descriptor
    def compare_similarities(self):
        pass

    # show up dataset images related to input
    def get_results(self):
        pass

    # script orchestration
    def execute(self):
        self.save_descriptors(self.data_path, self.desc_path)
        self.retrieve_descriptors(self.desc_path)
        self.compare_similarities()
        self.get_results()