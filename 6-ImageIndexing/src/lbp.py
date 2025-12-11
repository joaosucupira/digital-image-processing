from stdafx import *
from LocalBinaryPattern import LocalBinaryPattern

class App_test:
    def __init__(self, dataset=DATASET_PATH, input_image=INPUT_PATH, descriptors=DESCRIPTORS_PATH):
        self.descriptors = []
        self.scores = {}
        self.data_path = dataset
        self.input_path = input_image
        self.desc_path = descriptors
        self.execute()

    # BASE METHODS

    # register in txt file the descriptor information for the next run
    def save_descriptors(self, source: str, dest: str, filename: str) -> None:
        # clears previous content if there's already old registries

        with open(os.path.join(dest, filename), 'w') as f:
            f.write('')
        i = 0
        # acessing png from img folder
        for img in os.listdir(source):
            print(i)
            img_path = os.path.join(source, img)
            desc = LocalBinaryPattern(img_path)
            desc.save_info(dest, filename)
            i+=1

    # create previous Descriptor instances from the descriptors data folder 
    def retrieve_descriptors(self, source: str, filename: str) -> None:
        file_path = os.path.join(source, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                parts = line.split(';', 1)
                
                img_path = parts[0].strip()
                features_str = parts[1].strip()
                
                features_array = np.fromstring(features_str, sep=' ', dtype=np.float64)
                
                d = LocalBinaryPattern(img_path=img_path, retrieve_desc=True)
                d.descriptor = features_array

                self.descriptors.append(d)
                

    # rank up similarities between each descriptor
    def compare_similarities(self, source):
        input_desc = LocalBinaryPattern(source)

        for desc in self.descriptors:
            score = input_desc.get_similarity(desc)
            self.scores[desc.image_path] = score

        self.scores = dict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        
        
    # show up dataset images related to input
    def get_results(self):
        for i, (path, scor) in enumerate(self.scores.items()):
            if i >= 15:
                break
            img = cv2.imread(path)
            cv2.imshow(f'top {i} - {scor}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # TESTS
    def test_LBP(self):
        for desc in self.descriptors:
            # if there are no full zeros and no total dark images then this should print various positive numbers
            print(desc.descriptor)

    # ORCHESTRATION
    def execute(self):
        # self.save_descriptors(self.data_path, self.desc_path, FILE_LBP)
        self.retrieve_descriptors(self.desc_path, FILE_LBP)
        self.compare_similarities('../input/image.jpg')
        self.get_results()

        # testing:
        # self.test_LBP()
        # self.test_new_save(self.data_path, self.desc_path)

if __name__ == '__main__':
    a = App_test()