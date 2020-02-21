from ECGDataset import *

class TrainCommand:
    def __init__(self, data_loader, classifier):
        self.data_loader = data_loader
        self.classifier = classifier

    def execute(self):
        train_set, test_set = self.data_loader.split(test_ratio=0.2)
