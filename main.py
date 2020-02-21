from classifier import *
from m3 import *
from ECGDataset import *

classifier = None
num_classes = 4

model = M3(num_classes=num_classes)

classifier = Classifier(model, '')

splitter = TrainTestSplitter()
