import sys
from pathlib import Path

import matplotlib.pyplot as plt

from dataset_transforms import *
from classifier import Classifier
from ECGDataset import *
from m5 import M5

DATA_PATH = 'datasets/ecg'
CSV_PATH = 'datasets/ecg/REFERENCE.csv'
sampling_freq = 300


def main():
    args = sys.argv
    if len(args) < 4:
        raise Exception(f'usage:  main.py <mode> <csv_path> <data_dir>')

    mode = args[1]
    csv_path = Path(args[2])
    if not csv_path.is_file():
        raise Exception(f'{csv_path} does not exist.')
    data_dir = args[3]
    if not Path(data_dir).is_dir():
        raise Exception(f'{data_dir} does not exist.')

    data_loader = DataLoader(csv_path)
    train_set, test_set = data_loader.split(test_ratio=0.2)

    transform = Compose(
        [Normalize(),
         Filter(sampling_freq=sampling_freq),
         Rescale(output_size=9000),
         ToTensor()])

    train_dataset = ECGDataset(train_set, data_dir, transform)
    test_dataset = ECGDataset(test_set, data_dir, transform)

    batch_size = 4
    epochs = 32

    model = M5(num_classes=4)
    classifier = Classifier(model=model, state_path=f'./state_{epochs}_epochs_1.pth')

    if mode == 'fit':
        # Fit model on data
        train_loss_history, val_loss_history = classifier.fit(train_dataset, batch_size=batch_size, epochs=epochs,
                                                              validation_data=test_dataset)

        plt.figure()
        plt.title(f'Model Loss for {epochs} epochs')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(train_loss_history, label='train')
        plt.plot(val_loss_history, label='test')
        plt.legend()
        plt.show()
    elif mode == 'predict':
        classifier.predict(test_dataset, batch_size=batch_size)
    else:
        raise Exception(f'{mode} is not a supported mode.')


if __name__ == '__main__':
    main()

