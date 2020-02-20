import torch
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from scipy.io import loadmat

ECG_PATH = 'datasets/ecg'

class_ids = {
    'N': 0,
    'O': 1,
    'A': 2,
    '~': 3
}


class TrainTestSplitter:
    def __init__(self, csv_path, test_ration):
        data_frame = pd.read_csv(csv_path)
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_ration, random_state=42)
        for train_index, test_index in split.split(data_frame, data_frame[:, 1]):
            self.train_set = data_frame.loc[train_index]
            self.test_set = data_frame.loc[test_index]

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set

class ECGDataset(Dataset):
    def __init__(self, train_test_splitter, root_dir, is_train=True):
        """
        Args:
            train_test_splitter (class): An instance of a TrainTestSplitter class
            root_dir (string): Directory with all the audio files
            is_train (bool): Indicates weather a training set is requested.
        """
        self.train_test_splitter = train_test_splitter
        self.is_train = is_train
        if self.is_train:
            self.data_frame = train_test_splitter.get_train_set()
        else:
            self.data_frame = train_test_splitter.get_test_set()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ecg_filepath = Path(self.root_dir).joinpath(f'{self.data_frame.iloc[idx, 0]}.mat')

        data = loadmat(ecg_filepath)

        label = self.data_frame.iloc[idx, 1]
        label_id = class_ids.get(label)

        sample = {'ecg': data, 'label': label_id}

        return sample


