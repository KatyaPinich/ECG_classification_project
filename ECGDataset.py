import torch
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from scipy.io import loadmat
from commons import *


class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data_frame = self.load_data()

    def load_data(self):
        data_frame = pd.read_csv(self.csv_path, header=None)
        return data_frame

    def split(self, test_ratio=0.2):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        for train_index, test_index in splitter.split(self.data_frame, self.data_frame.iloc[:, 1]):
            train_set = self.data_frame.loc[train_index]
            test_set = self.data_frame.loc[test_index]
        return train_set, test_set

    def get_data(self):
        return self.data_frame


class TrainTestSplitter:
    def __init__(self, csv_path, test_ratio):
        data_frame = pd.read_csv(csv_path)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        for train_index, test_index in splitter.split(data_frame, data_frame[:, 1]):
            self.train_set = data_frame.loc[train_index]
            self.test_set = data_frame.loc[test_index]

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set


class ECGDataset(Dataset):
    def __init__(self, data_frame, data_dir, transform=None):
        """
        Args:
            data_frame (pandas data frame): The data frame of the set.
            data_dir (string): Directory with all signals.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = data_frame
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ecg_path = Path(self.data_dir).joinpath(f'{self.data_frame.iloc[idx, 0]}.mat')
        signal = loadmat(ecg_path)['val'][0, :]

        if self.transform:
            signal = self.transform(signal)

        label = self.data_frame.iloc[idx, 1]
        label_id = class_ids.get(label)

        sample = {'ecg': signal, 'label': label_id}

        return sample


