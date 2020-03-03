from pathlib import Path
from ECGDataset import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from filtering import *
from dataset_transforms import *
from classifier import Classifier
from m3 import M3
from m5 import M5

DATA_PATH = 'datasets/ecg'
CSV_PATH = 'datasets/ecg/REFERENCE.csv'
sampling_freq = 300

data_loader = DataLoader(Path(CSV_PATH))
train_set, test_set = data_loader.split(test_ratio=0.2)

transform = Compose(
    [Normalize(),
     Filter(sampling_freq=sampling_freq),
     Rescale(output_size=9000),
     ToTensor()])

train_dataset = ECGDataset(train_set, DATA_PATH, transform)
test_dataset = ECGDataset(test_set, DATA_PATH, transform)

batch_size = 4
iterations = 6000
epochs = 100#int(iterations / (len(train_set) / batch_size))

#model = M3(num_classes=4)
model = M5(num_classes=4)
classifier = Classifier(model=model, state_path=f'./state_{epochs}_epochs.pth')
classifier.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=test_dataset)
classifier.predict(test_dataset, batch_size=batch_size)

signal_filename = train_set.iloc[0, 0]
signal_label = train_set.iloc[0, 1]
signal_path = Path(DATA_PATH).joinpath(f'{signal_filename}.mat')
raw_signal = loadmat(signal_path)['val'][0, :]
raw_signal = (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)

samples = 3000#len(raw_signal) - 1
t = np.linspace(0.0, samples / sampling_freq, samples, endpoint=False)

fig, axs = plt.subplots(2)
fig.suptitle(f'{signal_filename}-{signal_label}')

axs[0].plot(t, raw_signal[:samples])
axs[0].set_title('Raw Signal')

# Filter
butter_filter = ButterFilter(sampling_freq=sampling_freq, order=3)

signal_highpass = butter_filter.highpass(raw_signal, 1)

#axs[1].plot(t, signal_highpass[:samples])
#axs[1].set_title('High-Pass')

signal_bandstop = butter_filter.bandstop(signal_highpass, 58, 62)

#axs[2].plot(t, signal_bandstop[:samples])
#axs[2].set_title('Band-Stop')

lowpass_butter = ButterFilter(sampling_freq=sampling_freq, order=4)
signals_lowpass = lowpass_butter.lowpass(signal_bandstop, cutoff_freq=25)

axs[1].plot(t, signals_lowpass[:samples])
axs[1].set_title('Low-Pass')

plt.show()
