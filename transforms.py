import numpy as np
from filtering import *
from torch import from_numpy


class Normalize(object):
    """Normalize a signal"""

    def __call__(self, signal):
        return (signal - np.mean(signal)) / np.std(signal)


class Filter(object):
    def __init__(self, sampling_freq):
        self.sampling_freq = sampling_freq

    def __call__(self, signal):
        butter_filter = ButterFilter(sampling_freq=self.sampling_freq, order=3)
        signal_highpass = butter_filter.highpass(signal, 1)
        signal_bandstop = butter_filter.bandstop(signal_highpass, 58, 62)

        lowpass_butter = ButterFilter(sampling_freq=self.sampling_freq, order=4)
        signal_lowpass = lowpass_butter.lowpass(signal_bandstop, cutoff_freq=25)

        return signal_lowpass


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, signal):
        original_length = len(signal)

        if original_length < self.output_size:
            signal = np.concatenate((signal, np.zeros(shape=(1, self.output_size - 1))))
        elif original_length > self.output_size:
            signal = signal[0:self.output_size]

        return signal


class ToTensor(object):
    def __call__(self, signal):
        signal = signal.reshape(-1, 1)

        return from_numpy(signal)
