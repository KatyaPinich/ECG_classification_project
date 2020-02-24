from scipy.signal import butter, lfilter


class ButterFilter:
    def __init__(self, sampling_freq, order):
        self.sampling_freq = sampling_freq
        self.nyquist_freq = self.sampling_freq / 2
        self.order = order

    def lowpass(self, signal, cutoff_freq):
        normalized_cutoff = cutoff_freq / self.nyquist_freq
        b, a = butter(self.order, normalized_cutoff, 'lowpass', analog=False)
        return lfilter(b, a, signal)

    def highpass(self, signal, cutoff_freq):
        normalized_cutoff = cutoff_freq / self.nyquist_freq
        b, a = butter(self.order, normalized_cutoff, 'highpass', analog=False)
        return lfilter(b, a, signal)

    def bandstop(self, signal, low_cutoff_freq, high_cutoff_freq):
        normalize_low = low_cutoff_freq / self.nyquist_freq
        normalized_high = high_cutoff_freq / self.nyquist_freq
        b, a = butter(self.order, [normalize_low, normalized_high], 'bandstop', analog=False)
        return lfilter(b, a, signal)