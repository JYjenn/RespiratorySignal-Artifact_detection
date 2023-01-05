# system packages
# 3-party packages
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d, interpolate
# custom packages


def interpolation_1d(data, outPutLen, interpol_type='linear'):
    f1 = interp1d(range(0, len(data)), data, kind=interpol_type)
    inter1Peaks = f1(np.linspace(0, len(data) - 1, num=outPutLen))
    return inter1Peaks


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def make_edr(ECG, peaks, outPutLen, interpol_type = 'linear'):
    peaks_amp = ECG[peaks]
    peaks_amp = peaks_amp - np.mean(peaks_amp)
    f1 = interp1d(peaks, peaks_amp, kind=interpol_type)
    inter1Peaks = f1(np.linspace(peaks[0], peaks[-1], num=outPutLen))
    return inter1Peaks


def make_hr(peaks, outPutLen, interpol_type='linear',
            medifilt=0, remove_outliers=False, remove_outliers_idx=[0.3, 2], ecg_fs=100):
    if medifilt:
        diff_peak = signal.medfilt(np.diff(peaks), kernel_size=medifilt)
    else:
        diff_peak = np.diff(peaks)

    if remove_outliers:
        start_out = int(remove_outliers_idx[0] * ecg_fs)
        end_out = int(remove_outliers_idx[1] * ecg_fs)
        diff_peak = list(map(lambda x: x if (x > start_out) and (x < end_out) else np.nan, diff_peak))
        diff_peak = fill_nan(np.array(diff_peak))

    f1 = interp1d(peaks[0:-1], diff_peak, kind=interpol_type)
    inter1Peaks = f1(np.linspace(peaks[0], peaks[-2], num=outPutLen))
    return inter1Peaks


def smoothing(data, order = 33):
    data = signal.savgol_filter(data, polyorder=3, window_length=order)
    data = signal.savgol_filter(data, polyorder=3, window_length=order)
    data = signal.savgol_filter(data, polyorder=3, window_length=order)
    return data


def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    # handing with first nan
    idx_first = 0
    idx_tail = len(A) - 1
    while np.isnan(A[idx_first]):
        idx_first += 1
    # handing with tail nan
    while np.isnan(A[idx_tail]):
        idx_tail -= 1
    # interpolation start & end
    if (idx_first == 0) & (idx_tail == len(A) - 1):
        pass
    else:
        A[idx_tail::] = A[idx_tail]
        A[0:idx_first] = A[idx_first]
    # interpolation nan data
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B
