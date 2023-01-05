# system packages
# 3-party packages
import numpy as np
from scipy import signal, stats
import pywt as pw
# custom packages


def extractWaveLetFeature(data, wavelets='sym3', levels=8, mode=None,
                          detrend=False):
    """

    extraction of feature from waveleted signal
    wavelet function list --> pywt.wavelist()
    featureName = ['mean',
                   'std',
                   'entropy'
                   'kurtosis'
                   'skewness']
    """
    ## signal detrend exclude low frequency noise
    if detrend:
        data = signal.detrend(data)
    else:
        pass

    if mode:
        waveletData = pw.wavedec(data, wavelet=wavelets, level=levels, mode=mode)
    else:
        waveletData = pw.wavedec(data, wavelet=wavelets, level=levels)

    feature = []
    for idx, i in enumerate(waveletData):
        feature.append(np.mean(i))
        feature.append(np.std(i))
        feature.append(stats.kurtosis(i))
        feature.append(stats.skew(i))

        if all([x == i[0] for x in i]):
            feature.append(0)
        else:
            i_mu, i_std = np.mean(i), np.std(i)
            feature.append(stats.entropy(stats.norm.pdf(i, i_mu, i_std)))

    return np.array(feature)
