# system packages
import os
# 3-party packages
import numpy as np
import biosppy.signals.tools as st
from matplotlib import pyplot as plt
# custom packages
from libs.signals import interpolation_1d
from libs.signal_search import bio_signal_marker
from libs.read_data import read_edf_file, get_edf_label


if __name__ == '__main__':
    ## data frame load
    data_path = 'E:\public_data\Stanford Technology Analytics and Genomics in Sleep\stages\psg\edf'
    label_path = 'E:\public_data\Stanford Technology Analytics and Genomics in Sleep\stages\psg\label'
    subject_names = list(map(lambda x: x.split('.')[0], os.listdir(data_path)))
    ## EDF 라벨이 제각각이므로 처리해줘야함
    subject_name = subject_names[99]
    ecg_label = get_edf_label(subject_name, ('ECG', 'EKG', 'ecg', 'ekg', 'Heartrate'), data_path=data_path)
    abdo_label = get_edf_label(subject_name, ('ABD', 'Abd', 'abd'), data_path=data_path)
    data_frame = read_edf_file(subject_name, data_path=data_path, label_path=label_path,
                               sig_labels=(ecg_label, abdo_label))
    ## load abdo
    abdo = data_frame['signal']['abdo']
    abdo_fs = data_frame['info']['sampling_rate_sig']['abdo']
    # apnea label
    apnea = data_frame['label']['apnea']
    apnea = interpolation_1d(apnea, int(len(apnea) / 10 * abdo_fs))
    ## filter(option)
    order = int(0.3 * abdo_fs)
    abdo_f, _, _ = st.filter_signal(signal=abdo,
                                    ftype='FIR',
                                    band='bandpass',
                                    order=order,
                                    frequency=[0.1, 2],
                                    sampling_rate=abdo_fs)
    ## plot abdo & apnea signal
    plt.figure(figsize=(10, 5.5))
    plt.subplot(211); plt.plot(abdo_f); plt.title(subject_name)
    plt.subplot(212); plt.plot(apnea); plt.title('Apnea check')
    plt.tight_layout()

    ## interpolate noise label
    annotation = np.zeros([len(abdo), ])
    ## data visuallization
    ts = range(len(abdo))
    data_visuallization_object = bio_signal_marker(ts, abdo_f, abdo, annotation,
                                                   abdo_fs, wheel_sec=100, screen_sec=300)
    data_visuallization_object.run()
    # ------------------------------------------------------------------------------------------------------------------
    ## label save
    label = data_visuallization_object.signal_label
    # check
    plt.figure(figsize=(10, 8))
    plt.subplot(311); plt.plot(abdo_f); plt.title(subject_name)
    plt.subplot(312); plt.plot(label); plt.title("Noise check")
    plt.subplot(313); plt.plot(apnea); plt.title('Apnea check')
    plt.tight_layout()
    # save
    np.save(os.path.join('D:\\artifact_detection_project\\labels', subject_name), label)

    # ------------------------------------------------------------------------------------------------------------------
