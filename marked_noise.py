import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from libs.read_data import read_edf_file, get_edf_label


def plot_SignalWithNoiseMarking(subject_idx=0):
    data_path = 'E:\\public_data\\Stanford Technology Analytics and Genomics in Sleep\\stages\\psg\\edf'
    label_path = 'E:\\public_data\\Stanford Technology Analytics and Genomics in Sleep\\stages\\psg\\label'
    noise_label_path = 'D:\\artifact_detection_project\\labels'

    subject_names = list(map(lambda x: x.split('.')[0], os.listdir(data_path)))
    subject_name = subject_names[subject_idx]
    ecg_label = get_edf_label(subject_name, ('ECG', 'EKG', 'ecg', 'ekg', 'Heartrate'), data_path=data_path)
    abdo_label = get_edf_label(subject_name, ('ABD', 'Abd', 'abd'), data_path=data_path)
    data_frame = read_edf_file(subject_name, data_path=data_path, label_path=label_path,
                               sig_labels=(ecg_label, abdo_label))
    abdo = data_frame['signal']['abdo']

    noise_label_list = os.listdir(noise_label_path)
    noise_label = np.load(os.path.join(noise_label_path, noise_label_list[subject_idx]), allow_pickle=True)

    plt.figure()
    plt.subplot(211)
    plt.plot(abdo)
    plt.title('Abdo raw signal')
    plt.subplot(212)
    plt.plot(noise_label)
    plt.title('Noise marking')


def apnea_hypopnea_label_parsing(label_raw, window=60, label_fs=10, threshold=0.2):
    '''
    :param label_raw: apnea + hypopnea label raw
    :return True or False
    '''
    label_epoch = int(window * label_fs)

    label_ah = []
    for tmp_idx in range(0, len(label_raw), label_epoch):
        tmp_ah_raw = label_raw[tmp_idx:tmp_idx + label_epoch]

        if len(tmp_ah_raw) < label_epoch:
            continue

        tmp_per_ah = np.sum(tmp_ah_raw) / len(tmp_ah_raw)
        label_ah.append(tmp_per_ah)

    label_ah = np.array(label_ah)

    # binary label -> 0: normal / 1: apnea or hypopnea
    label_ah = label_ah > threshold

    return label_ah


if __name__ == '__main__':
    ## data load
    save_path = 'D:\\artifact_detection_project\\labels'
    save_labels = os.listdir(save_path)
    # options
    win = 60
    abdo_fs = 100
    epoch = int(win * abdo_fs)

    labels_all = []
    for tmp_label in tqdm(save_labels):
        tmp_load_label = np.load(os.path.join(save_path, tmp_label), allow_pickle=True)
        # 1분 에폭 단위로 라벨 생성하기
        tmp_label_ah = apnea_hypopnea_label_parsing(tmp_load_label, label_fs=abdo_fs, threshold=0.1)
        labels_all.append(tmp_label_ah)
