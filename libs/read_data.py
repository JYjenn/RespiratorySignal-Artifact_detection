# system packages
import xml.etree.ElementTree as ET
from datetime import datetime, time as datetime_time, timedelta
import os
# 3-party packages
import pyedflib
import numpy as np
import pandas as pd
from biosppy.signals import tools as st
# custom packages
from libs.signals import smoothing
from libs.signals import make_edr, make_hr, interpolation_1d


def cfs_ecg_load_sleep_stage_slicing(subject_code, abdo_filt=True,
                                     air_flow_binding=False,
                                     ecg_filt_fre_band=[3, 45]):
    # 파일 경로에 따라 바꿀 것
    data_ecg_peak_path = 'F:\ex_data\sleep_org\cfs\polysomnography\cfs_peak'
    cfs_annotation_path = 'F:\ex_data\sleep_org\cfs\polysomnography\\annotations-events-nsrr'
    data_edf_cfs = 'F:\ex_data\sleep_org\cfs\polysomnography\edfs'
    # 고정 값
    subject_name = 'cfs-visit5-%d' % subject_code
    ## start & end info
    tmp_xml_file_name = subject_name + '-nsrr.xml'

    try:
        xml_doc = ET.parse(os.path.join(cfs_annotation_path, tmp_xml_file_name))
    except:
        return 0

    start_sec, end_sec = xml_parsing_start_info(xml_doc)
    try:
        tmp_cfs_edf_f = pyedflib.EdfReader(os.path.join(data_edf_cfs, subject_name + '.edf'))
    except:
        return 0

    ecg_1_num, ecg_2_num = np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'ECG1')[0][0], np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'ECG2')[0][0]
    spo2_ch_num = np.where(np.logical_or(np.array(tmp_cfs_edf_f.getSignalLabels() )=='SpO2', np.array(tmp_cfs_edf_f.getSignalLabels() )=='SaO2'))[0]
    position_ch_num = np.where(np.logical_or(np.array(tmp_cfs_edf_f.getSignalLabels() )=='POSITION', np.array(tmp_cfs_edf_f.getSignalLabels() )=='Position'))[0][0]
    abdo_num = np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'ABDO EFFORT')[0][0]

    if air_flow_binding:
        air_flow_num = np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'AIRFLOW')[0][0]
        air_pres_num = np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'NASAL PRES')[0][0]
    else:
        air_flow_num = 0
        air_pres_num = 0

    _, ecg_2_fs = tmp_cfs_edf_f.getSampleFrequencies()[ecg_1_num], tmp_cfs_edf_f.getSampleFrequencies()[ecg_2_num]
    abdo_fs = tmp_cfs_edf_f.getSampleFrequencies()[abdo_num]
    position_fs = tmp_cfs_edf_f.getSampleFrequencies()[position_ch_num]

    if air_flow_binding:
        air_flow_fs = tmp_cfs_edf_f.getSampleFrequencies()[air_flow_num]
        air_pres_fs = tmp_cfs_edf_f.getSampleFrequencies()[air_pres_num]
    else:
        air_flow_fs = 0
        air_pres_fs = 0

    ecg = tmp_cfs_edf_f.readSignal(ecg_2_num)[int(start_sec * ecg_2_fs):int(end_sec * ecg_2_fs)]
    ecg_1 = tmp_cfs_edf_f.readSignal(ecg_1_num)[int(start_sec * ecg_2_fs):int(end_sec * ecg_2_fs)]

    if air_flow_binding:
        air_flow = tmp_cfs_edf_f.readSignal(air_flow_num)[int(start_sec * air_flow_fs):int(end_sec * air_flow_fs)]
        air_pres = tmp_cfs_edf_f.readSignal(air_pres_num)[int(start_sec * air_pres_fs):int(end_sec * air_pres_fs)]
    else:
        air_flow = 0
        air_pres = 0

    order = int(0.3 * ecg_2_fs)
    ## ecg filtering
    ecg, _, _ = st.filter_signal(signal=ecg,
                                 ftype='FIR',
                                 band='bandpass',
                                 order=order,
                                 frequency=ecg_filt_fre_band,
                                 sampling_rate=ecg_2_fs)

    ecg_sum, _, _ = st.filter_signal(signal=ecg + ecg_1,
                                     ftype='FIR',
                                     band='bandpass',
                                     order=order,
                                     frequency=ecg_filt_fre_band,
                                     sampling_rate=ecg_2_fs)
    ## ecg peak
    peak = np.load(os.path.join(data_ecg_peak_path, str(subject_code) + '.npy'))
    ## edf close
    if abdo_filt:
        tmp_abdo = smoothing(tmp_cfs_edf_f.readSignal(abdo_num), 33)
    else:
        tmp_abdo = tmp_cfs_edf_f.readSignal(abdo_num)

    abdo = tmp_abdo[int(start_sec * abdo_fs):int(end_sec * abdo_fs)]
    spo2_raw = tmp_cfs_edf_f.readSignal(spo2_ch_num)[start_sec:end_sec]
    position = tmp_cfs_edf_f.readSignal(position_ch_num)[int(start_sec * position_fs):int(end_sec * position_fs)]
    tmp_cfs_edf_f._close()

    apnea_osa, apnea_central, hypopnea_pre, sp02_de, arousals, sleep_stage, unsure = xml_parsing(xml_doc)
    apnea = (apnea_osa + apnea_central)[int(start_sec * 10):int(end_sec * 10)]
    hypopnea = hypopnea_pre[int(start_sec * 10):int(end_sec * 10)]
    sleep_stage = sleep_stage[int(start_sec * 10):int(end_sec * 10)]
    sp02_de = sp02_de[int(start_sec * 10):int(end_sec * 10)]
    arousals = arousals[int(start_sec * 10):int(end_sec * 10)]
    unsure = unsure[int(start_sec * 10):int(end_sec * 10)]
    edr = smoothing(make_edr(ecg, peak, len(abdo)), 33)
    hr = make_hr(peak, len(abdo))

    return {'signal' :{'ecg' :ecg, 'ecg_peak': peak,
                      'ecg_sum': ecg_sum,
                      'edr' :edr, 'hr': hr, 'abdo': abdo, 'air_flow': air_flow, 'air_pres': air_pres,
                      'spo2' :spo2_raw},
            'label' :{'apnea' :apnea, 'hypopnea' :hypopnea, 'desaturation' :sp02_de, 'arousal' :arousals, 'unsure' :unsure,
                     'sleep_stage': sleep_stage},
            'position' :{'raw_data' :position,
                        'info' :'0: right 1: left 2: back 3: prone '},
            'signal_info' :{'fs_ecg' :ecg_2_fs, 'fs_edr' :abdo_fs, 'fs_abdo' :abdo_fs, 'fs_hr' :abdo_fs, 'fs_air_flow': air_flow_fs, 'fs_air_pres': air_pres_fs,
                           'fs_apnea': 10, 'fs_hypopnea': 10, 'fs_sleep_stage': 10,
                           'fs_position' :1}}


def mesa_load_data_sleep_stage_slicing(subject_code,
                                       filt_abdo=True, label_only=False):
    # 파일 경로에 따라 바꿀것
    mesa_edf_path = 'F:\\ex_data\\sleep_org\\mesa-commercial-use\\polysomnography\\edfs'
    mesa_annotation_path = 'F:\ex_data\sleep_org\mesa-commercial-use\polysomnography\\annotations-events-nsrr'
    mesa_ecg_peak_path = 'F:\ex_data\sleep_org\mesa-commercial-use\polysomnography\ekg_peak\mesa_peak'
    # 고정 값
    mesa_edf_file_name = ('mesa-sleep-%4d.edf' % subject_code).replace(' ', '0')
    mesa_annotation_name = ('mesa-sleep-%4d-nsrr.xml' % subject_code).replace(' ', '0')
    mesa_ecg_peak_name = '%d.npy' % int(subject_code)

    try:
        xml_doc = ET.parse(os.path.join(mesa_annotation_path, mesa_annotation_name))
        tmp_mesa_edf_f = pyedflib.EdfReader(os.path.join(mesa_edf_path, mesa_edf_file_name))
    except:
        return 0
    ## start info
    start_time_sec, end_time_sec, _ = xml_parsing_start_info_flag_ver(xml_doc)

    if end_time_sec == 0:
        return 0

    apnea, apnea_central, hypopnea, sp02_de, arousals, sleep_stage, unsure = xml_parsing(xml_doc)
    ## label info
    mesa_apnea = (apnea + apnea_central)[int(start_time_sec * 10):int(end_time_sec * 10)]
    mesa_hypopnea = hypopnea[int(start_time_sec * 10):int(end_time_sec * 10)]
    mesa_sleep_stage = sleep_stage[int(start_time_sec * 10):int(end_time_sec * 10)]
    mesa_unsure = unsure[int(start_time_sec * 10):int(end_time_sec * 10)]
    arousals = arousals[int(start_time_sec * 10):int(end_time_sec * 10)]
    sp02_de = sp02_de[int(start_time_sec * 10):int(end_time_sec * 10)]

    if label_only:
        mesa_spo2 = tmp_mesa_edf_f.readSignal(24)[start_time_sec:end_time_sec]
        mesa_spo2 = np.round(mesa_spo2)
        return {'signal': {'ecg': 0, 'ecg_peak': 0, 'thor': 0,
                           'edr': 0, 'hr': 0, 'abdo': 0,
                           'spo2': mesa_spo2},
                'position': {'raw_data': 0,
                             'info': '0: right 1: back 2: left 3: front 4: upright '},
                'label': {'apnea': mesa_apnea, 'hypopnea': mesa_hypopnea, 'sleep_stage': mesa_sleep_stage,
                          'unsure': mesa_unsure, 'desaturation': sp02_de, 'arousal': arousals},
                'signal_info': {'fs_ecg': 256, 'fs_edr': 32, 'fs_abdo': 32, 'fs_hr': 32, 'fs_thor': 32,
                                'fs_apnea': 10, 'fs_hypopnea': 10, 'fs_sleep_stage': 10, 'fs_unsure': 10}}

    ## signal
    mesa_ekg = -tmp_mesa_edf_f.readSignal(0)[int(start_time_sec * 256):int(end_time_sec * 256)]
    ## define filter order
    order = int(0.3 * 256)
    ## ecg filtering
    mesa_ekg, _, _ = st.filter_signal(signal=mesa_ekg,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=256)
    mesa_ecg_peak = np.load(os.path.join(mesa_ecg_peak_path, mesa_ecg_peak_name), allow_pickle=True)
    if filt_abdo:
        mesa_abdo = smoothing(tmp_mesa_edf_f.readSignal(11)[int(start_time_sec * 32):int(end_time_sec * 32)], 33)
        mesa_thor = smoothing(tmp_mesa_edf_f.readSignal(10)[int(start_time_sec * 32):int(end_time_sec * 32)], 33)
    else:
        mesa_abdo = tmp_mesa_edf_f.readSignal(11)[int(start_time_sec * 32):int(end_time_sec * 32)]
        mesa_thor = tmp_mesa_edf_f.readSignal(10)[int(start_time_sec * 32):int(end_time_sec * 32)]
    mesa_spo2 = tmp_mesa_edf_f.readSignal(24)[start_time_sec:end_time_sec]
    mesa_spo2 = np.round(mesa_spo2)
    mesa_edr = smoothing(make_edr(mesa_ekg, mesa_ecg_peak, len(mesa_abdo)), 33)
    mesa_hr = make_hr(mesa_ecg_peak, len(mesa_abdo))
    mesa_position = tmp_mesa_edf_f.readSignal(14)[int(start_time_sec * 32):int(end_time_sec * 32)]
    tmp_mesa_edf_f._close()

    return {'signal': {'ecg': mesa_ekg, 'ecg_peak': mesa_ecg_peak,
                       'edr': mesa_edr, 'hr': mesa_hr, 'abdo': mesa_abdo, 'thor': mesa_thor,
                       'spo2': mesa_spo2},
            'position': {'raw_data': mesa_position,
                         'info': '0: right 1: back 2: left 3: front 4: upright '},
            'label': {'apnea': mesa_apnea, 'hypopnea': mesa_hypopnea, 'sleep_stage': mesa_sleep_stage,
                      'unsure': mesa_unsure, 'desaturation': sp02_de, 'arousal': arousals},
            'signal_info': {'fs_ecg': 256, 'fs_edr': 32, 'fs_abdo': 32, 'fs_hr': 32, 'fs_thor': 32,
                            'fs_apnea': 10, 'fs_hypopnea': 10, 'fs_sleep_stage': 10, 'fs_unsure': 10}}


def load_kd_data(subject_name, abdo2_smoothing=False):
    ## data path (파일 경로에 따라 바꿀 것)
    KD_DATA_PATH = 'F:\ex_data\sleep_org\kd\\rawdata'
    KD_PEAK_PATH = 'F:\\ex_data\\sleep_org\\kd\\peak'
    KD_PEAK_PATH_ref = 'F:\\ex_data\\sleep_org\\kd\\peak_PSG_ECG_reverse'
    KD_PEAK_PATH_ref_nor_reverse = 'F:\\ex_data\\sleep_org\\kd\\peak_PSG_ECG'
    ## load_data_frame
    kd_data_frame = np.load(os.path.join(KD_DATA_PATH, subject_name+'.npy'), allow_pickle=True).item()
    ## signal binding
    # ecg
    ecg = -kd_data_frame['ECG fs256']
    order = int(0.3 * 256)
    ecg, _, _ = st.filter_signal(signal=ecg,
                                 ftype='FIR',
                                 band='bandpass',
                                 order=order,
                                 frequency=[3, 45],
                                 sampling_rate=256)
    peak = np.load(os.path.join(KD_PEAK_PATH, subject_name + '.npy'))
    # ecg ref (PSG data)
    ecg_ref = kd_data_frame['ECGref fs200']
    order_ref = int(0.3 * 200)
    ecg_ref, _, _ = st.filter_signal(signal=ecg_ref,
                                     ftype='FIR',
                                     band='bandpass',
                                     order=order_ref,
                                     frequency=[3, 45],
                                     sampling_rate=200)
    # ecg peak
    peak_ref = np.load(os.path.join(KD_PEAK_PATH_ref, subject_name + '.npy'))
    peak_ref_reverse = np.load(os.path.join(KD_PEAK_PATH_ref_nor_reverse, subject_name + '.npy'))
    # abdo 1
    abdo1 = smoothing(kd_data_frame['ACC_X fs32'], 33)
    # abdo 2
    if abdo2_smoothing:
        abdo2 = smoothing(kd_data_frame['abdo fs200'], 201)
    else:
        abdo2 = kd_data_frame['abdo fs200']
    # acc signal
    acc_x = kd_data_frame['ACC_X fs32']
    acc_y = kd_data_frame['ACC_Y fs32']
    acc_z = kd_data_frame['ACC_Z fs32']
    # hr
    hr = make_hr(peak, len(acc_x))
    hr_ref = make_hr(peak_ref, len(acc_x))
    # edr
    edr = smoothing(make_edr(ecg, peak, len(acc_x)))
    edr_ref = smoothing(make_edr(ecg_ref, peak_ref, len(acc_x)))
    # spo2
    spo2 = kd_data_frame['spo2 fs200']
    spo2 = interpolation_1d(spo2, int(len(spo2)/200))
    ## label binding
    apnea = kd_data_frame['apnea fs10']
    hypopnea = kd_data_frame['hypopnea fs10']
    flow_limitation = kd_data_frame['flowlimitation fs10']
    arousal = kd_data_frame['wake fs1']
    sleep_stage = kd_data_frame['sstage fs1']

    return {'signal':{'ecg': ecg, 'ecg_ref': ecg_ref, 'ecg_peak': peak, 'ecg_peak_ref': peak_ref, 'ecg_peak_ref_reverse': peak_ref_reverse,
                      'abdo_ref': abdo2,
                      'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
                      'hr': hr, 'hr_ref': hr_ref,
                      'edr': edr, 'edr_ref': edr_ref,
                      'spo2': spo2},
            'label':{'apnea': apnea, 'hypopnea': hypopnea, 'flow_limitation': flow_limitation,
                     'arousal': arousal, 'sleep_stage': sleep_stage},
            'signal_info':{'fs_ecg': 256,
                           'fs_abdo': 32, 'fs_abdo2': 32,
                           'fs_hr': 32, 'fs_edr': 32,
                           'fs_spo2': 1,
                           'fs_apnea': 10, 'fs_hypopnea': 10, 'fs_flowlimitation': 10, 'fs_arousal': 1,
                           'fs_sleep_stage': 1}}


def xml_parsing_start_info(xml_doc):
    root = xml_doc.getroot()
    event_type, event_concept, start, duration, spo2_base, spo2_nar = [], [], [], [], [], []

    for tmp_info_xml in root.iter('ScoredEvent'):
        tmp_event_type = tmp_info_xml.findtext('EventType')
        if 'Stages' in tmp_event_type:
            event_type.append(tmp_info_xml.findtext('EventType'))
            event_concept.append(tmp_info_xml.findtext('EventConcept'))
            start.append(tmp_info_xml.findtext('Start'))
            duration.append(tmp_info_xml.findtext('Duration'))
            spo2_base.append(tmp_info_xml.findtext('SpO2Baseline'))
            spo2_nar.append(tmp_info_xml.findtext('SpO2Nadir'))
        else:
            pass

    tmp_first_stage_start_time = np.float32(start[0])
    tmp_first_stage_duration = np.float32(duration[0])
    tmp_first_concep = event_concept[0]

    if np.logical_or('Wake' in tmp_first_concep, 'Unscored' in tmp_first_concep):
        tmp_start_sec = int(tmp_first_stage_duration - tmp_first_stage_start_time)
    else:
        tmp_start_sec = int(tmp_first_stage_start_time)

    tmp_end_start_time = np.float32(start[-1])
    tmp_end_duration = np.float32(duration[-1])
    tmp_end_concep = event_concept[-1]
    tmp_end_concep_pre = event_concept[-2]
    tmp_end_start_time_pre = np.float32(start[-2])
    if np.logical_or('Wake' in tmp_end_concep, 'Unscored' in tmp_end_concep):
        tmp_end_sec = int(tmp_end_start_time)
    else:
        tmp_end_sec = int(tmp_end_start_time + tmp_end_duration)

    if np.logical_and('Wake' in tmp_end_concep_pre, 'Unscored' in tmp_end_concep):
        tmp_end_sec = int(tmp_end_start_time_pre)
    else:
        pass

    return tmp_start_sec, tmp_end_sec


def xml_parsing(xml_doc, spo2_desa_index=3):
    root = xml_doc.getroot()
    event_type, event_concept, start, duration, spo2_base, spo2_nar = [], [], [], [], [], []

    for tmp_info_xml in root.iter('ScoredEvent'):
        event_type.append(tmp_info_xml.findtext('EventType'))
        event_concept.append(tmp_info_xml.findtext('EventConcept'))
        start.append(tmp_info_xml.findtext('Start'))
        duration.append(tmp_info_xml.findtext('Duration'))
        spo2_base.append(tmp_info_xml.findtext('SpO2Baseline'))
        spo2_nar.append(tmp_info_xml.findtext('SpO2Nadir'))

    data_len_sec = int(np.float32(duration[0]))
    db = pd.DataFrame({'evnet_type': event_type[1::],
                       'event_concept': event_concept[1::],
                       'start_sec': start[1::],
                       'duration_sec': duration[1::],
                       'spo2_base': spo2_base[1::],
                       'spo2_nar': spo2_nar[1::], })

    apnea, sp02_de, apnea_central, hypopnea, arousals, sleep_stage = np.zeros([data_len_sec * 10]), np.zeros(
        [data_len_sec * 10]), np.zeros([data_len_sec * 10]), \
                                                                     np.zeros([data_len_sec * 10]), np.zeros(
        [data_len_sec * 10]), np.ones([data_len_sec * 10]) * 10
    unsure = np.zeros([data_len_sec * 10])

    for tmp_db in db.iterrows():
        tmp_db = tmp_db[1]
        tmp_event_concept = tmp_db['event_concept']
        tmp_event_type = tmp_db['evnet_type']
        tmp_start_sec = tmp_db['start_sec']
        tmp_duration = tmp_db['duration_sec']
        tmp_spo2_base = tmp_db['spo2_base']
        tmp_spo2_nar = tmp_db['spo2_nar']

        start_cursor = int(np.float32(tmp_start_sec) * 10)
        end_cursor = start_cursor + int(np.float32(tmp_duration) * 10)

        if 'Hypopnea' in tmp_event_concept:
            hypopnea[start_cursor:end_cursor] = 1
        elif 'SpO2 desaturation' in tmp_event_concept:
            if (np.float32(tmp_spo2_base) - np.float32(tmp_spo2_nar)) >= spo2_desa_index:
                sp02_de[start_cursor:end_cursor] = 1
        elif 'Central apnea' in tmp_event_concept:
            apnea_central[start_cursor:end_cursor] = 1
        elif 'Obstructive apnea' in tmp_event_concept:
            apnea[start_cursor:end_cursor] = 1
        elif tmp_event_concept == 'REM sleep|5':
            sleep_stage[start_cursor:end_cursor] = 4
        elif tmp_event_concept == 'Stage 1 sleep|1':
            sleep_stage[start_cursor:end_cursor] = 1
        elif tmp_event_concept == 'Stage 2 sleep|2':
            sleep_stage[start_cursor:end_cursor] = 2
        elif tmp_event_concept == 'Stage 3 sleep|3':
            sleep_stage[start_cursor:end_cursor] = 3
        elif tmp_event_concept == 'Wake|0':
            sleep_stage[start_cursor:end_cursor] = 0
        elif 'Arousal' in tmp_event_concept:
            arousals[start_cursor:end_cursor] = 1
        elif np.logical_and('Respiratory' in tmp_event_type,
                            'Unsure' in tmp_event_concept):
            unsure[start_cursor:end_cursor] = 1

    return apnea, apnea_central, hypopnea, sp02_de, arousals, sleep_stage, unsure


def xml_parsing_start_info_flag_ver(xml_doc):
    root = xml_doc.getroot()
    event_type, event_concept, start, duration, spo2_base, spo2_nar = [], [], [], [], [], []

    for tmp_info_xml in root.iter('ScoredEvent'):
        tmp_event_type = tmp_info_xml.findtext('EventType')
        if 'Stages' in tmp_event_type:
            event_type.append(tmp_info_xml.findtext('EventType'))
            event_concept.append(tmp_info_xml.findtext('EventConcept'))
            start.append(tmp_info_xml.findtext('Start'))
            duration.append(tmp_info_xml.findtext('Duration'))
            spo2_base.append(tmp_info_xml.findtext('SpO2Baseline'))
            spo2_nar.append(tmp_info_xml.findtext('SpO2Nadir'))
        else:
            pass

    tmp_first_stage_start_time = np.float32(start[0])
    tmp_first_stage_duration = np.float32(duration[0])
    tmp_first_concep = event_concept[0]

    if np.logical_or('Wake' in tmp_first_concep, 'Unscored' in tmp_first_concep):
        tmp_start_sec = int(tmp_first_stage_duration - tmp_first_stage_start_time)
    else:
        tmp_start_sec = int(tmp_first_stage_start_time)

    tmp_end_start_time = np.float32(start[-1])
    tmp_end_duration = np.float32(duration[-1])
    tmp_end_concep = event_concept[-1]
    tmp_end_concep_pre = event_concept[-2]
    tmp_end_start_time_pre = np.float32(start[-2])
    if np.logical_or('Wake' in tmp_end_concep, 'Unscored' in tmp_end_concep):
        tmp_end_sec = int(tmp_end_start_time)
        flag = 0
    else:
        tmp_end_sec = int(tmp_end_start_time + tmp_end_duration)
        flag = 0

    if np.logical_and('Wake' in tmp_end_concep_pre, 'Unscored' in tmp_end_concep):
        tmp_end_sec = int(tmp_end_start_time_pre)
        flag = 1
    else:
        pass

    return tmp_start_sec, tmp_end_sec, flag

  
def time_slice_with_sleep_stage(subject_name, data_path='./label', tst=True):
    """:cvar
    subject의 파일 이름을 인자로 받아 해당 subject에 해당하는 annotation 값을 읽어 wake가 제외한
    sleep stage 가 처음 시작 되는 부분을 start_time, wake를 제외한 마지막 sleep stage를 end_time 으로 정의
    annotation 파일은 csv 형식으로 되어있음 (예: BOGN00005.csv)


    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    w | w | w | N1 | N2 | N2 | w | w | w | w | w | N2 | w | w | N2 | N1 | N1 | N1 | N1 | w | w |
               <----------------------------------------------------------------------->
               O                                                                       O
       (지점 start_time_sec)                                                  (지점 end_time_sec)
    """
    subject_df = pd.read_csv(os.path.join(data_path, subject_name + '.csv'), usecols=[0,1,2])
    start_time_sec, end_time_sec = subject_df['Start Time'].iloc[0], subject_df['Start Time'].iloc[-1]

    if tst:
        sleep_stage = ['stage1', 'stage2', 'stage3', 'rem']
        # forward
        for tmp_idx, tmp_row in subject_df.iterrows():
            tmp_event = tmp_row['Event']
            tmp_time = tmp_row['Start Time']
            # event 변수 공백 & 소문자 변환
            tmp_event = tmp_event.strip().lower()
            if tmp_event in sleep_stage:
                start_time_sec = tmp_time
                break

        # backward
        for tmp_i, tmp_r in subject_df[::-1].iterrows():
            reverse_tmp_event = tmp_r['Event']
            reverse_tmp_time = tmp_r['Start Time']
            # event 변수 공백 & 소문자 변환
            reverse_tmp_event = reverse_tmp_event.strip().lower()
            if reverse_tmp_event in sleep_stage:
                end_time_sec = reverse_tmp_time
                break
    else:
        pass

    return start_time_sec, end_time_sec


def read_label(subject_name, fs=10, data_path='./label'):
    """":cvar
    subject 파일 이름을 인자로 받아 sleep stage 와 apnea hypopnea 를 인자로 주어진 sampling rate 로 추출
    """
    data_frame = pd.read_csv(os.path.join(data_path, subject_name + '.csv'), usecols=[0, 1, 2])
    start, end = data_frame['Start Time'].iloc[0], data_frame['Start Time'].iloc[-1]
    duration_sec = time_diff(start, end).seconds
    # label 정의
    apnea = np.zeros([int(duration_sec * fs), ])
    hypopnea = np.zeros([int(duration_sec * fs), ])
    sleep_stage = np.zeros([int(duration_sec * fs), ])
    desaturation = np.zeros([int(duration_sec * fs), ])
    arousal = np.zeros([int(duration_sec * fs), ])
    # Events
    stage_name = ['wake', 'stage1', 'stage2', 'stage3', 'rem']
    event_name = ['centralapnea', 'obstructiveapnea', 'hypopnea', 'desaturation', 'arousal']

    # data iteration
    for tmp_db in data_frame.iterrows():
        # pasing iterated data
        tmp_time, tmp_duration, tmp_event = tmp_db[1]
        # event 변수 공백 제거 & 소문자 변환
        tmp_event = tmp_event.strip().lower()
        # 시작 지점 계산
        tmp_event_start_sec = time_diff(start, tmp_time).seconds
        tmp_event_start_idx = int(tmp_event_start_sec * fs)
        # 관심 event or stage에 해당할 경우 label 추출
        if tmp_event in event_name:
            tmp_event_duration = int(tmp_duration * fs)
            tmp_event_end_idx = tmp_event_start_idx + tmp_event_duration
            # apnea
            if tmp_event in ['centralapnea', 'obstructiveapnea']:
                apnea[tmp_event_start_idx:tmp_event_end_idx] = 1
            # hypopnea
            elif tmp_event == 'hypopnea':
                hypopnea[tmp_event_start_idx:tmp_event_end_idx] = 1
            # desaturation
            elif tmp_event == 'desaturation':
                desaturation[tmp_event_start_idx:tmp_event_end_idx] = 1
            # arousal
            elif tmp_event == 'arousal':
                arousal[tmp_event_start_idx:tmp_event_end_idx] = 1
        # stage: [wake: 0, n1&n2: 1, n3: 2, rem: 3]
        elif tmp_event in stage_name:
            # duration 0인 경우는 30초 고정 & 0 아닌 경우 그대로 사용
            if int(tmp_duration) == 0:
                tmp_event_duration = int(30 * fs)
            else:
                tmp_event_duration = int(tmp_duration * fs)
            tmp_event_end_idx = tmp_event_start_idx + tmp_event_duration
            # wake
            if tmp_event == 'wake':
                sleep_stage[tmp_event_start_idx:tmp_event_end_idx] = 0
            # n1&n2
            elif tmp_event == 'stage1' or tmp_event == 'stage2':
                sleep_stage[tmp_event_start_idx:tmp_event_end_idx] = 1
            # n3
            elif tmp_event == 'stage3':
                sleep_stage[tmp_event_start_idx:tmp_event_end_idx] = 2
            # rem
            elif tmp_event == 'rem':
                sleep_stage[tmp_event_start_idx:tmp_event_end_idx] = 3
        else:
            pass
    # 리턴 값 반환
    return apnea, hypopnea, sleep_stage, desaturation, arousal


def read_edf_file(subject_name,
                  data_path='./edf', label_path='./label', label_fs=10,
                  tst=True,
                  sig_labels=('EKG', 'ABDM'),
                  sig_labels_save=('ecg', 'abdo')):
    """:cvar
    subject의 파일 이름을 인자로 받아 해당 subject에 해당 되는 신호를 EDF 파일로부터 읽어, time_slice_with_sleep_stage() 함수를 사용하여 앞뒤로 잘라 반환
    모든 신호의 초단위 길이는 같아야한다.

    return value는 딕셔너리,
    signal 은 일단 ecg, abdominal 만 만들고 추후 dynamic하게 인자를 받아 들여 해당 인자의 신호만 읽도록 변경
    label은 apnea, hypopnea, sleep stage 3개 반환

    ## test code
    from matplotlib import pyplot as plt
    from scipy import signal
    data_frame = read_edf_file('BOGN00003')
    sig = data_frame['signal']['ABDM']
    annotation = data_frame['label']['apnea'] + data_frame['label']['hypopnea']
    print('apnea sec: {} sig sec: {}'.format(len(annotation) / 10, sig.__len__() / 50))
    plt.plot(sig)
    plt.plot(signal.resample(annotation, len(sig)) * 2000)

    subject_names = os.listdir('G:\\data\\public_data\\Stanford Technology Analytics and Genomics in Sleep\\label')
    subject_names = list(map(lambda x: x.split('.')[0], subject_names))
    for tmp_subject in subject_names:
        tmp_data_frame = read_edf_file(tmp_subject,
                                       data_path='G:\\data\\public_data\\Stanford Technology Analytics and Genomics in Sleep\\edf',
                                       label_path='G:\\data\\public_data\\Stanford Technology Analytics and Genomics in Sleep\\label')
    """
    # error handling
    if len(sig_labels) != len(sig_labels_save):
        raise ValueError('sig_labels length and sig_labels_save length must be same')
    # parameter 정의
    sigs = []
    sampling_rates = []
    is_label_slice = False
    # edf 파일 읽기
    try:
        tmp_mesa_edf_f = pyedflib.EdfReader(os.path.join(data_path, subject_name + '.edf'))
    except:
        raise ValueError('can not read edf file ! ! ')
    # 시작, 끝 시간 반환
    start_time_label, end_time_label = time_slice_with_sleep_stage(subject_name, data_path=label_path, tst=False)
    start_time_sec_edf = '{}:{}:{}'.format(tmp_mesa_edf_f.starttime_hour, tmp_mesa_edf_f.starttime_minute,
                                           tmp_mesa_edf_f.starttime_second)
    # EDF 와 label 시작 시간 싱크 맞추기 (3가지 경우 EDF의 시작시간이 label보다 짧을 경우, 긴경우, 같을 경우)
    start_time_label_second = time_diff('00:00:00', start_time_label).seconds
    start_time_edf_second = time_diff('00:00:00', start_time_sec_edf).seconds

    if start_time_label_second < 43200:
        start_time_label_second += 86400

    if start_time_edf_second < 43200:
        start_time_edf_second += 86400

    if start_time_label_second < start_time_edf_second:
        # label이 edf 보다 더 빨리 측정된 경우 (라벨을 잘라야함) 추후 테스트
        # print('label is longer than PSG ! ! !') # 테스트를 위한 print 문
        edf_time_sec_start = 0
        label_time_sec_start = start_time_edf_second - start_time_label_second
        # 잘라준 만큼 start_time_label를 뒤로 shift
        start_time_label_shift = datetime.strptime('1999-01-02 ' + start_time_label, '%Y-%m-%d %H:%M:%S') + timedelta(seconds=label_time_sec_start)
        start_time_label = '{}:{}:{}'.format(start_time_label_shift.hour, start_time_label_shift.minute,
                                             start_time_label_shift.second)
        is_label_slice = True
    elif start_time_label_second > start_time_edf_second:
        # edf 가 label 보다 더 빨리 측정된 경우 (신호를 잘라야함)
        edf_time_sec_start = start_time_label_second - start_time_edf_second
        label_time_sec_start = 0
    else:
        # edf 시작, label 시작 같을 경우
        edf_time_sec_start = 0
        label_time_sec_start = 0

    start_time_tst, end_time_tst = time_slice_with_sleep_stage(subject_name, data_path=label_path, tst=tst)
    # tst 기준 잘라야할 데이터 길이와 duration 반환
    time_diff_tst = time_diff(start_time_label, start_time_tst).seconds
    time_duration_tst = time_diff(start_time_tst, end_time_tst).seconds
    # tst 기간이 1시간 이하 일 때 0 반환
    if time_duration_tst < 3600:
        return 0

    if is_label_slice:
        # label 이 잘렸을 경우, tst 의 시작 시간이 label 의 잘린 시작 시간 보다 앞에 있는지 확인
        label_start_time_second = time_diff('00:00:00', start_time_label).seconds
        tst_start_time_second = time_diff('00:00:00', start_time_tst).seconds

        if label_start_time_second < 43200:
            label_start_time_second += 86400

        if tst_start_time_second < 43200:
            tst_start_time_second += 86400

        # tst 가 위 부분 label 이 psg 보다 더 긴경우 잘렸을 때 잘린 시간안에 포함 될 때
        if label_start_time_second > tst_start_time_second:
            time_diff_tst = 0
            print('label was sliced and tst faster than label')
    # 끝 시간 을 label, PSG중 짧은 시간 기준으로 계산
    # EDF 에서 읽은 끝 시간과 label 파일에서 읽은 끝시간을 초로 변환
    time_duration_label = time_diff(start_time_label, end_time_label).seconds - label_time_sec_start
    time_duration_edf = tmp_mesa_edf_f.file_duration - edf_time_sec_start
    # edf, label 두 데이터 중 짧게 측정된 신호로 끝 부분 slicing
    if time_duration_edf >= time_duration_label:
        edf_time_sec_end = time_duration_label
    else:
        edf_time_sec_end = time_duration_edf
    # 만약, edf 가 너무 적게 측정되어 tst의 range가 edf 신호보다 작을 때 tst duration 을  edf 의 duration 으로 조정해줘야함
    if edf_time_sec_end < time_duration_tst:
        time_duration_tst = edf_time_sec_end - time_diff_tst
    # signal 읽기
    # edf label 번호 및 sampling rate 읽기
    edf_labels = np.array(tmp_mesa_edf_f.getSignalLabels())
    edf_sampling_rates = np.array(tmp_mesa_edf_f.getSampleFrequencies())
    # signal label 로 순회
    for tmp_sig_label in sig_labels:
        tmp_target_index = np.where(edf_labels == tmp_sig_label)[0][0]
        tmp_sig_sampling_rate = edf_sampling_rates[tmp_target_index]
        tmp_sig = tmp_mesa_edf_f.readSignal(tmp_target_index)
        # label 이나 edf 길이로 slicing
        tmp_sig = tmp_sig[int(tmp_sig_sampling_rate * edf_time_sec_start):int(tmp_sig_sampling_rate * edf_time_sec_start) + int(tmp_sig_sampling_rate * edf_time_sec_end)]

        if tst:
            # 받은 신호를 tst에 맞게 slicing (tib - tst 시작시간 ~ tst 기준 duration)
            tmp_sig_start_point_tst = int(tmp_sig_sampling_rate * time_diff_tst)
            tmp_sig_end_point_tst = int(tmp_sig_sampling_rate * time_diff_tst) + int(tmp_sig_sampling_rate * time_duration_tst)
            tmp_sig = tmp_sig[tmp_sig_start_point_tst:tmp_sig_end_point_tst]

        sigs.append(tmp_sig)
        sampling_rates.append(tmp_sig_sampling_rate)
    # tmp_mesa_edf_f 닫기
    tmp_mesa_edf_f._close()
    # label 처리
    apnea, hypopnea, sleep_stage, desaturation, arousal = read_label(subject_name, fs=label_fs, data_path=label_path)
    # label 이나 edf 길이로 slicing
    apnea = apnea[int(label_fs * label_time_sec_start):int(label_fs * label_time_sec_start) + int(label_fs * edf_time_sec_end)]
    hypopnea = hypopnea[int(label_fs * label_time_sec_start):int(label_fs * label_time_sec_start) + int(label_fs * edf_time_sec_end)]
    sleep_stage = sleep_stage[int(label_fs * label_time_sec_start):int(label_fs * label_time_sec_start) + int(label_fs * edf_time_sec_end)]
    desaturation = desaturation[int(label_fs * label_time_sec_start):int(label_fs * label_time_sec_start) + int(label_fs * edf_time_sec_end)]
    arousal = arousal[int(label_fs * label_time_sec_start):int(label_fs * label_time_sec_start) + int(label_fs * edf_time_sec_end)]

    if tst:
        # 받은 신호를 tst에 맞게 slicing (tib - tst 시작시간 ~ tst 기준 duration)
        label_start_point_tst = int(label_fs * time_diff_tst)
        label_end_point_tst = int(label_fs * time_diff_tst) + int(label_fs * time_duration_tst)
        apnea = apnea[label_start_point_tst:label_end_point_tst]
        hypopnea = hypopnea[label_start_point_tst:label_end_point_tst]
        sleep_stage = sleep_stage[label_start_point_tst:label_end_point_tst]
        desaturation = desaturation[label_start_point_tst:label_end_point_tst]
        arousal = arousal[label_start_point_tst:label_end_point_tst]

    # label과 signal의 길이가 맞치안으면 에러 출력
    if len(apnea) / label_fs != len(sigs[0]) / sampling_rates[0]:
        raise ValueError('label length and signal length dismatch')

    return {'signal':{tmp_key: sigs[idx] for idx, tmp_key in enumerate(sig_labels_save)},
            'label': {'apnea': apnea,
                      'hypopnea': hypopnea,
                      'sleep_stage': sleep_stage,
                      'desaturation': desaturation,
                      'arousal': arousal},
            'info': {'sampling_rate_sig': {tmp_key: sampling_rates[idx] for idx, tmp_key in enumerate(sig_labels_save)}}}


def time_diff(start, end):
    start = datetime.strptime(start, '%H:%M:%S')
    end = datetime.strptime(end, '%H:%M:%S')
    if isinstance(start, datetime_time): # convert to datetime
        assert isinstance(end, datetime_time)
        start, end = [datetime.combine(datetime.min, t) for t in [start, end]]
    if start <= end: # e.g., 10:33:26-11:15:49
        return end - start
    else: # end < start e.g., 23:55:00-00:25:00
        end += timedelta(1) # +day
        assert end > start
        return end - start


def get_edf_label(subject_name, orders_label, data_path='./edf'):
    """:cvar
    orders_label 입력 값
    심전도: ('ECG', 'EKG', 'ecg', 'ekg', 'Heartrate')
    abdominal: ('ABD', 'Abd', 'abd',)
    """
    # ekg 의 edf 라벨을 구함
    try:
        edf_f = pyedflib.EdfReader(os.path.join(data_path, subject_name + '.edf'))
    except:
        raise ValueError('can not read edf file ! ! ')
    # edf label 추출
    labels = edf_f.getSignalLabels()
    # 주어진 orders_label 을 사용하여 edf label 을 필터링 후 정렬
    labels_target = list(filter(lambda x: any([tmp_data in x for tmp_data in orders_label]), labels))
    labels_target.sort()
    # 찾은 라벨이 1개도 없다면 반칸 반환
    if len(labels_target) == 0:
        print('can not find corresponding label ({})'.format(orders_label))
        return ''
    # edf f 닫기
    edf_f._close()
    # 첫번째 label 이름 (1번 채널) 반환
    return labels_target[0]
