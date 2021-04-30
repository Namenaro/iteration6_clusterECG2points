import json
import easygui
import os
import numpy as np


def select_and_load_json(file_path=None):
    if file_path is None:
        file_path = easygui.fileopenbox("Select json dataset")
    with open(file_path, 'r') as f:
        return json.load(f)


def get_healthy_7():
    folder = '/home/yana/ECG_DATA'
    filename = '7_pacients_ideally_healthy_and_normal_axis.json'
    path = os.path.join(folder, filename)
    return select_and_load_json(path)

def get_triplets(patient, component, lead_name='i'):
    return patient['Leads'][lead_name]['DelineationDoc'][component]

def get_lead_signal(ecg, lead_name):
    return ecg['Leads'][lead_name]['Signal']

def cut_patch(signal, coord, patch_len):
    start = coord - int(patch_len/2)
    end = start + patch_len
    if start >=0 and end < len(signal):
        return signal[start:end]
    return None

def get2datasets(): # QRS i
    json_data = get_healthy_7()
    patch_len1 = 13
    patch_len2 = 13
    X1 = []
    X2 = []
    dists = 0
    delta = 0
    for patient_id in json_data.keys():
        ecg_json = json_data[patient_id]
        i_qrs_triplets = get_triplets(ecg_json, "qrs", "i")
        i_signal = get_lead_signal(ecg_json, "i")
        for triplet in i_qrs_triplets:
            p1 = triplet[1]+delta
            p2 = triplet[1]+int( patch_len1/2)+ 1 + int(patch_len2/2) +delta
            print(p1 - p2)

            x1 = cut_patch(i_signal, p1, patch_len1)
            x2 = cut_patch(i_signal, p2, patch_len2)
            if x1 is not None and x2 is not None:
                X1.append(x1)
                X2.append(x2)
                dists = abs(p1-p2) + dists
    dist = dists/len(X1)
    print ("average distance btw p1 and p2 = "+ str(dist))
    return np.array(X1), np.array(X2)