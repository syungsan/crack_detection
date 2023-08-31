#!/usr/bin/env python
# coding: utf-8

import os
import codecs
import csv
import librosa
import librosa.feature
import scipy.signal
import glob
from statistics import mean
import numpy as np
import warnings


# 1セクションの分割数
interval_division_number = 200

feature_max_length = 39

# Label
wav_labels = ["abnormal", "normal"]

target_sound = "fan"


def read_csv(file_path, delimiter):

    lists = []
    file = codecs.open(file_path, "r", "utf-8")

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


def write_csv(file_path, list):

    try:
        # 書き込み UTF-8
        with open(file_path, "w", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerows(list)

    # 起こりそうな例外をキャッチ
    except FileNotFoundError as e:
        print(e)
    except csv.Error as e:
        print(e)


def load_wave_librosa(file_path):

    # wavファイルを読み込む (ネイティブサンプリングレートを使用)
    wave, sr = librosa.load(file_path, sr=None)
    return wave, sr


# 高域強調
def preEmphasis(wave, p=0.97):

    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, wave)


def get_mfcc_librosa(file_path):

    wave, sr = load_wave_librosa(file_path)

    p = 0.97
    wave = preEmphasis(wave, p)

    n_fft = 512 # ウィンドウサイズ
    hop_length_sec = 0.010 # ずらし幅（juliusの解像度に合わせる）
    n_mfcc = 13 # mfccの次元数

    # mfccを求める
    warnings.simplefilter('ignore', FutureWarning)
    mfccs = librosa.feature.mfcc(wave, n_fft=n_fft, hop_length=librosa.time_to_samples(hop_length_sec, sr),
                                 sr=sr, n_mfcc=n_mfcc)
    warnings.resetwarnings()

    # mfccの1次元はいらないから消す
    mfccs = np.delete(mfccs, 0, axis=0)

    # 対数パワー項を末尾に追加
    S = cal_logpower_librosa(wave, sr)
    mfccs = np.vstack((mfccs, S))

    # mfcc delta* を計算する
    delta_mfccs  = librosa.feature.delta(mfccs)  # mfcc delta
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)  # mfcc delta2

    all_mfccs = mfccs.tolist() + delta_mfccs.tolist() + delta2_mfccs.tolist()

    return all_mfccs


# 対数パワースペクトルの計算
def cal_logpower_librosa(wave, sr):

    n_fft = 512 # ウィンドウサイズ
    hop_length_sec = 0.010 # ずらし幅

    warnings.simplefilter('ignore', FutureWarning)
    S = librosa.feature.melspectrogram(wave, sr=sr, n_fft=n_fft,
                                       hop_length=librosa.time_to_samples(hop_length_sec, sr))
    warnings.resetwarnings()

    S = sum(S)
    PS = librosa.power_to_db(S)

    return PS


def interval_division_average(list2ds, division_number):

    idas = []

    for list1ds in list2ds:
        if len(list1ds) < division_number:

            print("Extend list length with 0 padding...")
            additions = [0.0] * (division_number - len(list1ds))
            list1ds += additions

        size = int(len(list1ds) // division_number)
        mod = int(len(list1ds) % division_number)

        index_list = [size] * division_number
        if mod != 0:
            for i in range(mod):
                index_list[i] += 1

        averages = []
        i = 0

        for index in index_list:
            averages.append(mean(list1ds[i: i + index]))
            i += index

        idas.append(averages)

    return idas


def flatten_with_any_depth(nested_list):

    """深さ優先探索の要領で入れ子のリストをフラットにする関数"""
    # フラットなリストとフリンジを用意
    flat_list = []
    fringe = [nested_list]

    while len(fringe) > 0:
        node = fringe.pop(0)

        # ノードがリストであれば子要素をフリンジに追加
        # リストでなければそのままフラットリストに追加
        if isinstance(node, list):
            fringe = node + fringe
        else:
            flat_list.append(node)

    return flat_list


def main():

    if not os.path.exists("../logs"):
        os.mkdir("../logs")

    target_dir = "../data/{}".format(target_sound)
    wav_dirs = os.listdir(target_dir)

    max_length = 0
    for wav_dir in wav_dirs:

        for wav_label in wav_labels:
            wav_files = glob.glob("{}/{}/{}/*.wav".format(target_dir, wav_dir, wav_label))
            max_length += len(wav_files)

    process_count = 0
    normals = []
    abnormals = []

    for wav_dir in wav_dirs:
        for i, wav_label in enumerate(wav_labels):

            wav_files = glob.glob("{}/{}/{}/*.wav".format(target_dir, wav_dir, wav_label))
            for j, wav_file in enumerate(wav_files):

                mfccs = get_mfcc_librosa(wav_file)
                features = interval_division_average(mfccs, interval_division_number)
                features = flatten_with_any_depth(features)

                features.insert(0, i)

                if i == 0:
                    abnormals.append(features)
                else:
                    normals.append(features)

                process_count += 1
                print("{} processed. == {:.2f}%".format(wav_file, process_count / max_length * 100.0))

    test_abnormals = abnormals[:50]
    train_abnormals = abnormals[50:]
    test_normals = normals[:50]
    train_normals = normals[50:]

    trains = train_abnormals + train_normals
    tests = test_abnormals + test_normals

    print("\ncsv writing now... ")

    write_csv("../logs/train.csv", trains)
    write_csv("../logs/test.csv", tests)

    print("\nall process was completed...")


if __name__ == "__main__":
    main()
