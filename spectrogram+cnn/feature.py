#!/usr/bin/env python
# coding: utf-8

import os
import codecs
import csv
import librosa
import librosa.feature
import glob
from statistics import mean
import numpy as np


# 1セクションの分割数
interval_division_number = 200

# Label
wav_labels = ["abnormal", "normal"]

target = "tile"


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


# change wave data to stft
def calculate_sp(x, n_fft=512, hop_length=256):

    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    sp = librosa.amplitude_to_db(np.abs(stft))

    return sp


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

    target_dir = "../data/{}".format(target)
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

                wave, sr = load_wave_librosa(wav_file)
                features = calculate_sp(wave).tolist()

                features = interval_division_average(features, interval_division_number)
                features = flatten_with_any_depth(features)

                features.insert(0, i)

                if i == 0:
                    abnormals.append(features)
                else:
                    normals.append(features)

                process_count += 1
                print("{} processed. == {:.2f}%".format(wav_file, process_count / max_length * 100.0))

    test_abnormals = abnormals[:100]
    train_abnormals = abnormals[100:]
    test_normals = normals[:100]
    train_normals = normals[100:]

    trains = train_abnormals + train_normals
    tests = test_abnormals + test_normals

    print("\ncsv writing now... ")

    write_csv("../logs/train.csv", trains)
    write_csv("../logs/test.csv", tests)

    print("\nall process was completed...")


if __name__ == "__main__":
    main()
