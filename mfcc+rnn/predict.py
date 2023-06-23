#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import keras.backend as K
import feature as ft
import train as tr
import test as ts
import data_collect as dc


class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(30.) # 5.

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1) * self.alpha


def main():

    if not os.path.exists("../temp"):
        os.mkdir("../temp")

    model, output_model, scaler, lof_model, lof_scaler = ts.load_models()

    dc.set_input_device()
    dc.recording("../temp/pred.wav")

    mfccs = ft.get_mfcc_librosa("../temp/pred.wav")
    features = ft.interval_division_average(mfccs, ft.interval_division_number)

    features = ft.flatten_with_any_depth(features)
    X = [float(x) for x in features]

    scaler.transform([X])
    X = np.reshape(X, (1, ft.interval_division_number, ft.feature_max_length), order="F")

    # softmaxによる確率分布
    probabilities = model.predict(X).flatten()

    # リストの要素中最大値のインデックスを取得
    index_of_max = probabilities.argmax()

    recog_label = ft.wav_labels[index_of_max]
    rnn_prob = probabilities[index_of_max]

    lof = output_model.predict(X)
    lof = lof.reshape((len(lof), -1))
    lof_scaler.transform(lof)

    error = lof_model.predict(lof)[0]
    lof_score = lof_model.score_samples(lof)[0]

    if error == 1:
        is_error = False
        y_pred = index_of_max
    else:
        is_error = True
        y_pred = 0

    if y_pred == 0:
        y_rnn_prob = ((1.0 - rnn_prob) * 0.5)
    else:
        y_rnn_prob = rnn_prob

    y_lof_score = 1.0 / (abs(lof_score) + 0.1 - tr.lof_contamination)
    y_score = (y_rnn_prob + y_lof_score) * 0.5

    result = [recog_label, "{:.2f}%".format(rnn_prob * 100.0), is_error, "{:.2f}".format(lof_score),
              y_pred, y_score]

    print(result)
    print("\nresult is {}".format(ft.wav_labels[y_pred]))

    print("\nall process was completed...")


if __name__ == "__main__":
    main()
