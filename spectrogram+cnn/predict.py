#!/usr/bin/env python
# coding: utf-8

import os
from keras.models import load_model
import numpy as np
import joblib
import feature as ft
import recording as rd
import train as tr
import tensorflow as tf
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam


best_metrics_model = "model.h5"


class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(tr.alpha) # 16.

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1) * self.alpha


def load_metrics_model(metrics_model):

    if os.path.exists("../logs/models/{}".format(metrics_model)):
        model = load_model("../logs/models/{}".format(metrics_model),
                           custom_objects={"L2ConstrainLayer": L2ConstrainLayer})

        model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model


def load_const_models():

    if os.path.exists("../logs/models/scaler.joblib"):
        scaler = joblib.load("../logs/models/scaler.joblib")

    if os.path.exists("../logs/models/lof_scaler.joblib"):
        lof_scaler = joblib.load("../logs/models/lof_scaler.joblib")

    if os.path.exists("../logs/models/lof_model.joblib"):
        lof_model = joblib.load("../logs/models/lof_model.joblib")

    return scaler, lof_model, lof_scaler


def predict(output_model, lof_model, lof_scaler, X):

    metrics_probs = output_model.predict(X, batch_size=1)
    metrics_probs = metrics_probs.reshape((len(metrics_probs), -1))
    lof_scaler.transform(metrics_probs)

    scores = -1 * lof_model.decision_function(metrics_probs)
    y_preds = lof_model.predict(metrics_probs)

    return y_preds, scores


def main():

    if not os.path.exists("../temp"):
        os.mkdir("../temp")

    model = load_metrics_model(best_metrics_model)
    scaler, lof_model, lof_scaler = load_const_models()
    output_model = Model(inputs=model.input, outputs=model.layers[-3].output)

    rd.set_input_device()
    rd.recording("../temp/pred.wav")

    wave, sr = ft.load_wave_librosa("../temp/pred.wav")
    features = ft.calculate_sp(wave).tolist()

    features = ft.interval_division_average(features, ft.interval_division_number)
    features = ft.flatten_with_any_depth(features)
    X = [float(x) for x in features]

    scaler.transform([X])
    X = np.reshape(X, (1, 257, ft.interval_division_number, 1), order="F")

    y_preds, scores = predict(output_model, lof_model, lof_scaler, X)

    if y_preds[0] == -1:
        correctness = ft.wav_labels[0]
    else:
        correctness = ft.wav_labels[1]

    print("\nresult is {}".format(correctness))
    print("score: {}".format(scores[0]))

    print("\nall process was completed...")


if __name__ == "__main__":
    main()
