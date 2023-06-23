#!/usr/bin/env python
# coding: utf-8

import os
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.models import Model
import keras.backend as K
import joblib
import feature as ft
import train as tr
import plot_evaluation as pe
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


best_rnn_model = "model.h5"


class L2ConstrainLayer(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super(L2ConstrainLayer, self).__init__(**kwargs)
    self.alpha = tf.Variable(30.) # 16.

  def call(self, inputs):
    return K.l2_normalize(inputs, axis=1) * self.alpha


def load_models():

  if os.path.exists("../logs/models/{}".format(best_rnn_model)):
    model = load_model("../logs/models/{}".format(best_rnn_model), custom_objects={"L2ConstrainLayer": L2ConstrainLayer})
    output_model = Model(inputs=model.input, outputs=model.layers[-2].output)

  if os.path.exists("../logs/models/scaler.joblib"):
    scaler = joblib.load("../logs/models/scaler.joblib")

  if os.path.exists("../logs/models/lof_scaler.joblib"):
    lof_scaler = joblib.load("../logs/models/lof_scaler.joblib")

  if os.path.exists("../logs/models/lof_model.joblib"):
    lof_model = joblib.load("../logs/models/lof_model.joblib")

  return model, output_model, scaler, lof_model, lof_scaler


def main():

  y_preds = []
  y_scores = []
  test_count = 0
  results = [["target", "recognition", "rnn_probability", "is_error",
              "lof_score", "prediction", "total_score", "correctness"]]

  model, output_model, scaler, lof_model, lof_scaler = load_models()

  features = ft.read_csv(file_path="../logs/test.csv", delimiter=",")
  Xs, ys = tr.get_data(data2ds=features)
  ys = [int(y) for y in ys]

  for index, X in enumerate(Xs):

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

    y_preds.append(y_pred)

    if y_pred == 0:
        y_rnn_prob = ((1.0 - rnn_prob) * 0.5)
    else:
        y_rnn_prob = rnn_prob

    y_lof_score = 1.0 / (abs(lof_score) + 0.1 - tr.lof_contamination)
    y_score = (y_rnn_prob + y_lof_score) * 0.5
    y_scores.append(y_score)

    if ys[index] == y_pred:
      correctness = True
    else:
      correctness = False

    result = [ft.wav_labels[ys[index]], recog_label, "{:.2f}%".format(rnn_prob * 100.0), is_error,
              "{:.2f}".format(lof_score), y_pred, y_score, correctness]

    results.append(result)
    print(result)

    test_count += 1
    print("Test count: {}\n".format(test_count))

  cm = confusion_matrix(ys, y_preds)
  pe.plot_confusion_matrix(cm, "../logs/graphs/confusion_matrix.png")
  print(cm)

  accuracy = accuracy_score(ys, y_preds) * 100.0
  results.append(["accuracy: {}%".format(accuracy)])

  precision = precision_score(ys, y_preds)
  results.append(["precision: {}".format(precision)])

  recall = recall_score(ys, y_preds)
  results.append(["recall: {}".format(recall)])

  f1 = f1_score(ys, y_preds)
  results.append(["f1_score: {}".format(f1)])

  print("\nAccuracy: {}%".format(accuracy))
  print("Precision: {}".format(precision))
  print("Recall: {}".format(recall))
  print("F1Score: {}".format(f1))

  roc_auc = pe.plot_roc_curve(ys, y_scores, "../logs/graphs/roc_curve.png")
  pr_auc = pe.plot_pr_curve(ys, y_scores, "../logs/graphs/pr_curve.png")

  results.append(["ROC-AUC: {}".format(roc_auc)])
  results.append(["PR-AUC: {}".format(pr_auc)])

  ft.write_csv("../logs/result.csv", results)
  print("\nall process completed...")


if __name__ == '__main__':

  # キーボードの入力待ち
  answer = input("テストを実施しますか？ 古い結果は上書きされます。 (Y/n)\n")

  if answer == "Y" or answer == "y" or answer == "":
    main()
  else:
    exit()
