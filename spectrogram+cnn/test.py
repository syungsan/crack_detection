#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from keras.models import Model
import keras.backend as K
import feature as ft
import train as tr
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import predict as pr
import glob
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from keras.utils import to_categorical
import plot_evaluation as pe


is_anomaly_detect = False


class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(tr.alpha) # 16.

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1) * self.alpha


def main():

    if os.path.exists("../logs/all_result.csv.csv"):
        os.remove("../logs/all_result.csv.csv")

    tests = ft.read_csv(file_path="../logs/test.csv", delimiter=",")
    X_test, y_test = tr.get_data(data2ds=tests)
    y_trues = [int(y) for y in y_test]

    num_of_category = len(ft.wav_labels)
    y_test = to_categorical(y_test, num_of_category)

    results = [["model-name", "metrics-loss", "metrics-accuracy", "final-accuracy", "precision", "recall", "f1", "ROC-AUC", "PR-AUC"]]
    accuracies = []

    scaler, lof_model, lof_scaler = pr.load_const_models()
    scaler.transform(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], 257, ft.interval_division_number, 1), order="F")

    metrics_models = glob.glob("../logs/models/*.h5")

    for metrics_model in metrics_models:

        _model = None
        _model = pr.load_metrics_model(os.path.basename(metrics_model))

        score = _model.evaluate(X_test, y_test, verbose=0)

        print("Test loss: {}".format(score[0]))
        print("Test accuracy: {}".format(score[1]))

        result = [os.path.basename(metrics_model), score[0], score[1]]
        y_pred_prob = []

        if is_anomaly_detect:

            output_model = Model(inputs=_model.input, outputs=_model.layers[-2].output)
            y_preds, _ = pr.predict(output_model, lof_model, lof_scaler, X_test)

            for index, y_pred in enumerate(y_preds):

                if y_pred == -1:
                    y_preds[index] = 0
                else:
                    y_preds[index] = 1

        else:
            # `evaluate`メソッドは損失値と評価指標を返しますが、ここでは混同行列を取得するためには不要です
            # 予測結果を得るために`predict`メソッドを使用します
            y_pred_prob = _model.predict(X_test)

            # 予測確率からクラスを取得
            y_preds = np.argmax(y_pred_prob, axis=1)

        cm = confusion_matrix(y_trues, y_preds)
        pe.plot_confusion_matrix(cm, "../logs/graphs/test_confusion_matrix_anomaly-{}_model-{}.png"
                                 .format(is_anomaly_detect, os.path.basename(metrics_model)))
        print("\n")
        print(cm)

        accuracy = accuracy_score(y_trues, y_preds) * 100.0
        result += [accuracy]
        accuracies.append(accuracy)

        precision = precision_score(y_trues, y_preds)
        result += [precision]

        recall = recall_score(y_trues, y_preds)
        result += [recall]

        f1 = f1_score(y_trues, y_preds)
        result += [f1]

        print("\nAccuracy: {}%".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1Score: {}".format(f1))

        if is_anomaly_detect:

            X_test_normals = []
            X_test_abnormals = []

            for index, X in enumerate(X_test):
                if y_trues[index] == 1:
                    X_test_normals.append(X)
                else:
                    X_test_abnormals.append(X)

            X_test_normals = np.array(X_test_normals)
            X_test_abnormals = np.array(X_test_abnormals)

            test_normals = output_model.predict(X_test_normals, batch_size=1)
            test_abnormals = output_model.predict(X_test_abnormals, batch_size=1)

            test_normals = test_normals.reshape((len(test_normals), -1))
            test_abnormals = test_abnormals.reshape((len(test_abnormals), -1))

            Z1 = -1 * lof_model.decision_function(test_normals)
            Z2 = -1 * lof_model.decision_function(test_abnormals)
            y_scores = np.hstack((Z1, Z2))

        else:
            y_scores = y_pred_prob[:, 1]

        fpr, tpr, thresholds = roc_curve(y_trues, y_scores)
        _auc = auc(fpr, tpr)

        pe.plot_roc_curve(y_trues, y_scores, "../logs/graphs/test_roc_curve_anomaly-{}_model-{}.png"
                          .format(is_anomaly_detect, os.path.basename(metrics_model)))
        pe.plot_pr_curve(y_trues, y_scores, "../logs/graphs/test_pr_curve_anomaly-{}_model-{}.png"
                         .format(is_anomaly_detect, os.path.basename(metrics_model)))

        print("ROC-AUC {}".format(_auc))
        result += [_auc]

        precision, recall, thresholds = precision_recall_curve(y_trues, y_scores)
        _auc = auc(recall, precision)

        print("PR-AUC {}".format(_auc))
        result += [_auc]

        results.append(result)

    best_model = os.path.basename(metrics_models[np.argmax(np.array(accuracies))])
    best_accuracy = "{}%".format(max(accuracies))

    print("\nbest model == {}".format(best_model))
    print("best accuracy == {}%".format(max(accuracies)))
    results.append([])
    results.append(["best model"])
    results.append([best_model])
    results.append([best_accuracy])

    ft.write_csv("../logs/all_result.csv", results)
    print("\nall process completed...")


if __name__ == '__main__':

    # キーボードの入力待ち
    answer = input("テストを実施しますか？ 古い結果は上書きされます。 (Y/n)\n")

    if answer == "Y" or answer == "y" or answer == "":
        main()
    else:
        exit()
