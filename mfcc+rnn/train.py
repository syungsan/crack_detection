#!/usr/bin/env python
# coding: utf-8

import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.neighbors import LocalOutlierFactor
import joblib
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam
from keras.utils import to_categorical
import os
import shutil
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.manifold import TSNE
import feature as ft
import plot_evaluation as pe
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


alpha = 16. # 30.


# L2-constrained Softmax Loss
# This is custom layer way
# If you use trainable variables, you should write this way
# ref : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda#variables
class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(alpha)

    def call(self, inputs):
        # about l2_normalize https://www.tensorflow.org/api_docs/python/tf/keras/backend/l2_normalize?hl=ja
        return K.l2_normalize(inputs, axis=1) * self.alpha


def plot_result(history):

    """
    plot result
    全ての学習が終了した後に、historyを参照して、accuracyとlossをそれぞれplotする
    """

    # accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="acc", marker=".")
    plt.plot(history.history["val_accuracy"], label="val_acc", marker=".")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.grid()
    plt.legend(loc="best")
    plt.title("Accuracy")
    plt.savefig("../logs/graphs/accuracy.png".format(ft.target))
    plt.show()

    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="loss", marker=".")
    plt.plot(history.history["val_loss"], label="val_loss", marker=".")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid()
    plt.legend(loc="best")
    plt.title("Loss")
    plt.savefig("../logs/graphs/loss.png")
    plt.show()


# データの分布の様子を次元を落として表示
def plot_tsene(X, model, output_model):

    color_codes = ["red", "blue"] # , "green", "black", "magenta", "cyan", "grey", "aqua", "springgreen", "salmon"]
    test_range = range(int(len(X)))

    output = model.predict(X)
    markers = []
    colors = []

    for i in test_range:
        markers.append(np.argmax(output[i, :]))
        colors.append(color_codes[np.argmax(output[i, :])])

    hidden_output = output_model.predict(X)
    X_reduced = TSNE(n_components=2, random_state=0, perplexity=3).fit_transform(hidden_output)

    plt.figure(figsize=(10, 10))
    for i in test_range:
        plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c=colors[i], marker="${}$".format(markers[i]), s=300)

    # plt.savefig("../graphs/t-SNE.svg", format="svg")
    plt.savefig("../logs/graphs/t-SNE.png")

    plt.show()


def get_data(data2ds):

    Xs = []
    ys = []

    for data1ds in data2ds:
        Xs.append(data1ds[1:])
        ys.append(data1ds[0])

    X = [[float(x) for x in y] for y in Xs]

    return np.array(X), ys


def main(epochs=5, batch_size=128):

    if os.path.isdir("../../../crack_detection/crack_detection/logs/models"):
        shutil.rmtree("../../../crack_detection/crack_detection/logs/models")
    os.makedirs("../../../crack_detection/crack_detection/logs/models", exist_ok=True)

    if os.path.isdir("../../../crack_detection/crack_detection/logs/graphs"):
        shutil.rmtree("../../../crack_detection/crack_detection/logs/graphs")
    os.mkdir("../../../crack_detection/crack_detection/logs/graphs")

    if os.path.exists("../../../crack_detection/crack_detection/logs/result.csv"):
        os.remove("../../../crack_detection/crack_detection/logs/result.csv")

    if os.path.isdir("../../../crack_detection/crack_detection/logs/tensor_board"):
        shutil.rmtree("../../../crack_detection/crack_detection/logs/tensor_board")

    trains = ft.read_csv(file_path="../../../crack_detection/crack_detection/logs/train.csv", delimiter=",")
    X_train, y_train = get_data(data2ds=trains)

    tests = ft.read_csv(file_path="../../../crack_detection/crack_detection/logs/test.csv", delimiter=",")
    X_test, y_test = get_data(data2ds=tests)

    # one-hot vector形式に変換する
    num_of_category = len(ft.wav_labels)
    y_train = to_categorical(y_train, num_of_category)
    y_test = to_categorical(y_test, num_of_category)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    X_train = np.reshape(X_train, (X_train.shape[0], ft.interval_division_number, ft.feature_max_length), order="F")
    X_test = np.reshape(X_test, (X_test.shape[0], ft.interval_division_number, ft.feature_max_length), order="F")

    #Initializing model
    model = keras.models.Sequential()

    #Adding the model layers
    model.add(keras.layers.LSTM(256, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(128))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    """
    #If you don't need to learn alpha , you can choose below way too.
    alpha = 30
    def l2_constrain(x):
        return alpha * K.l2_normalize(x, axis=1)
    model.add(layers.Lambda(l2_constrain))
    """

    model.add(L2ConstrainLayer())
    model.add(keras.layers.Dense(num_of_category, activation='softmax'))

    #Compiling the model
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=0.0001, amsgrad=True), # RMSprop(), # (learning_rate=1e-4),
                  metrics=["accuracy"], run_eagerly=True)

    model.summary()

    # callback function
    csv_cb = CSVLogger("../../../crack_detection/crack_detection/logs/models/train_log.csv")
    fpath = "../logs/models/model-{epoch:02d}-{loss:.2f}-{accuracy:.2f}-{val_loss:.2f}-{val_accuracy:.2f}-.h5"
    cp_cb = ModelCheckpoint(filepath=fpath, monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
    es_cb = EarlyStopping(monitor="val_loss", patience=2, verbose=1, mode="auto")
    tb_cb = TensorBoard(log_dir="../../../crack_detection/crack_detection/logs/tensor_board", histogram_freq=1)

    #Fitting data to the model
    history = model.fit(
        x=X_train, y=y_train,
        # steps_per_epoch=X_train.shape[0] // batch_size,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[csv_cb, cp_cb, tb_cb], # , es_cb],
        verbose=1)

    # result
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1]))

    recog_results = [["accuracy", "loss"], [score[1], score[0]]]
    ft.write_csv("../logs/models/recog_result.csv", recog_results)

    plot_result(history)

    model.save("../logs/models/model.h5")
    joblib.dump(scaler, "../logs/models/scaler.joblib")

    output_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    plot_tsene(X=X_test, model=model, output_model=output_model)

    y_train_normals = np.argmax(y_train, axis=1)
    X_train_normals = []

    for index, X in enumerate(X_train):
        if y_train_normals[index] == 1:
            X_train_normals.append(X)

    y_test_normals = np.argmax(y_test, axis=1)
    X_test_normals = []
    X_test_abnormals = []

    for index, X in enumerate(X_test):
        if y_test_normals[index] == 1:
            X_test_normals.append(X)
        else:
            X_test_abnormals.append(X)

    X_train_normals = np.array(X_train_normals)
    X_test_normals = np.array(X_test_normals)
    X_test_abnormals = np.array(X_test_abnormals)

    trains = output_model.predict(X_train_normals, batch_size=1)
    test_normals = output_model.predict(X_test_normals, batch_size=1)
    test_abnormals = output_model.predict(X_test_abnormals, batch_size=1)

    trains = trains.reshape((len(trains), -1))
    test_normals = test_normals.reshape((len(test_normals), -1))
    test_abnormals = test_abnormals.reshape((len(test_abnormals), -1))

    lof_scaler = MinMaxScaler()
    lof_scaler.fit_transform(trains)
    lof_scaler.transform(test_normals)
    lof_scaler.transform(test_abnormals)

    lof_model = LocalOutlierFactor(n_neighbors=5, novelty=True) # 20, novelty=True, contamination=0.001)

    if len(trains) >= 1000:
        train_length = 1000
    else:
        train_length = len(trains)

    print("\nanomaly detection model creating...\n")
    lof_model.fit(trains[:train_length])

    joblib.dump(lof_scaler, "../logs/models/lof_scaler.joblib")
    joblib.dump(lof_model, "../logs/models/lof_model.joblib", compress=True)

    Z1 = -1 * lof_model.decision_function(test_normals)
    Z2 = -1 * lof_model.decision_function(test_abnormals)

    y_preds = lof_model.predict(np.concatenate([test_normals, test_abnormals]))

    for index, y_pred in enumerate(y_preds):
        if y_pred == -1:
            y_preds[index] = 1
        else:
            y_preds[index] = 0

    y_trues = np.zeros(len(test_normals) + len(test_abnormals))
    y_trues[len(test_normals):] = 1

    cm = confusion_matrix(y_trues, y_preds)
    pe.plot_confusion_matrix(cm, "../logs/graphs/confusion_matrix.png")
    print(cm)

    results = []

    accuracy = accuracy_score(y_trues, y_preds) * 100.0
    results.append(["accuracy: {}%".format(accuracy)])

    precision = precision_score(y_trues, y_preds)
    results.append(["precision: {}".format(precision)])

    recall = recall_score(y_trues, y_preds)
    results.append(["recall: {}".format(recall)])

    f1 = f1_score(y_trues, y_preds)
    results.append(["f1_score: {}".format(f1)])

    print("\nAccuracy: {}%".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1Score: {}".format(f1))

    y_scores = np.hstack((Z1, Z2))
    roc_auc = pe.plot_roc_curve(y_trues, y_scores, "../logs/graphs/roc_curve.png")
    pr_auc = pe.plot_pr_curve(y_trues, y_scores, "../logs/graphs/pr_curve.png")

    results.append(["ROC-AUC: {}".format(roc_auc)])
    results.append(["PR-AUC: {}".format(pr_auc)])

    K.clear_session()
    ft.write_csv("../logs/result.csv", results)

    print("\nall process was completed...")


if __name__ == "__main__":

    # キーボードの入力待ち
    answer = input("モデルを構築しますか？ 古いモデルは上書きされます。(Y/n)\n")

    if answer == "Y" or answer == "y" or answer == "":
        epochs = 200
        batch_size = 128
        main(epochs, batch_size)
    else:
        exit()
