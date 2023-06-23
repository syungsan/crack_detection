#!/usr/bin/env python
# coding: utf-8

import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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


lof_contamination = 0.07


# L2-constrained Softmax Loss
#This is custom layer way
# If you use trainable variables, you should write this way
# ref : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda#variables
class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(30.) # 16.

    def call(self, inputs):
        #about l2_normalize https://www.tensorflow.org/api_docs/python/tf/keras/backend/l2_normalize?hl=ja
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
    plt.savefig("../logs/graphs/accuracy.png")
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


# 異常検出モデルの作成
def lof(output_model, X_train):

    X_train = output_model.predict(X_train)
    X_train = X_train.reshape((len(X_train), -1))

    lof_scaler = MinMaxScaler()
    lof_scaler.fit(X_train)
    lof_scaler.transform(X_train)

    print("anomaly detection model creating...")
    # contamination = 学習データにおける外れ値の割合（大きいほど厳しく小さいほど緩い）{ 0.0 ~ 0.5 }
    # example-> k(n_neighbors=10**0.5=3) 10=num of class
    model = LocalOutlierFactor(n_neighbors=2, novelty=True, contamination=lof_contamination) # 20, novelty=True, contamination=0.001)
    model.fit(X_train[:1000])

    joblib.dump(lof_scaler, "../logs/models/lof_scaler.joblib")
    joblib.dump(model, "../logs/models/lof_model.joblib", compress=True)


def main(epochs=5, batch_size=128):

    if os.path.isdir("../logs/models"):
        shutil.rmtree("../logs/models")
    os.mkdir("../logs/models")

    if os.path.isdir("../logs/graphs"):
        shutil.rmtree("../logs/graphs")
    os.mkdir("../logs/graphs")

    features = ft.read_csv(file_path="../logs/train.csv", delimiter=",")
    X, y = get_data(data2ds=features)

    # one-hot vector形式に変換する
    num_of_category = len(ft.wav_labels)
    y = to_categorical(y, num_of_category)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

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

    l2con = L2ConstrainLayer()
    model.add(l2con)
    model.add(keras.layers.Dense(num_of_category, activation='softmax'))

    #Compiling the model
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=0.0001, amsgrad=True), # RMSprop(), # (learning_rate=1e-4),
                  metrics=["accuracy"], run_eagerly=True)

    model.summary()

    # callback function
    csv_cb = CSVLogger("../logs/models/train_log.csv")
    fpath = "../logs/models/model-{epoch:02d}-{loss:.2f}-{accuracy:.2f}-{val_loss:.2f}-{val_accuracy:.2f}-.h5"
    cp_cb = ModelCheckpoint(filepath=fpath, monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
    es_cb = EarlyStopping(monitor="val_loss", patience=2, verbose=1, mode="auto")
    tb_cb = TensorBoard(log_dir="../logs/tensor_board", histogram_freq=1)

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
    lof(output_model=output_model, X_train=X_train)

    K.clear_session()
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
