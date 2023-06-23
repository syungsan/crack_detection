#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns
sns.set()


def plot_confusion_matrix(cm, save_dir):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, fmt='d')
    ax.set_title("Confusion Matrix")

    plt.savefig(save_dir, format="png")
    plt.show()


def plot_roc_curve(y_trues, y_scores, save_dir):

    fpr, tpr, thresholds = roc_curve(y_trues, y_scores)
    _auc = auc(fpr, tpr)
    print("ROC-AUC {}".format(_auc))

    plt.plot(fpr, tpr, label="ROC Curve (area = %.2f)" % _auc)
    plt.plot(np.linspace(1, 0, len(fpr)), np.linspace(1, 0, len(fpr)), label="Random ROC Curve (area = %.2f)" % 0.5, linestyle="--", color="gray")

    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)

    plt.savefig(save_dir, format="png")

    plt.show()
    plt.close()

    return _auc

def plot_pr_curve(y_trues, y_scores, save_dir):

    precision, recall, thresholds = precision_recall_curve(y_trues, y_scores)

    _auc = auc(recall, precision)
    print("PR-AUC {}".format(_auc))

    plt.plot(recall, precision, label="PR Curve (area = %.2f)" % _auc)
    plt.legend()
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)

    plt.savefig(save_dir, format="png")

    plt.show()
    plt.close()

    return _auc
