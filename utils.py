# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 13:46
# @Author  : Zeqi@@
# @FileName: utils.py
# @Software: PyCharm


import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

def load_yaml(file_path: Path) -> dict():
    with open(file_path, "r") as f:
        try:
            data = yaml.safe_load(f)
            return data
        except yaml.YAMLError as exc:
            print("Yaml exception {}".format(exc))


def plot_traning_curves(history, dir):
    path = os.path.join(dir, 'history')
    os.makedirs(path, exist_ok=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(path + '/Accuracy_plot.png', dpi=600)

    plt.figure()
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path + '/loss_plot.png', dpi=600)
    plt.show()