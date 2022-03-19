# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 13:40
# @Author  : Zeqi@@
# @FileName: preprocess.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test

def build_dataloader(dataset_config):
    # load data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    y_train = to_categorical(
        y_train,
        dataset_config['num_classes']
    )
    y_test = to_categorical(
        y_test,
        dataset_config['num_classes']
    )
    x_train, x_test = color_preprocessing(x_train, x_test)

    datagen = ImageDataGenerator(
        width_shift_range=0.125,
        height_shift_range=0.125,
        fill_mode='constant',
        horizontal_flip=True,
        rotation_range=15,
        cval=0.
    )
    datagen.fit(x_train)

    train_generator = datagen.flow(
        x_train,
        y_train,
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, 32, 32, 3], [None, 100]),
    )

    train_ds= train_ds.take(dataset_config['n_train_samples'])
    train_ds = train_ds.map(lambda img, label: (tf.image.resize(img,
                                                                dataset_config['img_size'],
                                                                method='bilinear'), label))
    train_ds.shuffle(buffer_size=dataset_config['n_train_samples'])
    train_ds.batch(dataset_config['batch_size'])
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)


    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(lambda img, label: (tf.image.resize(img,
                                                              dataset_config['img_size'],
                                                              method='bilinear'), label))
    test_ds.take(dataset_config['n_val_samples'])
    test_ds = test_ds.batch(dataset_config['batch_size'])


    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds