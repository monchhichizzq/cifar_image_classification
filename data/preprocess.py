# -*- coding: utf-8 -*-
# @Time    : 2022/3/13 9:00
# @Author  : Zeqi@@
# @FileName: preprocess.py.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()

def image_pad(img, pad=4):
    h,w,c = img.shape
    pad_h = h + 2*pad
    pad_w = w + 2*pad
    img = tf.image.pad_to_bounding_box(
        img, offset_height=pad, offset_width=pad, target_height=pad_h, target_width=pad_w
    )
    return img

def random_crop(img, size=(32, 32, 3)):
    img = tf.image.random_crop(value=img, size=size)
    return img

def random_flip(img):
    img = tf.image.random_flip_left_right(img, seed=32)
    return img

def random_rotate(img, rg=15):
    img = tf.keras.preprocessing.image.random_rotation(img, rg, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest',
    cval=0.0, interpolation_order=1)
    return img

def resize(img, size):
    img = tf.image.resize(
        img, size, method='bilinear', preserve_aspect_ratio=False,
        antialias=False, name=None
    )
    return img

def normalize(img, mean, std):
    img = img/255
    #img = tf.cast(img, dtype=tf.float32)
    # image = img[..., ::-1]
    # n_img = tf.Variable(tf.zeros(shape=img.shape, dtype=tf.float32))
    for i in range(img.shape[-1]):
        img[..., i] = (img[..., i] - mean[i]) / std[i]
        # n_img[..., i].assign((img[..., i] - mean[i]) / std[i])
    return img

def preprocess_img(img, label, mean, std, size, augment=False):
    img = tf.cast(img, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)

    if augment:
        img = image_pad(img)
        img = random_crop(img, size)
        img = random_flip(img)
        img = random_rotate(img)
    else:
        # img = resize(img, size)
        # img =  tf.convert_to_tensor(img, dtype=tf.float32)
        img = np.array(img)
        # img = img

    img = normalize(img, mean, std)
    img = tf.cast(img, dtype=tf.float32)

    nb_classes = y_train.max() + 1
    label = to_categorical(label, num_classes=nb_classes)
    label = tf.squeeze(label)

    img = tf.cast(img, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)

    return img, label

@tf.autograph.experimental.do_not_convert
def load_image_wrapper(img, label):
    # result_tensors = tf.py_function(preprocess_img, inp=[img, label, mean, std, in_size, True], Tout=[tf.float32, tf.float32])
    img.set_shape([32, 32, 3])
    label.set_shape([100])
    return img, label


def get_train_dataloader(mean, std, in_size, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(len(y_train))
    dataset = dataset.map(
        lambda x, y: tf.py_function(preprocess_img, inp=[x, y, mean, std, in_size, True], Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(load_image_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def get_test_dataloader(mean, std, in_size, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.map(
        lambda x, y: tf.py_function(preprocess_img, inp=[x, y, mean, std, in_size, False], Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(load_image_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

if __name__ == '__main__':
    train_data = get_train_dataloader(in_size=(32,32,3), batch_size=64)

    for data in train_data:
        img, label = data
        print(img.shape, label.shape)




