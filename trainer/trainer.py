# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 13:37
# @Author  : Zeqi@@
# @FileName: train.py.py
# @Software: PyCharm

import os
import sys
from os.path import dirname, realpath, sep, pardir
abs_path = dirname(realpath(__file__)) + sep + pardir
sys.path.append(abs_path)

from loss.loss import Total_loss
from models.resnet import resnet18
from models.resnet_deep import resnet50
from data.preprocess import color_preprocessing, scheduler
from optim.lr_scheduler import multistep_lr
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy

lr                 = 0.1
momentum           = 0.9
activation_type    = 'relu'
num_classes        = 100
batch_size         = 128         # 64 or 32 or other
epochs             = 300
iterations         = 390
USE_BN=True
DROPOUT=0.2 # keep 80%
CONCAT_AXIS=3
weight_decay=1e-4
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
patience           = 20

# load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)

# Load model
model = resnet50(num_classes=num_classes)
# model = resnet18('ResNet18', nb_classes=num_classes,
#                  input_shape=(32, 32, 3),
#                  use_bn=True, use_bias=False)
# model.load_weights(model_path, by_name=True, skip_mismatch=True)
model.summary()

# set optimizer
sgd = SGD(lr=lr, momentum=momentum, nesterov=True)
total_loss = Total_loss(model)
loss_fn = CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
model.compile(loss=loss_fn,
              optimizer=sgd,
              metrics=['accuracy', 'top_k_categorical_accuracy'])

# set callback
""" Build callbacks
"""
ck_dir = 'logs'
checkpoint_dir = os.path.join(ck_dir, 'resnet50_label_smooth')
os.makedirs(checkpoint_dir, exist_ok=True)
file_path = os.path.join(checkpoint_dir, '{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.h5')
ckp = ModelCheckpoint(
    file_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
    initial_value_threshold=0.7,
)

lr_callback = LearningRateScheduler(schedule=multistep_lr, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=patience, verbose=1)
cbks = [lr_callback, ckp]

# set data augmentation
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             # rotation_range=15,
                             fill_mode='constant', cval=0.)
datagen.fit(x_train)
train_loader = datagen.flow(x_train, y_train, shuffle=True, batch_size=batch_size)
test_loader = datagen.flow(x_test, y_test, batch_size=batch_size)

model.fit(train_loader,
        # steps_per_epoch =iterations,
        epochs=epochs,
        callbacks=cbks,
        validation_data=test_loader,
        # use_multiprocessing=True,
        # workers=32,
        verbose=2)

