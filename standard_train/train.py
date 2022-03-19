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

import argparse
from optim.lr_scheduler import multistep_lr
from models.resnet import resnet18
from data.preprocess import build_dataloader
from utils import load_yaml
from utils import plot_traning_curves
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

if __name__ == '__main__':
    '''Load config file
    '''
    n_cfg = '../configs/normal_training_rn18.yaml'
    config = load_yaml(n_cfg)
    dataset_config = config['dataset_config']
    model_config = config['model_config']

    '''Build DataLoader
    '''
    ds_train, ds_test = build_dataloader(dataset_config)

    '''Build Model
    '''
    model = resnet18()
    model.summary()

    """ Build loss function
    """
    opt_cfg = config['optimizer_config']
    loss_fn = CategoricalCrossentropy(from_logits=False, label_smoothing=0.0)
    opt = SGD(lr=opt_cfg['lr'], momentum=opt_cfg['momentum'], nesterov=False)
    # top_k_acc = tf.keras.metrics.top_k_categorical_accurac()
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    """ Build callbacks
    """
    check_cfg = config['checkpoint_config']
    checkpoint_dir = os.path.join(check_cfg['dir'], config['mode'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    file_path = os.path.join(checkpoint_dir, '{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.h5')
    ckp = ModelCheckpoint(
        file_path,
        monitor="accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
        initial_value_threshold=None,
    )

    lr_callback = LearningRateScheduler(schedule=multistep_lr, verbose=1)

    """ Build trainer
     """
    # model.fit(cifar100_training_loader)
    history = model.fit(ds_train,
                        epochs=dataset_config['num_epochs'],
                        verbose=2,
                        steps_per_epoch = dataset_config['n_train_samples']//dataset_config['batch_size'],
                        validation_data=ds_test,
                        use_multiprocessing=True,
                        workers=dataset_config['num_workers'],
                        callbacks=[lr_callback, ckp])

    # Plot
    plot_traning_curves(history)