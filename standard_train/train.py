# -*- coding: utf-8 -*-
# @Time    : 2022/3/13 9:59
# @Author  : Zeqi@@
# @FileName: train.py.py
# @Software: PyCharm

import os
import sys
sys.path.append('..')
print(os.getcwd())
from utils import load_yaml, plot_traning_curves
from data.preprocess import get_train_dataloader
from data.preprocess import get_test_dataloader
from optim.lr_scheduler import multistep_lr
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

if __name__ == '__main__':

    """ Load configuration
    """
    cfg_path = '../configs/base_config.yaml'
    cfg = load_yaml(cfg_path)

    """ Build dataloader
    """
    cfg_data = cfg['dataset_config']
    cifar100_training_loader = get_train_dataloader(
        cfg_data['in_mean'],
        cfg_data['in_std'],
        in_size=cfg_data['input_shape'],
        batch_size=cfg_data['batch_size'],
    )

    cifar100_test_loader = get_test_dataloader(
        cfg_data['in_mean'],
        cfg_data['in_std'],
        in_size=cfg_data['input_shape'],
        batch_size=cfg_data['batch_size'],
    )

    # for data in cifar100_training_loader:
    #     img, label = data
    #     print(img.shape, label.shape)

    from models.resnet import resnet18
    model = resnet18()
    model.summary()

    """ Build loss function
     """
    opt_cfg = cfg['optimizer_config']
    loss_fn =  CategoricalCrossentropy(from_logits=False, label_smoothing=0.0)
    opt = SGD(lr=opt_cfg['lr'], momentum=opt_cfg['momentum'], nesterov = False)
    # top_k_acc = tf.keras.metrics.top_k_categorical_accurac()
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    """ Build callbacks
    """
    check_cfg = cfg['checkpoint_config']
    checkpoint_dir = os.path.join(check_cfg['dir'], cfg['mode'])
    os.makedirs(checkpoint_dir,  exist_ok=True)
    file_path = os.path.join(checkpoint_dir, '{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.h5')
    ckp = ModelCheckpoint(
            file_path,
            monitor="accuracy",
            verbose=0,
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
    history = model.fit(cifar100_training_loader,
                      epochs=cfg_data['num_epochs'],
                      verbose=2,
                      validation_data=cifar100_test_loader,
                      use_multiprocessing=True,
                      workers = cfg_data['num_workers'],
                      callbacks=[lr_callback, ckp])

    # Plot
    plot_traning_curves(history)