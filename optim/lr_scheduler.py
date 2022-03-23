# -*- coding: utf-8 -*-
# @Time    : 2022/3/13 10:42
# @Author  : Zeqi@@
# @FileName: lr_scheduler.py.py
# @Software: PyCharm

import tensorflow as tf

def multistep_lr(epoch, init_lr):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """

    init_lr = 0.1
    milestones = [60, 120, 160]
    gamma = 0.2

    print('initial lr: {}, milestones: {}, gamma: {}'.format(init_lr, milestones, gamma))

    if epoch > milestones[2]:
        lr = init_lr * gamma ** 3
    elif  epoch > milestones[1]:
        lr = init_lr * gamma ** 2
    elif epoch > milestones[0]:
        lr = init_lr * gamma ** 1
    else:
        lr = init_lr * gamma ** 0
    tf.print('Learning rate: ', lr)
    return lr
