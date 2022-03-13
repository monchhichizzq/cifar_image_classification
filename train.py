# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 15:48
# @Author  : Zeqi@@
# @FileName: train.py.py
# @Software: PyCharm

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from utils import load_yaml
from datasets.preprocess import get_training_dataloader
from datasets.preprocess import get_test_dataloader
from optim.lr_scheduler import WarmUpLR
#
# from utils import  get_training_dataloader, get_test_dataloader, WarmUpLR, \
#     most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

def train(epoch):

    start = time.time()
    net.train()
    correct = 0.0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        # print(batch_index, images.shape, labels.shape)

        if cfg['gpus']:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAccuracy: {:.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            correct.float() / len(cifar100_training_loader.dataset),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * cfg_data['batch_size'] + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        if epoch <= opt_cfg['warm_up']:
            warmup_scheduler.step()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if cfg['gpus']:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if cfg['gpus']:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    """ Load configuration
    """
    cfg_path = 'configs/base_config.yaml'
    cfg = load_yaml(cfg_path)


    """ Build dataloader
    """
    cfg_data = cfg['dataset_config']
    cifar100_training_loader = get_training_dataloader(
        cfg_data['in_mean'],
        cfg_data['in_std'],
        num_workers=cfg_data['num_workers'],
        batch_size=cfg_data['batch_size'],
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        cfg_data['in_mean'],
        cfg_data['in_std'],
        num_workers=cfg_data['num_workers'],
        batch_size=cfg_data['batch_size'],
        shuffle=True
    )

    """ Build network
    """
    # from models.resnet import resnet50
    # net = resnet50()
    # from models.senet import seresnet50
    # net = seresnet50()
    # from models.shufflenetv2 import ShuffleNetV2
    # net = shufflenetv2()
    # from models.shufflenet import shufflenet
    # net = shufflenet()

    from models.mobilenet import mobilenet
    net = mobilenet()

    if cfg['gpus']:
        net = net.cuda()

    """ Build loss function
    """
    loss_function = nn.CrossEntropyLoss()

    """ Build optimizer
    """
    opt_cfg = cfg['optimizer_config']
    optimizer = optim.SGD(net.parameters(), lr=opt_cfg['lr'], momentum=opt_cfg['momentum'], weight_decay=opt_cfg['weight_decay'])
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt_cfg['milestones'], gamma=opt_cfg['gamma'])
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * opt_cfg['warm_up'])

    check_cfg = cfg['checkpoint_config']
    checkpoint_dir = os.path.join(check_cfg['dir'], cfg['mode'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    """ Build trainer
    """
    best_acc = 0
    for epoch in range(1, cfg_data['num_epochs'] + 1):
        if epoch > opt_cfg['warm_up']:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        # save best acc model
        if best_acc < acc:
            weights_path = os.path.join(checkpoint_dir, '%s-epoch_%d-acc_%.4f.pth'%(cfg['mode'], epoch, acc))
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue


