# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 14:12
# @Author  : Zeqi@@
# @FileName: eval.py.py
# @Software: PyCharm

import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from utils import load_yaml
from datasets.preprocess import get_training_dataloader
from datasets.preprocess import get_test_dataloader
from tqdm import  tqdm
# from conf import settings
# from utils import get_network, get_test_dataloader

from models.resnet import resnet18

net = resnet18()
pretrained_weights = 'logs/resnet18_cifar_100/resnet18_cifar_100-epoch_193-acc_0.7628.pth'
net.load_state_dict(torch.load(pretrained_weights))



def evaluate(net, data_loader):
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(tqdm(data_loader)):

            if cfg['gpus']:
                image = image.cuda()
                label = label.cuda()

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)
            # _, pred = output.max(1)
            # pred = torch.unsqueeze(pred, 1)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top1
            correct_1 += correct[:, :1].sum()

    # if cfg['gpus']:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')

    print()
    top1_acc = np.round((correct_1 / len(data_loader.dataset)).cpu(), 4)
    top5_acc = np.round((correct_5 / len(data_loader.dataset)).cpu(), 4)

    print("Top 1 err: {}%".format((1 - top1_acc) * 100))
    print("Top 1 acc: {}%".format(top1_acc * 100))
    print("Top 5 err: {}%".format((1 - top5_acc) * 100))
    print("Top 5 acc: {}%".format(top5_acc * 100))


if __name__ == '__main__':

    """ Load configuration
     """
    cfg_path = 'configs/resnet50_config.yaml'
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
        shuffle=False
    )

    net.eval()
    if cfg['gpus']:
        net.cuda()

    print(" ")
    print("Train: ")
    evaluate(net, cifar100_training_loader)

    print(" ")
    print("Test: ")
    evaluate(net, cifar100_test_loader)

    # correct_1 = 0.0
    # correct_5 = 0.0
    # total = 0
    #
    # with torch.no_grad():
    #     for n_iter, (image, label) in enumerate(tqdm(cifar100_test_loader)):
    #
    #         if cfg['gpus']:
    #             image = image.cuda()
    #             label = label.cuda()
    #
    #         output = net(image)
    #         _, pred = output.topk(5, 1, largest=True, sorted=True)
    #         # _, pred = output.max(1)
    #         # pred = torch.unsqueeze(pred, 1)
    #
    #         label = label.view(label.size(0), -1).expand_as(pred)
    #         correct = pred.eq(label).float()
    #
    #         #compute top 5
    #         correct_5 += correct[:, :5].sum()
    #
    #         #compute top1
    #         correct_1 += correct[:, :1].sum()
    #
    # # if cfg['gpus']:
    # #     print('GPU INFO.....')
    # #     print(torch.cuda.memory_summary(), end='')
    #
    # print()
    # top1_acc = np.round((correct_1 / len(cifar100_test_loader.dataset)).cpu(), 4)
    # top5_acc = np.round((correct_5 / len(cifar100_test_loader.dataset)).cpu(), 4)
    # print("Top 1 err: {}%".format((1 - top1_acc)*100))
    # print("Top 1 acc: {}%".format(top1_acc*100))
    # print("Top 5 err: {}%".format((1 - top5_acc) * 100))
    # print("Top 5 acc: {}%".format(top5_acc * 100))
    print("Parameter numbers: {} M".format(sum(p.numel() for p in net.parameters())/10**6))
