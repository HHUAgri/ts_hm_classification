# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import argparse
import torch

from data import CropDataset
from data import tree_hierarchy_label_crop, tree_total_nodes_crop, tree_levels_crop
from model import HMTSNet, TotalLoss
from model_train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default='./pre-trained/resnet50-19c8e357.pth',
                        help='Path of pre-trained model')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--proportion', type=float, default=1.0, help='Proportion of species label')
    parser.add_argument('--num_epoch', default=80, type=int, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--dataset', type=str, default='dijon_s1_mean', help='dataset name')
    parser.add_argument('--lr_adjust', type=str, default='Cos', help='Learning rate schedule', choices=['Cos', 'Step'])
    parser.add_argument('--device', nargs='+', default='2', help='GPU IDs for DP training')

    args = parser.parse_args()
    return args


def main():
    print("###########################################################")
    print("### PyTorch HMTS classification ###########################")
    print("###########################################################")

    # cmd line
    opts = parse_args()
    num_workers = opts.num_workers
    batch_size = opts.batch_size
    num_epoch = opts.num_epoch

    # dataset = 'dijon_8m_4mean'
    dataset = 'dijon_s1_mean'

    # dataset
    train_dataset = CropDataset(dataset, 'train')
    # test_dataset = CropDataset(dataset, 'test')
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    # test_loader = None

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # network model
    net = HMTSNet(in_planes=3, num_classes=[4, 12]).to(device)

    # loss and optimizer
    total_loss = TotalLoss(tree_hierarchy_label_crop, tree_total_nodes_crop, tree_levels_crop).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    # train
    train(net, train_loader, test_loader, optimizer, scheduler, total_loss, num_epoch, device)

    # over
    print("\n### Task over #############################################")


if __name__ == "__main__":
    main()
