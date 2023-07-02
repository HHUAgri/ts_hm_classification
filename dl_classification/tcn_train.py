# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import time
import argparse
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from tcn_classifier import TCNClassifier
from ts_dataset import TSDataset
from classification_loss import ClassificationLosses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Time-series classification using TCN')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading (default: 2)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=99, help='upper epoch limit (default: 200)')

    parser.add_argument('--input_channel', type=int, default=3, help='number of channels of input (default: 4)')
    parser.add_argument('--num_classes', type=int, default=13, help='number of classes in input (default: 13)')

    parser.add_argument('--kernel_size', type=int, default=5, help='kernel size (default: 5)')
    parser.add_argument('--levels', type=int, default=2, help='# of levels (default: 4)')
    parser.add_argument('--nhid', type=int, default=120, help='number of hidden units per layer (default: 48)')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (default: 0.05)')

    parser.add_argument('--optimizer', type=str, default='adamax', help='optimizer to use (default: adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate (default: 1e-4)')
    parser.add_argument('--loss', type=str, default='ce', help='loss function to use (default: ce)')

    args = parser.parse_args()
    return args


def train(net, train_loader, val_loader, optimizer, scheduler, criterion, num_epoch):

    max_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epoch):
        net.train()

        epoch_start = time.time()
        print('\nEpoch: %d' % epoch)

        train_correct = 0
        train_total = 0
        train_loss = 0
        idx = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            idx = batch_idx
            # can't be net.zero_grad(), i dont know why?
            optimizer.zero_grad()

            # Inference
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Compute accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels.data).cpu().sum().item()
        # for batch
        scheduler.step()

        # accuracy
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / (idx + 1)

        epoch_end = time.time()
        print('train_acc = %.4f, train_loss = %.4f, Time = %.1fs' % (train_acc, train_loss, (epoch_end - epoch_start)))

        # validation
        test_acc, test_loss = validate(net, val_loader, criterion)

        # save model
        save_model_path = os.path.join('./model_save/tcn/', 'model_{}_{}'.format(epoch, max_val_acc))
        if test_acc > max_val_acc:
            max_val_acc = test_acc
            best_epoch = epoch

            net.cpu()
            torch.save(net, save_model_path)
            net.to(device)
    # for epoch

    print('\n\n### Best Epoch: %d, Best Results: %.4f' % (best_epoch, max_val_acc))
    return True


def validate(net, val_loader, criterion):

    epoch_start = time.time()

    with torch.no_grad():
        net.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0
        idx = 0

        val_softs = []
        val_preds = []
        val_trues = []

        for batch_idx, (inputs, labels) in enumerate(val_loader):
            idx = batch_idx

            # prediction
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels.data).cpu().sum().item()

            # val_softs.extend(outputs.detach().cpu().numpy())
            val_preds.extend(predicted.detach().cpu().numpy())
            val_trues.extend(labels.detach().cpu().numpy())
        # for batch

        # accuracy
        test_acc = 100. * val_correct / val_total
        test_loss = val_loss / (idx + 1)

        acc = 100 * accuracy_score(val_trues, val_preds)
        f1 = 100 * f1_score(val_trues, val_preds, average='weighted', zero_division=0)
        precision = 100 * precision_score(val_trues, val_preds, average='weighted', zero_division=0)
        recall = 100 * recall_score(val_trues, val_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(val_trues, val_preds)

        epoch_end = time.time()
        print('test_acc = %.4f, test_loss = %.4f, Time = %.1s' % (test_acc, test_loss, epoch_end - epoch_start))
        print('test_acc = %.4f, test_f1 = %.4f, test_p = %.4f, test_r = %.4f.' % (acc, f1, precision, recall))
        print(cm)

    return test_acc, test_loss


def main():
    print("### PyTorch Time-series Classification Training ##################################")

    # Reading option
    args = parse_args()
    num_workers = args.num_workers
    batch_size = args.batch_size
    num_epoch = args.epochs

    num_classes = args.num_classes
    input_channels = args.input_channel

    kernel_size = args.kernel_size
    levels = args.levels
    nhid = args.nhid
    channel_sizes = [nhid] * levels

    dropout = args.dropout
    optimizer = args.optimizer
    lr = args.lr
    criterion_name = args.loss

    criterion_name = 'focal'
    # dataset = 'dijon_8m_4mean'
    dataset = 'dijon_s1_mean'


    # Declaration of data loader
    train_dataset = TSDataset(dataset, 'train')

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


    # Declaration of network
    net = TCNClassifier(input_channels, num_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout).to(device)
    # loss and optimizer
    criterion = ClassificationLosses().build_loss(criterion_name).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)


    # train
    train(net, train_loader, test_loader, optimizer, scheduler, criterion, num_epoch)


    #################################################################
    print('')


if __name__ == "__main__":
    main()
