# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import time
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

from data import get_hierarchy_label

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(net, train_loader, test_loader, optimizer, scheduler, total_loss, num_epoch, device):
    """

    :param net:
    :param train_loader:
    :param test_loader:
    :param optimizer:
    :param scheduler:
    :param total_loss:
    :param num_epoch:
    :param device:
    :return:
    """

    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    max_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epoch):
        epoch_start = time.time()
        print('\nEpoch: %d' % epoch)

        net.train()

        train_loss = 0

        l1_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0
        l1_total = 0
        species_total = 0

        idx = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            idx = batch_idx
            optimizer.zero_grad()

            # pack ground truth labels
            inputs, labels = inputs.to(device), labels.to(device)
            l1_target, l2_target = get_hierarchy_label(labels)
            target = (l1_target, l2_target)

            # prediction
            y_sig_l1, y_sig_l2, y_l1, y_l2 = net(inputs)
            output = (y_sig_l1, y_sig_l2, y_l1, y_l2)

            loss = total_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 计算精度(不可导)
            # y_sig_l1, y_sig_l2, y_l1, y_l2 = outputs
            with torch.no_grad():
                # level 1 node
                _, l1_predicted = torch.max(y_l1.data, 1)
                l1_total += l1_target.size(0)
                l1_correct += l1_predicted.eq(l1_target.data).cpu().sum().item()
                # level 2 node
                _, l2_predicted_soft = torch.max(y_l2.data, 1)
                _, l2_predicted_sig = torch.max(y_sig_l2.data, 1)
                species_total += l2_target.size(0)
                species_correct_soft += l2_predicted_soft.eq((l2_target-4).data).cpu().sum().item()
                species_correct_sig += l2_predicted_sig.eq((l2_target-4).data).cpu().sum().item()
        # for batch
        scheduler.step()

        train_order_acc = 100. * l1_correct / l1_total
        train_species_acc_soft = 100. * species_correct_soft / species_total
        train_species_acc_sig = 100. * species_correct_sig / species_total
        train_loss = train_loss / (idx + 1)
        epoch_end = time.time()
        print('train_order_acc = %.4f, train_species_acc_soft = %.4f, train_species_acc_sig = %.4f, train_loss = %.4f, Time = %.1fs' % \
            (train_order_acc, train_species_acc_soft, train_species_acc_sig, train_loss, (epoch_end - epoch_start)))

        test_order_acc, test_species_acc_soft, test_species_acc_sig, test_loss = test(net, test_loader, total_loss, device)

        save_model_path = os.path.join('./model_save', 'model_{}_{}'.format(epoch, max_val_acc))
        if test_species_acc_soft > max_val_acc:
            max_val_acc = test_species_acc_soft
            best_epoch = epoch

            net.cpu()
            # torch.save(net, save_model_path)
            torch.save(net.state_dict(), save_model_path)
            net.to(device)
    # for epoch in range(epoch)

    print('\n\n### Best Epoch: %d, Best Results: %.4f' % (best_epoch, max_val_acc))
    return True


def test(net, test_loader, total_loss, device):
    """

    :param net:
    :param test_loader:
    :param total_loss:
    :param device:
    :return:
    """
    epoch_start = time.time()

    with torch.no_grad():
        net.eval()

        test_loss = 0
        order_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0
        order_total = 0
        species_total = 0
        idx = 0

        l1_test_preds = []
        l1_test_trues = []
        l2_test_preds = []
        l2_test_trues = []

        for batch_idx, (inputs, labels) in enumerate(test_loader):
            idx = batch_idx

            # pack ground truth labels
            inputs, labels = inputs.to(device), labels.to(device)
            l1_target, l2_target = get_hierarchy_label(labels)
            target = (l1_target, l2_target)

            # prediction
            y_sig_l1, y_sig_l2, y_l1, y_l2 = net(inputs)
            output = (y_sig_l1, y_sig_l2, y_l1, y_l2)

            loss = total_loss(output, target)
            test_loss += loss.item()

            # 上层节点
            _, l1_predicted = torch.max(y_l1.data, 1)
            order_total += l1_target.size(0)
            order_correct += l1_predicted.eq(l1_target.data).cpu().sum().item()
            # 下层节点
            _, l2_predicted_soft = torch.max(y_l2.data, 1)
            _, l2_predicted_sig = torch.max(y_sig_l2.data, 1)
            species_total += l2_target.size(0)
            species_correct_soft += l2_predicted_soft.eq((l2_target - 4).data).cpu().sum().item()
            species_correct_sig += l2_predicted_sig.eq((l2_target - 4).data).cpu().sum().item()

            # store results
            l1_test_preds.extend(l1_predicted.detach().cpu().numpy())
            l1_test_trues.extend(l1_target.detach().cpu().numpy())
            l2_test_preds.extend(l2_predicted_soft.detach().cpu().numpy())
            l2_test_trues.extend((l2_target - 4).detach().cpu().numpy())
        # for batch

        # accuracy
        test_order_acc = 100. * order_correct / order_total
        test_species_acc_soft = 100. * species_correct_soft / species_total
        test_species_acc_sig = 100. * species_correct_sig / species_total
        test_loss = test_loss / (idx + 1)

        l1_acc = accuracy_score(l1_test_trues, l1_test_preds)
        l1_f1 = f1_score(l1_test_trues, l1_test_preds, average='weighted', zero_division=0)
        l1_precision = precision_score(l1_test_trues, l1_test_preds, average='weighted', zero_division=0)
        l1_recall = recall_score(l1_test_trues, l1_test_preds, average='weighted', zero_division=0)
        l1_mcc = matthews_corrcoef(l1_test_trues, l1_test_preds)
        l1_cm = confusion_matrix(l1_test_trues, l1_test_preds)

        l2_acc = accuracy_score(l2_test_trues, l2_test_preds)
        l2_f1 = f1_score(l2_test_trues, l2_test_preds, average='weighted', zero_division=0)
        l2_precision = precision_score(l2_test_trues, l2_test_preds, average='weighted', zero_division=0)
        l2_recall = recall_score(l2_test_trues, l2_test_preds, average='weighted', zero_division=0)
        l2_mcc = matthews_corrcoef(l2_test_trues, l2_test_preds)
        l2_cm = confusion_matrix(l2_test_trues, l2_test_preds)

        epoch_end = time.time()
        print('test_order_acc = %.4f, test_species_acc_soft = %.4f, test_species_acc_sig = %.4f, test_loss = %.4f, Time = %.1s' % \
              (test_order_acc, test_species_acc_soft, test_species_acc_sig, test_loss, epoch_end - epoch_start))
        print('l1_test_acc = %.4f, l1_test_f1 = %.4f, l1_test_p = %.4f, l1_test_r = %.4f, l1_test_mcc = %.4f.' % (l1_acc, l1_f1, l1_precision, l1_recall, l1_mcc))
        print(l1_cm)
        print('l2_test_acc = %.4f, l2_test_f1 = %.4f, l2_test_p = %.4f, l2_test_r = %.4f, l2_test_mcc = %.4f.' % (l2_acc, l2_f1, l2_precision, l2_recall, l2_mcc))
        print(l2_cm)

    return test_order_acc, test_species_acc_soft, test_species_acc_sig, test_loss

