# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tree_hierarchy_label_crop_source = [
    [1, 21],
    [2, 21],
    [3, 21],
    [4, 21],
    [5, 22],
    [6, 23],
    [7, 23],
    [8, 23],
    [9, 24],
    [10, 24],
    [11, 24],
    [12, 24]
]
tree_hierarchy_label_crop = [
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 1],
    [9, 2],
    [10, 2],
    [11, 2],
    [12, 3],
    [13, 3],
    [14, 3],
    [15, 3]
]
tree_total_nodes_crop = 16
tree_levels_crop = 2


def generate_state_space(hierarchy, total_nodes, levels):
    """

    :param hierarchy:
    :param total_nodes:
    :param levels:
    :return:
    """
    state_space = torch.zeros(total_nodes + 1, total_nodes).to(device)
    recorded = torch.zeros(total_nodes)

    ii = 1

    if levels == 2:
        for path in hierarchy:
            # path[0] = path[0] + 3
            # path[1] = path[1] - 21

            if recorded[path[1]] == 0:
                state_space[ii, path[1]] = 1
                recorded[path[1]] = 1
                ii += 1
            state_space[ii, path[1]] = 1
            state_space[ii, path[0]] = 1
            ii += 1

    elif levels == 3:
        # for path in hierarchy:
        #     path[0] = path[0] - 1
        #     path[1] = path[1] - 21
        #
        #     if recorded[path[1]] == 0:
        #         state_space[ii, path[1]] = 1
        #         recorded[path[1]] = 1
        #         ii += 1
        #     if recorded[path[2]] == 0:
        #         state_space[ii, path[1]] = 1
        #         state_space[ii, path[2]] = 1
        #         recorded[path[2]] = 1
        #         ii += 1
        #     state_space[ii, path[1]] = 1
        #     state_space[ii, path[2]] = 1
        #     state_space[ii, path[0]] = 1
        #     ii += 1
        pass

    if ii == total_nodes + 1:
        return state_space
    else:
        print('Invalid State Space')
        return None


def find_hierarchy_label(leaf_label):
    return_value = None
    for it in tree_hierarchy_label_crop:
        if it[0] == leaf_label:
            return_value = it
    if return_value is None:
        print("Could not find hierarchy label in tree")
    return return_value


def get_hierarchy_label(leaf_label, dataset='crop'):

    l1_label_list = []
    l2_label_list = []

    for i in range(leaf_label.size(0)):
        if dataset == 'crop':
            hierarchy_label = find_hierarchy_label(int(leaf_label[i]))
            l1_label_list.append(hierarchy_label[1])

        l2_label_list.append(int(leaf_label[i]))
    # for

    l1_label_list = torch.from_numpy(np.array(l1_label_list)).to(device)
    l2_label_list = torch.from_numpy(np.array(l2_label_list)).to(device)
    return l1_label_list, l2_label_list


class TreeLoss(nn.Module):
    def __init__(self, hierarchy, total_nodes, levels):
        """

        :param hierarchy:
        :param total_nodes:
        :param levels:
        """
        super(TreeLoss, self).__init__()
        self.state_space = generate_state_space(hierarchy, total_nodes, levels).to(device)

    def forward(self, pred_label, gt_label):
        index = torch.mm(self.state_space.to(torch.float), pred_label.to(torch.float).T)
        joint = torch.exp(index)
        z = torch.sum(joint, dim=0)
        loss = torch.zeros(pred_label.shape[0], dtype=torch.float64).to(device)
        for i in range(len(gt_label)):
            marginal = torch.sum(torch.index_select(joint[:, i], 0, torch.where(self.state_space[:, gt_label[i]] > 0)[0]))
            loss[i] = -torch.log(marginal / z[i])
        return torch.mean(loss)

    def inference(self, pred_label):
        with torch.no_grad():
            index = torch.mm(self.state_space, pred_label.T)
            joint = torch.exp(index)
            z = torch.sum(joint, dim=0)
            margin = torch.zeros((pred_label.shape[0], pred_label.shape[1]), dtype=torch.float64).to(device)
            for i in range(pred_label.shape[0]):
                for j in range(pred_label.shape[1]):
                    margin[i, j] = torch.sum(torch.index_select(joint[:, i], 0, torch.where(self.state_space[:, j] > 0)[0]))
            return margin


class CategoryLoss(nn.Module):
    def __init__(self):
        super(CategoryLoss, self).__init__()

    def forward(self, output, target):
        # target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(output.to(torch.float64).to(device), target.to(torch.long).to(device))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class TotalLoss(nn.Module):
    def __init__(self, hierarchy, total_nodes, levels):
        super(TotalLoss, self).__init__()

        self.loss_tree = TreeLoss(hierarchy, total_nodes, levels)
        self.loss_category_l1 = CategoryLoss()
        self.loss_category_l2 = CategoryLoss()

        self.weight_tree = 2.0
        self.weight_category_l1 = 1.0
        self.weight_category_l2 = 4.0

    def forward(self, output, target):
        l1_label, l2_label = target
        l1_sig, l2_sig, l1, l2 = output
        l12_sig = torch.cat([l1_sig, l2_sig], 1).to(device)

        loss = 0.0
        loss += self.loss_tree(l12_sig, l2_label) * self.weight_tree
        loss += self.loss_category_l1(l1, l1_label) * self.weight_category_l1
        loss += self.loss_category_l2(l2, l2_label-4) * self.weight_category_l2

        return loss


def main():
    # batch_size x channels x time_step
    criterion = TotalLoss(tree_hierarchy_label_crop, tree_total_nodes_crop, tree_levels_crop)

    target = torch.randint(4, 15, (9,))
    l1_label, l2_label = get_hierarchy_label(target)

    # y_sig_l1, y_sig_l2, y_l1, y_l2
    y1, y2, y3, y4 = torch.randint(0, 3, (9, 4)), torch.randint(4, 15, (9, 12)), torch.randint(1, 3, (9, 4)), torch.randint(4, 15, (9, 12))
    output = torch.cat([y1, y2], 1).to(device)

    loss_value = criterion((output, y3, y4), (l1_label.to(device), l2_label.to(device)))


if __name__ == "__main__":
    main()
