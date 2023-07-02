# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""

import torch
import torch.nn as nn

from .backbone import MSResNet
from .tpn import TemporalProposalLayer


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class HMTSNet(nn.Module):
    def __init__(self, in_planes, num_classes=[4, 12]):
        super(HMTSNet, self).__init__()

        # backbone feature block
        self.num_feat = 768
        self.bk = MSResNet(input_channel=in_planes)

        # more features for two branches.
        # l1 and l2 for upper and lower level
        self.cb_planes = 128
        self.tpn = TemporalProposalLayer(self.num_feat)
        self.cb_l1 = BasicConv(self.num_feat, self.cb_planes)
        self.cb_l2 = BasicConv(self.num_feat, self.cb_planes)
        self.pooling_l1 = nn.AdaptiveAvgPool1d(1)
        self.pooling_l2 = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

        # classification block for two branches.
        # l1 and l2 for upper and lower level
        self.fc_planes = 128
        self.fc_l1 = nn.Sequential(
            nn.BatchNorm1d(self.cb_planes),
            nn.Linear(self.cb_planes, self.fc_planes),
            nn.BatchNorm1d(self.fc_planes),
            # nn.Dropout(p=0.4),
            nn.ELU(inplace=True),
            nn.Linear(self.fc_planes, self.fc_planes)
        )
        self.fc_l2 = nn.Sequential(
            nn.BatchNorm1d(self.cb_planes),
            nn.Linear(self.cb_planes, self.fc_planes),
            nn.BatchNorm1d(self.fc_planes),
            # nn.Dropout(p=0.4),
            nn.ELU(inplace=True),
            nn.Linear(self.fc_planes, self.fc_planes)
        )

        # classification headers.
        # l1 and l2 for upper and lower level
        self.classifier_sig1 = nn.Sequential(nn.Linear(self.fc_planes, num_classes[0]), nn.Sigmoid())
        self.classifier_sig2 = nn.Sequential(nn.Linear(self.fc_planes, num_classes[1]), nn.Sigmoid())
        self.classifier_ll = nn.Sequential(nn.Linear(self.fc_planes, num_classes[0]))
        self.classifier_l2 = nn.Sequential(nn.Linear(self.fc_planes, num_classes[1]))

    def forward(self, x):
        batch_size, channels, times = x.size()
        feat = self.bk(x)

        # l1 and l2 for upper and lower level
        x_l1 = self.cb_l1(feat)
        x_l2 = self.cb_l2(feat)
        # x_l2 = self.cb_l2(self.tpn(feat))

        x_fc_l1 = self.pooling_l1(x_l1)
        x_fc_l1 = x_fc_l1.view(x_fc_l1.size(0), -1)
        x_fc_l1 = self.fc_l1(x_fc_l1)

        # x_l2 = torch.cat([x_l2, x_l1], dim=1)
        x_fc_l2 = self.pooling_l2(x_l2)
        x_fc_l2 = x_fc_l2.view(x_fc_l2.size(0), -1)
        x_fc_l2 = self.fc_l2(x_fc_l2)

        y_sig_l1 = self.classifier_sig1(self.relu(x_fc_l1))
        y_sig_l2 = self.classifier_sig2(self.relu(x_fc_l2))
        y_11 = self.classifier_ll(self.relu(x_fc_l1))
        y_12 = self.classifier_l2(self.relu(x_fc_l2))

        return y_sig_l1, y_sig_l2, y_11, y_12
        # return y_12


def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_msr():
    # batch_size x channels x time_step
    x = torch.randn(9, 2, 32)
    net = HMTSNet(in_planes=2)
    print(net)

    y = net(x)
    print(str(check_parameters(net))+' Mb')
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape)


if __name__ == "__main__":
    test_msr()
