# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import torch
from torch import nn
import torch.nn.functional as F
from tcn import TemporalConvNet


class TCNClassifier(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        '''
        :param input_size: integer, number of input channels
        :param output_size: integer, number of output classes
        :param num_channels: array of integer, number of output channels per level
        :param kernel_size:
        :param dropout:
        '''
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        x = self.tcn(inputs)
        # Classification
        x = self.linear(x[:, :, -1])
        return F.log_softmax(x, dim=1)


def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_tcnclassifier():
    # batch_size x channels x time_step
    x = torch.randn(9, 4, 17)
    tcn_net = TCNClassifier(4, 13, num_channels=[8,8,8,8], kernel_size=3, dropout=0.2)
    print(tcn_net)

    y = tcn_net(x)
    print(y[1].shape)


if __name__ == "__main__":
    test_tcnclassifier()
