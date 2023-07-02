# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalProposalLayer(nn.Module):
    """

    Note: affine_matrix中，am[0][0]与am[0][2]分别对应X方向上的缩放比例和平移距离
    """
    def __init__(self, in_channel):
        super(TemporalProposalLayer, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv1d(in_channel, out_channels=8, kernel_size=7),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveMaxPool1d(output_size=6),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(1 * 8 * 6, 24),
            nn.ReLU(True),
            nn.Linear(24, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def tpl(self, x):
        in_batch, in_channel, in_size = x.size()

        xs = self.localization(x)
        xs = xs.view(-1, 1 * 8 * 6)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        affine_matrix = theta

        # affine_matrix = torch.zeros_like(theta)
        # mask = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.bool, device=x.device)
        # mask = torch.stack([mask] * in_batch).bool()
        # affine_matrix[mask] = theta.masked_select(mask)

        # print(affine_matrix)
        x = x.unsqueeze(-2)
        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, padding_mode='border', align_corners=False)

        return x.squeeze(-2)

    def forward(self, x):
        out = self.tpl(x)
        return out


def test_tpn():
    model = TemporalProposalLayer(in_channel=1).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    loss_fun = torch.nn.MSELoss()

    src_data = np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
                         [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]])
    src_batch = torch.from_numpy(src_data).float()
    # src_batch = torch.from_numpy(src_data).unsqueeze(0).float()

    dst_data = np.array([[[4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7]],
                         [[14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17]]])
    dst_batch = torch.from_numpy(dst_data).float()

    model.train()
    for i in range(500):
        optimizer.zero_grad()

        data, target = src_batch.to(device), dst_batch.to(device)
        output = model(data)

        loss = loss_fun(output, target)
        loss.backward()
        print(loss.item())
        optimizer.step()

        if i % 50 == 0:
            print(output)


# test_tpn()
