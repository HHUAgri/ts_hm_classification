# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, num_layers, dropout=0):
        '''

        '''
        super(LSTMClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size  # number of hidden states
        self.num_layers = num_layers    # number of LSTM layers (stacked)

        # according to pytorch docs LSTM output is
        # (batch_size, time_step, num_directions * hidden_size)
        # when considering batch_first = True
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.hidden_cell = None

    def forward(self, x):
        batch_size = x.shape[0]

        if self.hidden_cell is None:
            hidden_state = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device).requires_grad_()
            cell_state = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device).requires_grad_()
            self.hidden_cell = (hidden_state.data, cell_state.data)

        x, self.hidden_cell = self.lstm(x, self.hidden_cell)

        self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        x = x[:, -1].view(-1, self.hidden_size)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_lstmclassifier():
    # batch_size x time_step x channels
    x = torch.randn(9, 60, 4).to(device)
    lstm_net = LSTMClassifier(4, 13, 64, 2).to(device)
    print(lstm_net)

    y = lstm_net(x)
    print(str(check_parameters(lstm_net)) + ' Mb')
    print(y[1].shape)


if __name__ == "__main__":
    test_lstmclassifier()
