# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import random
import string
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

import torch
import torch.nn.functional as F

from data import CropDataset
from data import get_hierarchy_label
from model import HMTSNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# defines two global scope variables to store our gradients and activations
gradients = None
activations = None


def show_grad_cam(inputs, labels, heatmap):

    if True in np.isnan(heatmap):
        return None

    # source data
    x_array = np.arange(0, 40, 1)
    y_array_vv = np.array(inputs[0][0])
    y_array_vh = np.array(inputs[0][1])
    label = labels[0]
    heatmap = np.array(heatmap[0])

    # spline data
    vv_spline = make_interp_spline(x_array, y_array_vv)
    vh_spline = make_interp_spline(x_array, y_array_vh)
    heatmap_spline = make_interp_spline(x_array, heatmap)

    x_array_640 = np.linspace(x_array.min(), x_array.max(), 640)
    y_array_vv_640 = vv_spline(x_array_640)
    y_array_vh_640 = vh_spline(x_array_640)
    heatmap_640 = heatmap_spline(x_array_640)

    # draw
    extent = [x_array_640[0] - (x_array_640[1] - x_array_640[0]) / 2.,
              x_array_640[-1] + (x_array_640[1] - x_array_640[0]) / 2., 0, 1]

    plt.rcParams["figure.figsize"] = 8, 4
    plt.rcParams['font.size'] = 12
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Tahoma']
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.patch.set_alpha(0.5)
    ax1.imshow(heatmap_640[np.newaxis, :], cmap="Greens", aspect="auto", alpha=1, extent=extent)
    ax1.set_yticks([])
    ax1.set_xlim(extent[0], extent[1])

    # ax2.plot(x_array_640, heatmap_640)
    ax2.plot(x_array_640, y_array_vv_640, alpha=1, label='VV', color='red')
    ax2.plot(x_array_640, y_array_vh_640, alpha=1, label='VH', color='yellow')

    plt.tight_layout()

    # save and show
    random_str = ''.join(random.sample(string.ascii_letters + string.digits, 6))
    save_path = './640/{}_{}.png'.format(label+1, random_str)
    fig.savefig(save_path, transparent=True)
    # plt.show()

    pass


def grad_cam_ts_classification():

    ### model
    model_path = r'E:\develop_project\python\ts_hm_classification\model_save\loss-without\model_best.pt'

    the_model = HMTSNet(in_planes=3, num_classes=[4, 12]).to(device)
    the_model.load_state_dict(torch.load(model_path))
    the_model.eval()
    # print(the_model)

    ### hook functions
    def backward_hook(module, grad_input, grad_output):
        global gradients  # refers to the variable in the global scope
        print('Backward hook running...')
        gradients = grad_output
        # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
        print(f'Gradients size: {gradients[0].size()}')
        # We need the 0 index because the tensor containing the gradients comes
        # inside a one element tuple.

    def forward_hook(module, args, output):
        global activations  # refers to the variable in the global scope
        print('Forward hook running...')
        activations = output
        # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
        print(f'Activations size: {activations.size()}')

    target_layers = the_model.cb_l2
    backward_hook = target_layers.register_full_backward_hook(backward_hook)
    forward_hook = target_layers.register_forward_hook(forward_hook)

    ### dataset
    dataset = 'dijon_s1_mean'
    train_dataset = CropDataset(dataset, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=4, drop_last=True)

    ### grad cam
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # pack ground truth labels
        l1_target, l2_target = get_hierarchy_label(labels)
        labels = l2_target - 4
        # labels = l1_target
        inputs_tensor = inputs.to(device)
        labels_tensor = labels.to(device)

        # forward.
        # y_sig_l1, y_sig_l2, y_11, y_12
        outputs_tensor = the_model(inputs_tensor)[3]
        outputs_tensor = F.softmax(outputs_tensor, dim=1)
        outputs_prob, outputs_label = torch.max(outputs_tensor, dim=1)
        outputs_prob = outputs_prob.cpu().data.numpy()
        outputs_label = outputs_label.cpu().data.numpy()

        # backward.
        target_label = outputs_label[0]
        target_prob = outputs_prob[0]
        if (target_prob < 0.65) or (labels[0] != outputs_label[0]):
            continue

        the_model.zero_grad()
        outputs_tensor[0, target_label].backward()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2])
        # weight the channels by corresponding gradients
        for i in range(activations.size()[1]):
            activations[:, i, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        # relu on top of the heatmap
        heatmap = F.relu(heatmap)
        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.cpu().data.numpy()

        # show
        show_grad_cam(inputs, labels, heatmap)

    # for batch
    pass


def main():

    grad_cam_ts_classification()


if __name__ == "__main__":
    main()
