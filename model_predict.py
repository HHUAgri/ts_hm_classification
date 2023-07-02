# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import time
from collections import OrderedDict
import torch
from torch import nn
import torchvision

from data import CropDataset
from model import HMTSNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def predict_tpn():

    ### model
    model_path = r'E:\develop_project\python\ts_hm_classification\model_save\loss-tpn4\model_best.pt'

    the_model = HMTSNet(in_planes=3, num_classes=[4, 12]).to(device)
    the_model.load_state_dict(torch.load(model_path))
    the_model.eval()
    # print(the_model)

    # extract layer1 and layer3, giving as names `feat1` and feat2`
    sub_models = IntermediateLayerGetter(the_model, {'tpn': 'fc_loc'})
    # out = sub_models(torch.rand(1, 3, 224, 224))
    # print([(k, v.shape) for k, v in out.items()])

    ### dataset
    dataset = 'dijon_s1_mean'
    train_dataset = CropDataset(dataset, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=4, drop_last=True)

    ### prediction
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # pack ground truth labels
            inputs, labels = inputs.to(device), labels.to(device)

            # prediction
            output = the_model(inputs)

            # pred_np = output.cpu().detach().numpy()
            a = 1

    pass


def main():

    predict_tpn()

    pass


if __name__ == "__main__":
    main()
