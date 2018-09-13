# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn

class RouteLayer(nn.Module):
    """
    this layer is just a placeholder,
    do not use it in forwarding...
    """
    def __init__(self,route_layers):
        super(RouteLayer, self).__init__()
        self.route_layers=route_layers

    @DeprecationWarning
    def forward(self, x):
        raise ValueError('the function forward in this module has been deprecated')