# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn


class ShortcutLayer(nn.Module):
    def __init__(self,from_index,act):
        super(ShortcutLayer, self).__init__()
        self.from_index=from_index
        self.act=act

    def forward(self, x):
        raise ValueError('the function forward in this module has been deprecated')
