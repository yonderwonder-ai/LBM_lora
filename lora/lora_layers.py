import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer

        self.lora_down = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, linear_layer.out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.linear(x) + (self.lora_up(self.lora_down(x)) * self.scaling)


class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer, rank, alpha):
        super().__init__()
        self.conv = conv_layer

        self.lora_down = nn.Conv2d(
            conv_layer.in_channels,
            rank,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=False,
        )
        self.lora_up = nn.Conv2d(
            rank, conv_layer.out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.conv(x) + (self.lora_up(self.lora_down(x)) * self.scaling)
