import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DReLU(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1, padding='valid', dropout=0,
                 pool_kernel_size=0, pool_stride=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.pool = None
        if pool_kernel_size > 0 and pool_stride > 0:
            self.pool = torch.nn.AvgPool1d(pool_kernel_size, pool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

    def calculate_output_spatial_dim(self, in_size):
        with torch.no_grad():
            x = torch.rand([1, self.conv.in_channels, in_size])
            x = self.forward(x)
        return x.shape[-1]