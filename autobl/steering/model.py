import logging
from typing import Optional, Tuple

import botorch
import torch
import numpy as np
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
import rff

from autobl.util import *
from autobl.steering.components import *


class ProjectedSpaceSingleTaskGP(botorch.models.SingleTaskGP):

    def __init__(self, *args, projection_function, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_func = projection_function

    def set_projection_func(self, f):
        self.project_func = f

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        x = self.project_func(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    

class SLDASNet(torch.nn.Module):

    def __init__(self, n_neighbors=3, sigmoid_on_output=False, *args, **kwargs):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.sigmoid_on_output = sigmoid_on_output
        
    def create_mlp(self, dim_in, dim_out, dim_hidden: Optional[Tuple[int, ...] | int] = None):
        if dim_hidden is None:
            return torch.nn.Linear(dim_in, dim_out)
        else:
            if not hasattr(dim_hidden, '__len__'):
                dim_hidden = (dim_hidden,)
            net = [torch.nn.Linear(dim_in, dim_hidden[0]), torch.nn.ReLU()]
            for i in range(len(dim_hidden) - 1):
                net.append(torch.nn.Linear(dim_hidden[i], dim_hidden[i + 1]))
                net.append(torch.nn.ReLU())
            net.append(torch.nn.Linear(dim_hidden[-1], dim_out))
            return torch.nn.Sequential(*net)
        
    def generate_features(self, x, x_measured):
        feat_knn_dists = self.generate_k_nearest_neighbor_distance_feature(x, x_measured)
        x = x.reshape(-1, 1)
        feat = torch.cat([x, feat_knn_dists], dim=1)
        return feat
    
    def generate_k_nearest_neighbor_distance_feature(self, x, x_measured):
        dists = torch.abs(x.reshape(-1, 1) - x_measured)
        dists = torch.sort(dists, dim=1).values
        # Fill nans
        dists = torch.nan_to_num(dists, 1.0)
        if dists.shape[1] < self.n_neighbors:
            dists = torch.nn.functional.pad(dists, (0, self.n_neighbors - dists.shape[1]), mode="replicate")
        return dists[:, :self.n_neighbors]
        
    
class ConvMLPModel(SLDASNet):
    
    def __init__(self, dim_recon_spec, dim_spec_encoded=256, dim_feat_encoded=256, add_pooling=False, 
                 dim_hidden_feature=128, dim_hidden_final=128,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
                
        # 1D conv net for reconstructed spectrum
        pool_params = {'pool_kernel_size': 3, 'pool_stride': 2} if add_pooling else {}
        d = dim_recon_spec
        self.conv_net = [
            Conv1DReLU(kernel_size=9, in_channels=1, out_channels=80, stride=2, padding='valid', dropout=0.1, **pool_params),
            Conv1DReLU(kernel_size=9, in_channels=80, out_channels=80, stride=2, padding='valid', dropout=0.2, **pool_params),
            Conv1DReLU(kernel_size=9, in_channels=80, out_channels=80, stride=2, padding='valid', dropout=0.3, **pool_params),
        ]
        for layer in self.conv_net:
            d = layer.calculate_output_spatial_dim(d)
        self.conv_net = nn.Sequential(*self.conv_net)
        self.flatten = torch.nn.Flatten()
        self.conv_linear = torch.nn.Linear(d * 80, dim_spec_encoded)
        
        # MLP for position and other features
        self.encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=1 + self.n_neighbors, 
                                                    encoded_size=dim_feat_encoded // 2)
        self.feat_linear = torch.nn.Linear(dim_feat_encoded, dim_feat_encoded)
        self.feat_linear = self.create_mlp(dim_in=dim_feat_encoded, dim_out=dim_feat_encoded,
                                           dim_hidden=dim_hidden_feature)
        
        self.final_linear = self.create_mlp(dim_in=dim_feat_encoded + dim_spec_encoded, dim_out=1, 
                                            dim_hidden=dim_hidden_final)
        
    def forward(self, x, x_measured, y_interp):
        feat = self.generate_features(x, x_measured)
        feat = self.encoding(feat)
        x = self.feat_linear(feat)
        
        if len(y_interp.shape) == 2:
            y_interp = y_interp.reshape(-1, 1, y_interp.shape[1])
        y = self.conv_net(y_interp)
        y = self.flatten(y)
        y = self.conv_linear(y)
        
        erd = self.final_linear(torch.cat([x, y], dim=1))
        if self.sigmoid_on_output:
            erd = torch.nn.functional.sigmoid(erd)
        return erd
    

if __name__ == '__main__':
    model = ConvMLPModel(1000)
    y = model.forward(torch.rand(2,), torch.rand(2, 5), torch.rand(2, 1000))
    print(y)
