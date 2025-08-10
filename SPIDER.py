#!/usr/bin/env python3
"""
SPIDER.py - Scalable Probabilistic Inference for Differential Earthquake Relocation

This script now serves as a thin wrapper around the modular `spider` package.
It retains the `EikoNet` class, `setup_nn_model`, and the main execution entrypoint.
"""

import sys
import json
import torch

from spider.core import locate_all, prepare_input_dfs


class EikoNet(torch.nn.Module):

    def __init__(self, scale, vs=3.3, vp=6.0):
        super(EikoNet, self).__init__()
        self.scale = scale
        self.vs = vs
        self.vp = vp
        self.activation = torch.nn.ELU()
        self.n_hidden = 128

        self.linear1 = torch.nn.Linear(4, self.n_hidden)
        self.linear2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear3 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear4 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear5 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear6 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear7 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear8 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear9 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear10 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear11 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear_out = torch.nn.Linear(self.n_hidden, 1)

    def T0(self, x):
        scalar = torch.where(x[:,6] >= 0.5, self.vs, self.vp).unsqueeze(1)
        return torch.sqrt(((x[:,0:3]-x[:,3:6])**2).sum(dim=1)).unsqueeze(dim=1) / scalar

    def T1(self, x):
        r = torch.sqrt(((x[:,0:2] - x[:,3:5])**2).sum(dim=1)).unsqueeze(1)
        x = torch.cat((r, x[:,2,None], x[:,5:]), dim=1)
        x[:,:3] /= self.scale

        x = self.linear1(x)
        x = self.activation(x)

        x0 = x
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x) + x0

        x0 = x
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        x = self.activation(x) + x0

        x0 = x
        x = self.linear6(x)
        x = self.activation(x)
        x = self.linear7(x)
        x = self.activation(x) + x0

        x0 = x
        x = self.linear8(x)
        x = self.activation(x)
        x = self.linear9(x)
        x = self.activation(x) + x0

        x0 = x
        x = self.linear10(x)
        x = self.activation(x)
        x = self.linear11(x)
        x = self.activation(x) + x0

        x = self.linear_out(x)
        x = torch.abs(x)

        return x

    def forward(self, x):
        return self.T0(x) * self.T1(x)

    def EikonalPDE(self, x):
        x.requires_grad_()
        T = self.forward(x)
        dT_dx = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        s_rec = (dT_dx[:,3:6]**2).sum(dim=1).sqrt()
        return s_rec


def setup_nn_model(params, device):
    """
    Load and setup the neural network model.

    Args:
        params: Parameter dictionary containing 'model_file'
        device: Target device for the model

    Returns:
        Loaded and configured model
    """
    model_file = params["model_file"]
    model = torch.load(model_file, weights_only=False, map_location=torch.device('cpu'))
    model.eval()
    model.to(device)
    model = model.float()
    return model


if __name__== "__main__":

    if len(sys.argv) > 1:
        pass
    else:
        print("python SPIDER.py [pfile]")
        exit()

    spider_pfile = sys.argv[1]
    with open(spider_pfile) as f:
        params = json.load(f)

    device = params["devices"][0]

    model = setup_nn_model(params, device)

    print("Preparing input dataset")
    stations, dtimes, origins = prepare_input_dfs(params)

    print("Dataset has %d events with %d dtimes" % (origins.shape[0], dtimes.shape[0]))

    print("Running SPIDER")
    locate_all(params, origins, dtimes, model, device)