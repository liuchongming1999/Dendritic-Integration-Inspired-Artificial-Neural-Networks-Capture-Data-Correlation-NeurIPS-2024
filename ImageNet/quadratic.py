import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module

class quadratic(Module):
    __constants__ = ['in_features','out_features']

    def __init__(self, in_features, out_features):
        super(quadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quadratic = Parameter(torch.Tensor(out_features,in_features*in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.quadratic, std = 0)

    def forward(self, input):
        return  F.linear(torch.bmm(input.reshape(-1,self.in_features).unsqueeze(-1), input.reshape(-1,self.in_features).unsqueeze(-2)).reshape(input.size(0),input.size(1),input.size(2),self.in_features**2), self.quadratic)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )