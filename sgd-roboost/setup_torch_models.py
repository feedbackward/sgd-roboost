'''Model management for PyTorch application.'''

## External modules.
import torch.nn as nn
import torch.nn.functional as F


###############################################################################


## Contents:
# Hard-coded model parameters.
# Model class definitions.
# Dicts for organizing models and "todo" parameters.
# The main parser function.


## Hard-coded model parameters.
_int_num = 10 # number of units in intermediate layers.


## Model class definitions.

class Logistic_Reg(nn.Module):
    
    def __init__(self, bias=True, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features=kwargs["num_features"],
                                out_features=kwargs["num_classes"],
                                bias=bias)
        return None
    
    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)


class FF_L1(nn.Module):
    
    def __init__(self, bias=True, **kwargs):
        super().__init__()
        self.linear_0 = nn.Linear(in_features=kwargs["num_features"],
                                  out_features=_int_num,
                                  bias=bias)
        self.linear_1 = nn.Linear(in_features=_int_num,
                                  out_features=kwargs["num_classes"],
                                  bias=bias)
        return None
    
    def forward(self, x):
        x = F.relu(self.linear_0(x))
        x = F.log_softmax(self.linear_1(x), dim=1)
        return x


class FF_L2(nn.Module):
    
    def __init__(self, bias=True, **kwargs):
        super().__init__()
        self.linear_0 = nn.Linear(in_features=kwargs["num_features"],
                                  out_features=_int_num,
                                  bias=bias)
        self.linear_1 = nn.Linear(in_features=_int_num,
                                  out_features=_int_num,
                                  bias=bias)
        self.linear_2 = nn.Linear(in_features=_int_num,
                                  out_features=kwargs["num_classes"],
                                  bias=bias)
        return None
    
    def forward(self, x):
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.log_softmax(self.linear_2(x), dim=1)
        return x

    

class FF_L3(nn.Module):
    
    def __init__(self, bias=True, **kwargs):
        super().__init__()
        self.linear_0 = nn.Linear(in_features=kwargs["num_features"],
                                  out_features=_int_num,
                                  bias=bias)
        self.linear_1 = nn.Linear(in_features=_int_num,
                                  out_features=_int_num,
                                  bias=bias)
        self.linear_2 = nn.Linear(in_features=_int_num,
                                  out_features=_int_num,
                                  bias=bias)
        self.linear_3 = nn.Linear(in_features=_int_num,
                                  out_features=kwargs["num_classes"],
                                  bias=bias)
        return None
    
    def forward(self, x):
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.log_softmax(self.linear_3(x), dim=1)
        return x


class Linear_Reg(nn.Module):
    
    def __init__(self, bias=False, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features=kwargs["num_features"],
                                out_features=kwargs["num_classes"],
                                bias=bias)
        return None
    
    def forward(self, x):
        return self.linear(x)


class CNN_3L(nn.Module):

    def __init__(self, bias=True, **kwargs):
        super().__init__()
        self.pix_h = kwargs["pix_h"]
        self.pix_w = kwargs["pix_w"]
        self.in_channels = kwargs["channels"]
        self.conv2d_1 = nn.Conv2d(in_channels=self.in_channels,
                                  out_channels=16,
                                  kernel_size=3, stride=2, padding=1,
                                  bias=bias)
        self.conv2d_2 = nn.Conv2d(in_channels=16,
                                  out_channels=16,
                                  kernel_size=3, stride=2, padding=1,
                                  bias=bias)
        self.conv2d_3 = nn.Conv2d(in_channels=16,
                                  out_channels=kwargs["num_classes"],
                                  kernel_size=3, stride=2, padding=1,
                                  bias=bias)
        
    def forward(self, x):
        x = x.view(-1, self.in_channels, self.pix_h, self.pix_w)
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        x = F.avg_pool2d(x, 4)
        return x.view(-1, x.size(1))


## Dicts for organizing models and "todo" parameters.

models_dict = {"logistic": Logistic_Reg,
               "FF_L1": FF_L1,
               "FF_L2": FF_L2,
               "FF_L3": FF_L3,
               "linreg": Linear_Reg,
               "CNN_3L": CNN_3L}
paras_todo_dict = {"logistic": ["linear.weight", "linear.bias"],
                   "FF_L1": ["linear_0.weight", "linear_0.bias",
                             "linear_1.weight", "linear_1.bias"],
                   "FF_L2": ["linear_0.weight", "linear_0.bias",
                             "linear_1.weight", "linear_1.bias",
                             "linear_2.weight", "linear_2.bias"],
                   "FF_L3": ["linear_0.weight", "linear_0.bias",
                             "linear_1.weight", "linear_1.bias",
                             "linear_2.weight", "linear_2.bias",
                             "linear_3.weight", "linear_3.bias"],
                   "linreg": ["linear.weight"],
                   "CNN_3L": ["conv2d_1.weight",
                              "conv2d_2.weight",
                              "conv2d_3.weight",
                              "conv2d_1.bias",
                              "conv2d_2.bias",
                              "conv2d_3.bias"]}


## The main parser function.

def get_model(model_class):
    return models_dict[model_class], paras_todo_dict[model_class]


###############################################################################
