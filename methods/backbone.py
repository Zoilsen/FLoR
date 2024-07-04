# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# --- gaussian initialize ---
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = 10 * cos_dist
        return scores

# --- flatten tensor ---
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# --- Linear module ---
class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_fw, self).__init__(in_features, out_features, bias=bias)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out

# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        return out

# --- softplus module ---
def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)

# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = False
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment: # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3)
            self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.5)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature-wise transformation
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
            out = gamma*out + beta
        return out

# --- BatchNorm2d ---
class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
        return out

# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
        return out

# --- Simple Conv Block ---
class ConvBlock(nn.Module):
    maml = False
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN = FeatureWiseTransformation2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self,x):
        out = self.trunk(x)
        return out


#### use it during training ####
# --- My Norm ---
class MyNorm(nn.Module):
    def __init__(self, outdim, map_size):
        super(MyNorm, self).__init__()
        self.outdim = outdim
        self.map_size = int(map_size)
        self.IN = nn.InstanceNorm2d(outdim, affine=True)
        self.BN = nn.BatchNorm2d(outdim, affine=True)

    def forward(self, x, params):
        # x: [b, c, h, w]
        # epoch: train: from 0 to max_epoch-1; test: -1
        assert self.map_size == x.shape[-1]
        b, c, h, w = x.shape

        ratio_a = params.beta_a
        ratio_b = params.beta_b

        if self.training:
            import numpy as np
            w = np.random.beta(ratio_a, ratio_b, size=(b,))
            w = torch.tensor(w).cuda().float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            w = ratio_a / (ratio_a + ratio_b)


        x = w * self.IN(x) + (1 - w) * self.BN(x)

        return x

'''
#### use it during finetuning ####
# --- My Norm ---
class MyNorm(nn.Module):
    def __init__(self, outdim, map_size):
        super(MyNorm, self).__init__()
        self.outdim = outdim
        self.map_size = int(map_size)
        self.IN = nn.InstanceNorm2d(outdim, affine=True)
        self.BN = nn.BatchNorm2d(outdim, affine=True)
        
        #### use it during finetuning ###
        def reverse_sigmoid(x):
            import numpy as np
            return np.log(x / (1.0 - x))
        self.IN_w = nn.Parameter(torch.tensor(reverse_sigmoid(0.01/(0.01+0.01))))
        #################################


    def forward(self, x, params):
        # x: [b, c, h, w]
        # epoch: train: from 0 to max_epoch-1; test: -1
        assert self.map_size == x.shape[-1]
        b, c, h, w = x.shape

        ratio_a = params.beta_a
        ratio_b = params.beta_b

        if self.training:
            w = torch.sigmoid(self.IN_w) # finetune self.IN_w
        else:
            w = torch.sigmoid(self.IN_w)

        x = w * self.IN(x) + (1 - w) * self.BN(x)

        return x
'''


# --- Simple ResNet Block ---
class SimpleBlock(nn.Module):
    maml = False
    def __init__(self, indim, outdim, half_res, map_size, leaky=False, IN=False):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = FeatureWiseTransformation2d_fw(outdim) # feature-wise transformation at the end of each residual block
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            if IN:
                self.BN1 = MyNorm(outdim, (map_size / 2) if half_res else map_size)
            else:
                self.BN1 = nn.BatchNorm2d(outdim)
 
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1,bias=False)
            if IN:
                self.BN2 = MyNorm(outdim, (map_size / 2) if half_res else map_size)
            else:
                self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = FeatureWiseTransformation2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                if IN:
                    self.BNshortcut = MyNorm(outdim, (map_size / 2) if half_res else map_size)
                else:
                    self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x, params):
        out = self.C1(x)
        out = self.BN1(out, params)
        out = self.relu1(out)
       
        out = self.C2(out)       
        out = self.BN2(out, params)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x), params)
        out = out + short_out
        out = self.relu2(out)
        return out

# --- ConvNet module ---
class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return out

# --- ConvNetNopool module ---
class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool,self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,19,19]

    def forward(self,x):
        out = self.trunk(x)
        return out

# --- ResNet module ---
class ResNet(nn.Module):
    maml = False
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten=True, leakyrelu=False, IN_layers=None):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        self.grads = []
        self.fmaps = []
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if IN_layers is not None:
                if IN_layers[0] == True:
                    bn1 = MyNorm(64, map_size=112)
                else:
                    bn1 = nn.BatchNorm2d(64)
            else:
                bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        half_res_all = [False, True, True, True]
        
        map_size = 56
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                #half_res = (i>=1) and (j==0)
                assert list_of_num_layers[i] == 1
                half_res = half_res_all[i]

                if IN_layers == None:
                    IN = False
                else:
                    if IN_layers[i+1] == True:
                        IN = True
                    else:
                        IN = False
                
                if i > 0:
                    if half_res_all[i-1] == True:
                        map_size = map_size / 2

                B = block(indim, list_of_out_dims[i], half_res, map_size=map_size, leaky=leakyrelu, IN=IN) 
                trunk.append(B)
                
                indim = list_of_out_dims[i]

        if flatten:
            #avgpool = nn.AvgPool2d(7)
            avgpool = torch.nn.AdaptiveAvgPool2d(1)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)


    def forward(self,x, params):
        #out = self.trunk(x)
        for m in self.trunk:
            if 'SimpleBlock' in str(m) or 'MyNorm' in str(m):
                x = m(x, params)
            else:
                x = m(x)
        out = x
        #exit()

        return out

# --- Conv networks ---
def Conv4():
        return ConvNet(4)
def Conv6():
        return ConvNet(6)
def Conv4NP():
        return ConvNetNopool(4)
def Conv6NP():
        return ConvNetNopool(6)

# --- ResNet networks ---
def ResNet10(flatten=False, leakyrelu=False):
        return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, leakyrelu, IN_layers=[True, True, True, True, True])
def ResNet18(flatten=True, leakyrelu=False):
        return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten, leakyrelu)
def ResNet34(flatten=True, leakyrelu=False):
        return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten, leakyrelu)


model_dict = dict(Conv4 = Conv4,
                  Conv6 = Conv6,
                  ResNet10 = ResNet10,
                  ResNet18 = ResNet18,
                  ResNet34 = ResNet34)

