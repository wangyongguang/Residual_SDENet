import torch
import torch.nn as nn
import torch.nn.init as init
import math

__all__ = ['SDENet_24Layer_PlainBlocks']

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            #print(m.bias)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride,
                     bias=False)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride,
            padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class PlainBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None):
        super(PlainBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = ConcatConv2d(inplanes, planes, 3, 1, 1)
        self.norm1 = norm(planes)

        self.conv2 = ConcatConv2d(planes, planes, 3, 1, 1)
        self.norm2 = norm(planes)

    def forward(self, t, x):

        out = self.conv1(t, x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(t, out)
        out = self.norm2(out)
        out = self.relu(out)

        return out

class Drift(nn.Module):
    def __init__(self, dim=64):
        super(Drift, self).__init__()

        self.PlainBlock1 = PlainBlock(dim, dim)  # 3x3:(64,64); 3x3:(64,64)
        self.PlainBlock2 = PlainBlock(dim, dim)  # 3x3:(64,64); 3x3:(64,64)

        self.PlainBlock3 = PlainBlock(dim, dim*2)  # 3x3:(64,128);3x3:(128,128)
        self.PlainBlock4 = PlainBlock(dim*2, dim)  # 3x3:(128,64); 3x3:(64,64)

        self.PlainBlock5 = PlainBlock(dim, dim * 4)  # 3x3:(64,256);3x3:(256,256)
        self.PlainBlock6 = PlainBlock(dim * 4, dim)  # 3x3:(256,64); 3x3:(64,64)

        self.PlainBlock7 = PlainBlock(dim, dim * 8)  # 3x3:(64,512);3x3:(512,512)
        self.PlainBlock8 = PlainBlock(dim * 8, dim)  # 3x3:(512,64); 3x3:(64,64)

        self.PlainBlock9 = PlainBlock(dim, dim * 16)  # 3x3:(64,1024);3x3:(1024,1024)
        self.PlainBlock10 = PlainBlock(dim * 16, dim)  # 3x3:(1024,64); 3x3:(64,64)

    def forward(self, t, x):

        out = self.PlainBlock1(t, x)
        out = self.PlainBlock2(t, out)
        out = self.PlainBlock3(t, out)
        out = self.PlainBlock4(t, out)
        out = self.PlainBlock5(t, out)
        out = self.PlainBlock6(t, out)
        out = self.PlainBlock7(t, out)
        out = self.PlainBlock8(t, out)
        out = self.PlainBlock9(t, out)
        out = self.PlainBlock10(t, out)

        return out

class Diffusion(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Diffusion, self).__init__()

        self.PlainBlock1 = PlainBlock(dim_in,dim_out) #3x3:(64,64); 3x3:(64,64)
        self.PlainBlock2 = PlainBlock(dim_out,dim_in) #3x3:(64,64); 3x3:(64,64)

        self.PlainBlock3 = PlainBlock(dim_in, dim_out*2)  # 3x3:(64,128);3x3:(128,128)
        self.PlainBlock4 = PlainBlock(dim_out*2, dim_in)  # 3x3:(128,64); 3x3:(64,64)

        self.PlainBlock5 = PlainBlock(dim_in, dim_out * 4)  # 3x3:(64,256);3x3:(256,256)
        self.PlainBlock6 = PlainBlock(dim_out * 4, dim_in)  # 3x3:(256,64); 3x3:(64,64)

        self.PlainBlock7 = PlainBlock(dim_in, dim_out * 8)  # 3x3:(64,512);3x3:(512,512)
        self.PlainBlock8 = PlainBlock(dim_out * 8, dim_in)  # 3x3:(512,64); 3x3:(64,64)

        self.PlainBlock9 = PlainBlock(dim_in, dim_out * 16)  # 3x3:(64,1024);3x3:(1024,1024)
        self.PlainBlock10 = PlainBlock(dim_out * 16, dim_in)  # 3x3:(1024,64); 3x3:(64,64)

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                Flatten(),
                                nn.Linear(dim_out, 1),
                                nn.Sigmoid())

    def forward(self, t, x):

        out = self.PlainBlock1(t, x)
        out = self.PlainBlock2(t, out)
        out = self.PlainBlock3(t, out)
        out = self.PlainBlock4(t, out)
        out = self.PlainBlock5(t, out)
        out = self.PlainBlock6(t, out)
        out = self.PlainBlock7(t, out)
        out = self.PlainBlock8(t, out)
        out = self.PlainBlock9(t, out)
        out = self.PlainBlock10(t, out)

        out = self.fc(out)
        return out

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class SDENet_24Layer_PlainBlocks(nn.Module):
    def __init__(self, layer_depth, num_classes=10, dim = 64):
        super(SDENet_24Layer_PlainBlocks, self).__init__()
        self.layer_depth = layer_depth
        self.dim = dim
        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
        )

        self.drift = Drift(64)
        self.diffusion = Diffusion(64,64)

        self.fc_layers = nn.Sequential(
            norm(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(64,10)
        )
        self.deltat = 6./self.layer_depth
        self.apply(init_params)
        self.sigma = 50

    def forward(self, x, training_diffusion=False):
        out = self.downsampling_layers(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma*self.diffusion(t, out)
            diffusion_term = torch.unsqueeze(diffusion_term, 2)
            diffusion_term = torch.unsqueeze(diffusion_term, 3)
            for i in range(self.layer_depth):
                t = 6*(float(i))/self.layer_depth
                out = out + self.drift(t, out)*self.deltat + \
                      diffusion_term*math.sqrt(self.deltat)*torch.randn_like(out).to(x)
            final_out = self.fc_layers(out)
        else:
            t = 0
            final_out = self.diffusion(t, out.detach())
        return final_out

def test():
    model = SDENet_24Layer_PlainBlocks()
    return model  
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)

if __name__ == '__main__':
    model = test()
    num_params = count_parameters(model)
    print(num_params)