from xml.dom.expatbuilder import parseFragment
import torch
import torch.nn as nn
from torch.nn import init
import functools
# from torch.autograd import Variable
import numpy as np


###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

#定义生成器
def define_G(input_nc, output_nc, norm='batch', gpu_ids=[], use_parallel=True,
             learn_residual=False):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    netG = Generator(input_nc, output_nc, norm_layer=norm_layer, n_blocks=3,
                            gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)

    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

#定义判别器
def define_D(input_nc, ndf, norm='batch', use_sigmoid=False, gpu_ids=[], use_parallel=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    netD = Discriminator(input_nc, ndf=64, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                gpu_ids=gpu_ids, use_parallel=use_parallel)

    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def solve_padding(input_nc, ngf, kernel_size, stride, dilation):
    new_kernel_size = dilation*(kernel_size-1)+1
    padding = ((ngf - 1)*stride + new_kernel_size - input_nc)//2
    return padding


##############################################################################
# Generator & Discriminator
##############################################################################

# 生成器
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d,
                n_blocks=3, gpu_ids=[], use_parallel=True, learn_residual=False):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 16, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(16),
            nn.ReLU(True)
        ]

        # 下采样
        model += [
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        ]

        # 残差网络
        for i in range(n_blocks):
            model += [
                ResBlock(norm_layer=norm_layer, use_bias=use_bias)
            ]

        model += [
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(16),
            nn.ReLU(True),
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(16, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        #if self.learn_residual:
            # output = input + output
        #    fin_output = torch.clamp(input + output, min=-1, max=1)
        return {'input':input, 'output':output}#'finall_output':fin_output, 


# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=32, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(input_nc, 8, kernel_size=7, stride=3, padding=3),
            nn.LeakyReLU(0.2, True)
        ]


        sequence += [
            nn.Conv2d(8, 16, kernel_size=7, stride=3, padding=3, bias=use_bias),
            norm_layer(16),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(16, 32, kernel_size=7, stride=3, padding=3, bias=use_bias),
            norm_layer(32),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 64, kernel_size=7, stride=3, padding=3, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [CBAMBlock(channel=64, reduction=8, kernel_size=7)]#注意力机制模块

        sequence += [nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

        #print_network(self.model)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


##############################################################################
# Blocks
##############################################################################

# 定义残差层 + 空洞卷积
class ResBlock(nn.Module):
	def __init__(self, norm_layer, use_bias):
		super(ResBlock, self).__init__()

		blocks = [
                nn.ReflectionPad2d(2),
                nn.Conv2d(64, 64, kernel_size=5, bias=use_bias),
				norm_layer(64),
				nn.ReLU(True)
            ] + [
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 64, kernel_size=7, bias=use_bias),
				norm_layer(64)
			] + [
				nn.Dropout(0.5)
            ] + [
                nn.ReflectionPad2d(2),
                nn.Conv2d(64, 64, kernel_size=5, bias=use_bias),
				norm_layer(64)
			]

		dilated_conv1 = [
                nn.ReflectionPad2d(4),
                nn.Conv2d(64, 64, kernel_size=5, bias=use_bias, dilation=2),
				norm_layer(64),
				nn.ReLU(True)
            ] + [
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 64, kernel_size=7, bias=use_bias),
				norm_layer(64)
			] + [
				nn.Dropout(0.5)
            ] + [
                nn.ReflectionPad2d(4),
                nn.Conv2d(64, 64, kernel_size=5, bias=use_bias, dilation=2),
				norm_layer(64)
			]

		dilated_conv2 = [
                nn.ReflectionPad2d(6),
                nn.Conv2d(64, 64, kernel_size=5, bias=use_bias, dilation=3),
				norm_layer(64),
				nn.ReLU(True)
            ] + [
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 64, kernel_size=7, bias=use_bias),
				norm_layer(64)
			] + [
				nn.Dropout(0.5)
            ] + [
                nn.ReflectionPad2d(6),
                nn.Conv2d(64, 64, kernel_size=5, bias=use_bias, dilation=3),
				norm_layer(64)
			]

		self.conv_block = nn.Sequential(*blocks)
		self.dilated_conv1 = nn.Sequential(*dilated_conv1)
		self.dilated_conv2 = nn.Sequential(*dilated_conv2)

	def forward(self, x):
		conv=self.conv_block(x)
		d_conv1=self.dilated_conv1(x)
		d_conv2=self.dilated_conv2(x)
		out = x + conv+d_conv1+d_conv2
		return out

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=8):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):

    def __init__(self, channel=64,reduction=8,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual
