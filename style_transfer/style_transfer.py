from __future__ import print_function
from cmath import exp

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import copy
from .loss import ContentLoss, GramMatrix, StyleLoss


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(opt, cnn, style_dataset, content_img, style_weight=100, content_weight=1, content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    use_cuda = torch.cuda.is_available() and opt.cuda

    content_losses = []#内容损失
    style_losses = []#风格损失

    model = nn.Sequential()
    gram = GramMatrix()

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()
        content_img = content_img.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)
            if name in content_layers:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight, content_img)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                bright_feature_gram_list = []
                BG_feature_gram_list = []
                blooming_feature_gram_list = []
                for j, style_img in enumerate (style_dataset):
                    if use_cuda:
                        style_img = style_img.cuda()
                    target_feature = model(style_img).clone()

                    bright_mask, blooming_mask, background_mask = creat_mask(style_img, target_feature.shape[2], target_feature.shape[1])
                    bright_feature = target_feature.masked_fill(~bright_mask, 0)#掩膜机制（0，（0-1））
                    blooming_feature = target_feature.masked_fill(~blooming_mask, 0)
                    BG_feature = target_feature.masked_fill(~background_mask, 0)
                    bright_feature_gram_list.append(gram(bright_feature))
                    blooming_feature_gram_list.append(gram(blooming_feature))
                    BG_feature_gram_list.append(gram(BG_feature))
                
                bright_feature_gram = bright_feature_gram_list[0]
                blooming_feature_gram = blooming_feature_gram_list[0]
                BG_feature_gram = BG_feature_gram_list[0]
                target_img_num = len(bright_feature_gram_list)
                for j in range(target_img_num-1):
                    bright_feature_gram = bright_feature_gram + bright_feature_gram_list[j]
                    blooming_feature_gram = blooming_feature_gram + blooming_feature_gram_list[j]
                    BG_feature_gram = BG_feature_gram + BG_feature_gram_list[j]

                bright_feature_gram = torch.div(bright_feature_gram, target_img_num)
                blooming_feature_gram = torch.div(blooming_feature_gram, target_img_num)
                BG_feature_gram = torch.div(BG_feature_gram, target_img_num)
                #target_feature_gram = bright_feature_gram + blooming_feature_gram*0.4 + BG_feature_gram * 0.2
                #target_feature_gram = torch.div(target_feature_gram, 1.6)

                style_loss = StyleLoss(bright_feature_gram, blooming_feature_gram, BG_feature_gram, style_weight, content_img)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)
            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***
    #print(model)
    #print(style_loss)
    #print(content_loss)
    return model, style_losses, content_losses

#执行mask操作
def creat_mask(img, size, channel):
    r,g,b = img.split(1,dim=1)
    bright = r + g + b
    mean_bright = torch.div(bright,3)

    mean_bright = F.interpolate(mean_bright, size=[size,size], mode="bilinear")

    zero = torch.zeros_like(mean_bright)
    one = torch.ones_like(mean_bright)

    bright_mask1 = torch.where(mean_bright<=0, zero, one)

    max_pool = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)#膨胀掩膜范围
    blooming_mask1 = max_pool(bright_mask1)
    background_mask1 = -blooming_mask1+1
    blooming_mask1 = blooming_mask1 - bright_mask1

    bright_mask1 = bright_mask1.type(torch.bool)
    bright_mask = bright_mask1
    blooming_mask1 = blooming_mask1.type(torch.bool)
    blooming_mask = blooming_mask1
    background_mask1 = background_mask1.type(torch.bool)
    background_mask = background_mask1

    for i in range(channel-1):
        bright_mask = torch.cat([bright_mask,bright_mask1], dim=1)
        blooming_mask = torch.cat([blooming_mask,blooming_mask1], dim=1)
        background_mask = torch.cat([background_mask,background_mask1], dim=1)

    return bright_mask, blooming_mask, background_mask

def get_input_param_optimizer(input_img):
    input_param = nn.Parameter(input_img.data)

    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


def run_style_transfer(opt, cnn, style_dataset, content_img, num_steps=300, style_weight=1000, content_weight=1, use_cuda=False):

    model, style_losses, content_losses = get_style_model_and_losses(opt, cnn, style_dataset, content_img, style_weight, content_weight)#网络搭建

    content_img = content_img.cuda()

    input_img = content_img.clone()#用内容图像初始化
    mask = creat_mask(input_img, size=500,channel=3)
    input_param, optimizer = get_input_param_optimizer(input_img)#获取参数与梯度

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] == opt.epoch:
                print('Style Loss: {:4f}  Content Loss: {:4f}'.format(style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)
    return input_param.data
