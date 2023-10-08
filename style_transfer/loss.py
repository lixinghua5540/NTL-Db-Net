import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self, target, weight, content_img):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        self.content_img = content_img
        self.zero = torch.zeros_like(target)
        self.weight = weight
        self.criterion = nn.MSELoss()

    def creat_mask(self, size, channel):
        r,g,b = self.content_img.split(1,dim=1)
        bright = r + g + b
        mean_bright = torch.div(bright,3)

        mean_bright = F.interpolate(mean_bright, size=[size,size], mode="bilinear")
        zero = torch.zeros_like(mean_bright)
        one = torch.ones_like(mean_bright)
        mask = torch.where(mean_bright<=0, zero, one)

        max_pool = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)#膨胀掩膜范围
        mask = max_pool(mask)
        mask = mask.type(torch.bool)
        
        totle_mask = mask.clone()
        for i in range(channel-1):
            totle_mask = torch.cat([totle_mask,mask], dim=1)
        return totle_mask

    def forward(self, input):
        self.mask = self.creat_mask(self.target.shape[2], self.target.shape[1])
        self.content = input.masked_fill(~self.mask, 0)
        content_target = self.target.masked_fill(~self.mask, 0)
        #self.background = input.masked_fill(self.mask, 0)

        self.content_loss = self.criterion(self.content * self.weight, content_target)
        #self.background_loss = self.criterion(self.background * self.weight, self.zero)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.content_loss.backward(retain_graph=retain_graph)
        #self.background_loss.backward(retain_graph=retain_graph)
        return self.content_loss# + self.background_loss

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, bright_feature_gram, blooming_feature_gram, BG_feature_gram, weight, content_img):
        super(StyleLoss, self).__init__()
        self.bright_feature_gram = bright_feature_gram.detach() * weight
        self.blooming_feature_gram = blooming_feature_gram.detach() * weight*0.2
        self.BG_feature_gram = BG_feature_gram.detach() * weight*0.1

        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
        self.content_img = content_img

    def forward(self, input):
        self.input = input.clone()
        self.bright_mask, self.blooming_mask, self.background_mask = self.creat_mask(self.input.shape[2], self.input.shape[1])
        self.output = input.clone()
        self.bright_G = self.gram(self.input.masked_fill(~self.bright_mask, 0))
        self.blooming_G = self.gram(self.input.masked_fill(~self.blooming_mask, 0))
        self.background_G = self.gram(self.input.masked_fill(~self.background_mask, 0))
        self.bright_G.mul_(self.weight)
        self.blooming_G.mul_(self.weight)
        self.background_G.mul_(self.weight)

        self.bright_loss = self.criterion(self.bright_G, self.bright_feature_gram)
        self.blooming_loss = self.criterion(self.blooming_G, self.blooming_feature_gram)
        self.background_loss = self.criterion(self.background_G, self.BG_feature_gram)

        self.loss = self.bright_loss + self.blooming_loss + self.background_loss
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

    def creat_mask(self, size, channel):
        r,g,b = self.content_img.split(1,dim=1)
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
