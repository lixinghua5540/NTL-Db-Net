from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
import util.util as util
from util.image_pool import ImagePool
from torch.autograd import Variable
import math

###############################################################################
# Self-made Loss Functions
###############################################################################
'''
class SelfLoss(torch.nn.Module):#局部加权
	def __init__(self):
		super(SelfLoss, self).__init__()

	def forward(self):
		import torch
		import torch.nn as nn
		import torch.nn.functional as F

		import matplotlib.pyplot as plt


		output = F.log_softmax(torch.randn(1, 3, 24, 24), 1)
		target = torch.zeros(1, 24, 24, dtype=torch.long)
		target[0, 4:12, 4:12] = 1
		target[0, 14:20, 14:20] = 2

		# Edge calculation
		bin_target = torch.where(target > 0, torch.tensor(1), torch.tensor(0))
		plt.imshow(bin_target[0])

		o = F.avg_pool2d(bin_target.float(), kernel_size=3, padding=1, stride=1)
		plt.imshow(o[0])

		edge_idx = (o.ge(0.01) * o.le(0.99)).float()
		plt.imshow(edge_idx[0])

		weights = torch.ones_like(edge_idx, dtype=torch.float)
		weights_sum0 = weights.sum()
		weights = weights + edge_idx * 2.
		weights_sum1 = weights.sum()
		weights = weights / weights_sum1 * weights_sum0 # Rescale weigths
		plt.imshow(weights[0])

		# Calculate loss
		criterion = nn.NLLLoss(reduce=False)
		loss = criterion(output, target)
		loss = loss * weights
		loss = loss.sum() / weights.sum()
'''
class IntensityLoss(torch.nn.Module):#光强损失（去除背景值）
	def __init__(self):#,opt
		super(IntensityLoss, self).__init__()

	def forward(self, input, target):
		weight = target.clone().detach()
		weight = weight+1

		input=torch.mul(input, weight)

		in_intensity = input[0,0]+input[0,1]+input[0,2]
		tar_intensity = target[0,0]+target[0,1]+target[0,2]

		in_intensity = torch.div(in_intensity,3)
		tar_intensity = torch.div(tar_intensity,3)

		#暂时未加入阶梯惩罚
		in_ratio=in_intensity.unsqueeze(0)
		tar_ratio=in_intensity.unsqueeze(0)

		loss_type = nn.MSELoss()
		loss = loss_type(tar_ratio, in_ratio)
		return loss

class HSILoss(torch.nn.Module):#HSI损失
	def __init__(self):#,opt
		super(HSILoss, self).__init__()

	def RGB2HSInormbatch(self, RGBtorch):
		eps = 10e-6
		b,c,h,w=RGBtorch.shape
		#Hsitorch=Rgbtorch.clone()[]
		HSItorch=torch.ones([b,c,h,w],dtype=torch.float32).cuda()
		for i in range(b):
			Rgbtorcht=(RGBtorch[i].permute(1,2,0) + 1) / 2.0 * 5000.0#去计算归一化

		#hsi归一化，除2048并截留
			Rgbtorch0=torch.clip(Rgbtorcht/4096,min=0,max=1)
			R=Rgbtorch0[:,:,0]
			G=Rgbtorch0[:,:,1]
			B=Rgbtorch0[:,:,2]
			I=(R+G+B)/3
			Min1=torch.minimum(R,G)
			Min=torch.minimum(Min1,B)
			S=1-3*Min/(R+G+B+eps)
			#theta=torch.divide((R-G+R-B),(2*(torch.multiply(R-G,R-G)+torch.multiply(R-B,G-B))+eps))#arccos反三角值域0到pi,torch.div
			theta=torch.arccos(torch.divide((R-G+R-B),2*torch.sqrt(torch.multiply(R-G,R-G)+torch.multiply(R-B,G-B) +eps))/(1+eps*100))
			H=torch.zeros_like(theta)#会将梯度传给源tensor
			compare1=B>G#logic T/F
			compare1i=B<=G#logic T/F
			#此处要用矩阵逻辑判断,把compare1为True 的替换为2pi
			thetainv=2*torch.ones_like(theta)*math.pi-theta
			H[compare1]=thetainv[compare1]
			H[compare1i]=theta[compare1i]
			inds1=S==0
			inds2=S!=0
			#H[inds1]=0#inpalce
			Hue=torch.zeros_like(H)
			Hue[inds1]=0
			Hue[inds2]=H[inds2]
			Hue=Hue/(2*math.pi)
			Hue=Hue.unsqueeze(-1)
			S=S.unsqueeze(-1)
			I=I.unsqueeze(-1)
			Hsi=torch.cat((Hue,S,I),axis=-1)
			Hsi=Hsi.permute(2,0,1)
			Hsi=Hsi.unsqueeze(0)
			HSItorch[i]=Hsi
		return HSItorch

	def forward(self, input, target):
		HSI_input = self.RGB2HSInormbatch(input)
		HSI_target = self.RGB2HSInormbatch(target)

		loss_type = nn.MSELoss()
		loss = loss_type(HSI_target, HSI_input)
		
		return loss

###############################################################################
# Functions
###############################################################################

class ImageLoss():#图像损失 = 感知损失 + 内容损失
	def perceptualFunc(self):
		conv_3_3_layer = 7
		cnn = models.vgg13(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, perceptual_loss, content_loss):
		self.P_criterion = perceptual_loss#感知损失
		self.C_criterion = content_loss#内容损失

		self.perceptualFunc = self.perceptualFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.perceptualFunc.forward(fakeIm)
		f_real = self.perceptualFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss_P = self.P_criterion(f_fake, f_real_no_grad)
		loss_C = self.C_criterion(fakeIm, realIm)
		return loss_P + loss_C

class PixelLoss():#像素损失 = HSI损失 + 光强损失，光强损失未计算
	def __init__(self, spectrum_loss, intensity_loss):
		self.S_criterion = spectrum_loss
		self.I_criterion = intensity_loss

	def get_loss(self, fakeIm, realIm):
		spectrum_loss = self.S_criterion(fakeIm, realIm)
		#intensity_loss = self.I_criterion(fakeIm,realIm)

		return spectrum_loss# + intensity_loss
		
class GANLoss(nn.Module):#GAN损失
	def __init__(
			self, use_l1=True, target_real_label=1.0,
			target_fake_label=0.0, tensor=torch.FloatTensor):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_l1:
			self.loss = nn.L1Loss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		target_tensor = None
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)

class DiscLoss:
	def name(self):
		return 'DiscLoss'

	def __init__(self, opt, tensor):
		self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)
		self.fake_AB_pool = ImagePool(opt.pool_size)
		
	def get_g_loss(self,net, realA, fakeB):
		# First, G(A) should fake the discriminator
		pred_fake = net.forward(fakeB)
		return self.criterionGAN(pred_fake, 1)
		
	def get_loss(self, net, realA, fakeB, realB):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		self.pred_fake = net.forward(fakeB.detach())
		self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

		# Real
		self.pred_real = net.forward(realB)
		self.loss_D_real = self.criterionGAN(self.pred_real, 1)

		# Combined loss
		self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
		return self.loss_D
		
class DiscLossLS(DiscLoss):
	def name(self):
		return 'DiscLossLS'

	def __init__(self, opt, tensor):
		super(DiscLossLS, self).__init__(opt, tensor)
		#DiscLoss.initialize(self, opt, tensor)
		self.criterionGAN = GANLoss(use_l1=True, tensor=tensor)
		
	def get_g_loss(self, net, realA, fakeB):
		return DiscLoss.get_g_loss(self,net, realA, fakeB)
		
	def get_loss(self, net, realA, fakeB, realB):
		return DiscLoss.get_loss(self, net, realA, fakeB, realB)
		
class DiscLossWGANGP(DiscLossLS):
	def name(self):
		return 'DiscLossWGAN-GP'

	def __init__(self, opt, tensor):
		super(DiscLossWGANGP, self).__init__(opt, tensor)
		# DiscLossLS.initialize(self, opt, tensor)
		self.LAMBDA = 10
		
	def get_g_loss(self, net, realA, fakeB):
		# First, G(A) should fake the discriminator
		self.D_fake = net.forward(fakeB)
		return -self.D_fake.mean()
		
	def calc_gradient_penalty(self, netD, real_data, fake_data):
		alpha = torch.rand(1, 1)
		alpha = alpha.expand(real_data.size())
		alpha = alpha.cuda()

		interpolates = alpha * real_data + ((1 - alpha) * fake_data)

		interpolates = interpolates.cuda()
		interpolates = Variable(interpolates, requires_grad=True)
		
		disc_interpolates = netD.forward(interpolates)

		gradients = autograd.grad(
			outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
			create_graph=True, retain_graph=True, only_inputs=True
		)[0]

		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
		return gradient_penalty
		
	def get_loss(self, net, realA, fakeB, realB):
		self.D_fake = net.forward(fakeB.detach())
		self.D_fake = self.D_fake.mean()
		
		# Real
		self.D_real = net.forward(realB)
		self.D_real = self.D_real.mean()
		# Combined loss
		self.loss_D = self.D_fake - self.D_real
		gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
		return self.loss_D + gradient_penalty


def init_loss(opt, tensor):
	disc_loss = None#
	image_loss = None#图像损失
	pixel_loss = None#像素损失

	if opt.gan_type == 'wgan-gp':
		disc_loss = DiscLossWGANGP(opt, tensor)
	elif opt.gan_type == 'lsgan':
		disc_loss = DiscLossLS(opt, tensor)
	elif opt.gan_type == 'gan':
		disc_loss = DiscLoss(opt, tensor)
	else:
		raise ValueError("GAN [%s] not recognized." % opt.gan_type)
	# disc_loss.initialize(opt, tensor)


	#图像损失 = 感知损失 + 内容损失
	image_loss = ImageLoss(perceptual_loss=nn.MSELoss(), content_loss=nn.L1Loss())
	
	#像素损失 = 光谱损失 + 光强损失
	pixel_loss = PixelLoss(spectrum_loss=HSILoss(), intensity_loss=IntensityLoss())

	return disc_loss, image_loss, pixel_loss
