import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from . import networks
from .losses import init_loss
from PIL import Image

class Deblooming_GAN():
	def __init__(self, opt):
		self.opt = opt
		self.gpu_ids = opt.gpu_ids
		self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
		self.save_dir = opt.checkpoints_dir

		self.isTrain = opt.isTrain
		# 定义tensors
		self.input_blur = self.Tensor(opt.batchSize, opt.input_nc,  opt.fineSize, opt.fineSize)
		self.input_sharp = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

		# 定义网络
		use_parallel = opt.use_parallel
		self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.norm,
									 self.gpu_ids, use_parallel, opt.learn_residual)
		
		#是否为训练阶段
		if self.isTrain:
			use_sigmoid = False
			self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)
		if not self.isTrain or opt.continue_train:
			self.load_network(self.netG, 'G', opt.which_epoch)
			if self.isTrain:
				self.load_network(self.netD, 'D', opt.which_epoch)

		if self.isTrain:
			self.fake_AB_pool = ImagePool(opt.pool_size)
			self.old_lr = opt.lr

			# initialize optimizers
			self.optimizer_G = torch.optim.Adam( self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
			self.optimizer_D = torch.optim.Adam( self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
												
			self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1
			
			# define loss functions
			self.discLoss, self.imageLoss, self.pixelLoss= init_loss(opt, self.Tensor)

	def set_input(self, input):
		self.input = input
		#1, x, 500, 500
		input_blur = input['fake_blur']
		input_sharp = input['real_sharp']
		self.input_blur.resize_(input_blur.size()).copy_(input_blur)
		self.input_sharp.resize_(input_sharp.size()).copy_(input_sharp)

	def forward(self):
		self.blur_img = Variable(self.input_blur)
		self.all_img = self.netG.forward(self.blur_img)
		self.fake_img = self.all_img['output']
		self.fake_input = self.all_img['input']
		#self.fake_output = self.all_img['output']
		self.sharp_img = Variable(self.input_sharp)

	def test(self):
		self.blur_img = Variable(self.input_blur, volatile=True)
		self.fake_img = self.netG.forward(self.blur_img)
		self.sharp_img = Variable(self.input_sharp, volatile=True)

	def backward_D(self):
		self.loss_D = self.discLoss.get_loss(self.netD, self.blur_img, self.fake_img, self.sharp_img)

		self.loss_D.backward(retain_graph=True)

	def backward_G(self):
		self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.blur_img, self.fake_img) * self.opt.GAN_G_LossW
		# Second, G(A) = B
		self.loss_G_Image = self.imageLoss.get_loss(self.fake_img, self.sharp_img) * self.opt.image_LossW
		self.loss_G_Pixel = self.pixelLoss.get_loss(self.fake_img, self.sharp_img) * self.opt.pixel_LossW


		self.loss_G = self.loss_G_GAN + self.loss_G_Image + self.loss_G_Pixel

		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()

		for iter_d in range(self.criticUpdates):
			self.optimizer_D.zero_grad()
			self.backward_D()
			self.optimizer_D.step()

		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

	def get_current_errors(self):
		return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
							('G_Image', self.loss_G_Image.item()),
							('G_Pixel', self.loss_G_Pixel.item()),
							('D_real+fake', self.loss_D.item())
							])

	def get_current_visuals(self):
		blur_img = util.tensor2im(self.blur_img.data)
		fake_img = util.tensor2im(self.fake_img.data)
		fake_input = util.tensor2im(self.fake_input.data)
		#fake_output = util.tensor2im(self.fake_output.data)
		sharp_img = util.tensor2im(self.sharp_img.data)

		#fake_B = fake_B1 + fake_B2
		return OrderedDict([('blur_img', blur_img), ('fake_img', fake_img), ('fake_input', fake_input), ('sharp_img', sharp_img)])#('fake_output', fake_output), 

	def save(self, label):
		self.save_network(self.netG, 'G', label, self.gpu_ids)
		self.save_network(self.netD, 'D', label, self.gpu_ids)

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_D.param_groups:
			param_group['lr'] = lr
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		self.old_lr = lr
		if self.old_lr < 0:
			self.old_lr = 0

	def reset_learning_rate(self):
		self.opt.lr = self.opt.lr_stable
		self.old_lr = self.opt.lr_stable

    # helper saving function that can be used by subclasses
	def save_network(self, network, network_label, epoch_label, gpu_ids):
		save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
		save_path = os.path.join(self.save_dir, save_filename)
		torch.save(network.cpu().state_dict(), save_path)
		if len(gpu_ids) and torch.cuda.is_available():
			network.cuda(device=gpu_ids[0])


    # helper loading function that can be used by subclasses
	def load_network(self, network, network_label, epoch_label):
		save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
		save_path = os.path.join(self.save_dir, save_filename)
		network.load_state_dict(torch.load(save_path))

	def save_image(self, image_numpy, image_path):
		image_pil = None
		if image_numpy.shape[2] == 1:
			image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
			image_pil = Image.fromarray(image_numpy, 'L')
		else:
			image_pil = Image.fromarray(image_numpy)
		image_pil.save(image_path)
