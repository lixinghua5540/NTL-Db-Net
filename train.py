from re import A
import time
import torch
import torchvision.models as models
import numpy as np
import cv2

from alive_progress import alive_bar
#from util.visualizer import Visualizer
from util.metrics import PSNR
from multiprocessing import freeze_support
from torchvision import utils as vutils
from matplotlib.image import imsave

from options.styletransfer_options import StyleTransferOptions
from options.debloomingGAN_options import DebloomingGANOptions
from data_code.data_loader import StyleDataLoader, ContentDataLoader, DebloomingDataLoader
from deblommingGAN.debloomingGAN import Deblooming_GAN
from style_transfer.style_transfer import run_style_transfer

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


##风格迁移网络
def style_transfer(opt, content_dataloader, style_dataloader):
	start_time = time.time()
	content_dataset = content_dataloader.load_data()
	style_dataset = style_dataloader.load_data()

	dataset_size = len(content_dataloader)
	use_cuda = torch.cuda.is_available() and opt.cuda

	cnn = models.vgg16(pretrained=True).features
	if use_cuda:
		cnn = cnn.cuda()

	for i,content_data in enumerate(content_dataset):
		print('正在迁移图像 %d , 共计 %d 张图像...' % (i+1, dataset_size))
		filepath = content_dataset.dataset.paths_real_sharp[i]
		filename = os.path.basename(filepath)

		output_tensor = run_style_transfer(opt, cnn, style_dataset, content_data, opt.epoch, opt.style_weight, opt.content_weight, use_cuda)
		style_filename=opt.dataroot_fake_blur + '/' + filename
		style_filename_backup=opt.dataroot_fake_blur + '/' + filename
		vutils.save_image(output_tensor, style_filename)
		vutils.save_image(output_tensor, style_filename_backup)
	end_time = time.time()
	print('完成风格迁移, 总耗时 %f s' %(end_time-start_time))

#考虑物理机制的模糊图像 + 风格迁移
def physical_blur(opt, dataloader):
	dataset_size = len(dataloader)
	psf_parameter=opt.psf_parameter
	psfSize = opt.psf_size
	kernel = np.zeros([psfSize, psfSize], np.float32)
	sigma=psf_parameter[0]
	A=psf_parameter[1]
	Beta=psf_parameter[2]
	totleValue = 0
	for i in range(psfSize):
		for j in range(psfSize):
			value = A/(((i-psfSize//2)**2 + (j-psfSize//2)**2)/(sigma**2)+1)**Beta
			kernel[i,j]=value
			totleValue += value
	for i in range(psfSize):
		for j in range(psfSize):
			kernel[i,j]=kernel[i,j] / totleValue

	with alive_bar(dataset_size, force_tty=True, title="正在融合模糊物理机制...") as phy_bar:
		for i in range(dataset_size):
			blur_dataroot = dataloader.dataset.paths_fake_blur[i]
			sharp_dataroot = dataloader.dataset.paths_real_sharp[i]
			blurImg = cv2.imread(blur_dataroot)
			sharpImg = cv2.imread(sharp_dataroot)

			dilatekernel = np.ones((5, 5), dtype=np.uint8)
			dilateImg = cv2.dilate(sharpImg, dilatekernel, 2)

			psfBlurImg = cv2.filter2D(dilateImg, -1, kernel=kernel)
			result = cv2.addWeighted(psfBlurImg, 0.5, blurImg, 0.5, 0)
			output = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)#

			imsave(blur_dataroot,output)

			phy_bar()
	return dataset_size

def creat_blackimg(opt, num_img):
	sharp_root = opt.dataroot_real_sharp
	blur_root = opt.dataroot_fake_blur

	num_black = num_img // 4
	for i in range(num_black):
		black_img = np.zeros([500,500,3], dtype = np.uint8)
		cv2.imwrite(sharp_root+"/"+str(num_img+i+1)+".jpg", black_img)
		cv2.imwrite(blur_root+"/"+str(num_img+i+1)+".jpg", black_img)

	return

##除模糊网络DebloomingGAN
def train_debloomingGAN(opt, data_loader, model):
	dataset = data_loader.load_data()
	total_steps = 0
	start_time = time.time()
	with alive_bar(len(data_loader)*(opt.niter+opt.niter_decay), force_tty=True, title="正在训练debloomingGAN...") as deblooming_bar:
		for epoch in range(1, opt.niter + opt.niter_decay + 1):
			for i,data in enumerate(dataset):
				total_steps += opt.batchSize
				model.set_input(data)
				model.optimize_parameters()
				deblooming_bar()

				if (total_steps % opt.save_latest_freq == 0):# or (epoch == opt.niter + opt.niter_decay)
					print('保存最新模型参数... (当前epoch: %d, 总计steps: %d)' % (epoch, total_steps))
					model.save('latest')
					results = model.get_current_visuals()
					psnrMetric = PSNR(results['fake_img'], results['sharp_img'])
					print('epoch = %d, PSNR = %f' % (epoch, psnrMetric))
					loss = model.get_current_errors()
					print('Loss:\n G_GAN:%f\n G_Image:%f\n G_Pixel:%f\n D_real+fake:%f' % (loss['G_GAN'],loss['G_Image'],loss['G_Pixel'],loss['D_real+fake']))

				if epoch % opt.save_epoch_freq == 0:
					results = model.get_current_visuals()
					output = cv2.cvtColor(results['fake_img'], cv2.COLOR_BGR2RGB)
					cv2.imwrite("./dataset/train_result/epoch"+str(epoch)+"_img"+str(i)+".jpg", output)

			if epoch > opt.niter:
				model.update_learning_rate()
	
	end_time = time.time()
	print('完成Deblooming网络训练, 总耗时 %f s' %(end_time-start_time))


if __name__ == '__main__':
	freeze_support()

	blur_opt = StyleTransferOptions().parse()
	blur_opt.psf_parameter = [20.0, 255.0, 20.0]
	content_dataloader = ContentDataLoader(blur_opt)
	style_dataloader = StyleDataLoader(blur_opt)
	
	print('---------------风格转换网络训练---------------')
	style_transfer(blur_opt, content_dataloader, style_dataloader)

	deblooming_opt = DebloomingGANOptions().parse()
	phyblur_dataloader = DebloomingDataLoader(deblooming_opt)
	print('---------------物理机制模糊模拟---------------')
	num_img = physical_blur(blur_opt, phyblur_dataloader)

	creat_blackimg(blur_opt, num_img)
	deblooming_dataloader = DebloomingDataLoader(deblooming_opt)
	print('---------------Deblooming网络训练---------------')
	model_deblooming = Deblooming_GAN(deblooming_opt)
	train_debloomingGAN(deblooming_opt, deblooming_dataloader, model_deblooming)

