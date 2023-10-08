import os
import cv2
import numpy as np
from options.test_options import TestOptions
from data_code.data_loader import TestDataLoader
from multiprocessing import freeze_support
#from util.visualizer import Visualizer
from deblommingGAN.test_model import TestModel
from alive_progress import alive_bar

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#滑动窗口法裁剪拼接
def img_cut(opt):
	test_img = cv2.imread(opt.test_img_name)
	size = test_img.shape
	print("测试图像尺寸：%d px * %d px" %(size[0], size[1]))
	opt.step = 300
	opt.num_row = (size[1]) // opt.step
	opt.num_col = (size[0]) // opt.step
	print("测试图像将按照步长 %d px, 被分为 %d * %d 个 500px*500px 的影像块" %(opt.step, opt.num_row, opt.num_col))
	for j in range(opt.num_col):
		for i in range(opt.num_row):
			imgCut = test_img[j*opt.step:j*opt.step+500, i*opt.step:i*opt.step+500]
			adress = opt.test_dir+str(j*opt.num_row+i)+".jpg"
			cv2.imwrite(adress, imgCut)
	return opt.num_row, opt.num_col, opt.step
 
def img_splicing(opt, num_row, num_col, step):#步长为300
	opt.num_row = num_row
	opt.num_col = num_col
	totle_img = ''
	line_img = ''
	if step == 300:
		for j in range(opt.num_col):
			img_array = ''
			for i in range(opt.num_row):
				adress = opt.result_dir+str(j*opt.num_row+i)+".jpg"
				if i == 0:
					img = cv2.imread(adress)  # 打开图片
					if j == 0:
						img = img[0:400,0:400]
					elif j == opt.num_col-1:
						img = img[100:500,0:400]
					else:
						img = img[100:400,0:400]
					img_array = np.array(img)  # 转化为np array对象
				elif i == opt.num_row-1:
					img = cv2.imread(adress)
					if j == 0:
						img = img[0:400,100:500]
					elif j == opt.num_col-1:
						img = img[100:500,100:500]
					else:
						img = img[100:400,100:500]
					img_array2 = np.array(img)  # 转化为np array对象
					img_array = np.concatenate((img_array, img_array2), axis=1)
					line_img = img_array
				else:
					img = cv2.imread(adress)
					if j == 0:
						img = img[0:400,100:400]
					elif j == opt.num_col-1:
						img = img[100:500,100:400]
					else:
						img = img[100:400,100:400]
					img_array2 = np.array(img)  # 转化为np array对象
					img_array = np.concatenate((img_array, img_array2), axis=1)
			if j == 0:
				totle_img = line_img
			else:
				totle_img = np.concatenate((totle_img, line_img), axis=0)
	elif step == 100:
		for j in range(opt.num_col):
			img_array = ''
			for i in range(opt.num_row):
				adress = opt.result_dir+str(j*opt.num_row+i)+".jpg"
				if i == 0:
					img = cv2.imread(adress)  # 打开图片
					if j == 0:
						img = img[0:300,0:300]
					elif j == opt.num_col-1:
						img = img[200:500,0:300]
					else:
						img = img[200:300,0:300]
					img_array = np.array(img)  # 转化为np array对象
				elif i == opt.num_row-1:
					img = cv2.imread(adress)
					if j == 0:
						img = img[0:300,200:500]
					elif j == opt.num_col-1:
						img = img[200:500,200:500]
					else:
						img = img[200:300,200:500]
					img_array2 = np.array(img)  # 转化为np array对象
					img_array = np.concatenate((img_array, img_array2), axis=1)
					line_img = img_array
				else:
					img = cv2.imread(adress)
					if j == 0:
						img = img[0:300,200:300]
					elif j == opt.num_col-1:
						img = img[200:500,200:300]
					else:
						img = img[200:300,200:300]
					img_array2 = np.array(img)  # 转化为np array对象
					img_array = np.concatenate((img_array, img_array2), axis=1)
			if j == 0:
				totle_img = line_img
			else:
				totle_img = np.concatenate((totle_img, line_img), axis=0)

	result_name = "./result_"+os.path.basename(opt.test_img_name)
	cv2.imwrite(result_name,totle_img)

	return
	
def test():
	opt = TestOptions().parse()
	opt.nThreads = 1
	opt.batchSize = 1
	opt.serial_batches = True
	opt.no_flip = True
	
	num_row, num_col, step= img_cut(opt)
	data_loader = TestDataLoader(opt)
	dataset = data_loader.load_data()
	model = TestModel(opt)

	with alive_bar(len(data_loader), force_tty=True, title="正在测试...") as bar:
		for i, data in enumerate(dataset):
			model.set_input(data)
			model.test()
			visuals = model.get_current_visuals()

			filepath = dataset.dataset.paths_test_img[i]
			filename = os.path.basename(filepath)
			img_path = opt.result_dir + filename
			mask_path = opt.result_dir + 'mask/' + filename
			#output = model.fusion(visuals['fake_sharp'], visuals['real_blur'])
			model.save_image(visuals['fake_sharp'], visuals['real_blur'], img_path, mask_path)

			bar()

	img_splicing(opt, num_row, num_col, step)

if __name__ == '__main__':
	freeze_support()
	test()

