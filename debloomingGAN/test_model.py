import os
import torch
from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
#from .base_model import BaseModel
from . import networks
from PIL import Image
import numpy as np
import cv2

class TestModel():#BaseModel
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__()

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = opt.checkpoints_dir
        self.input = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc,  opt.norm, self.gpu_ids, False,
                                      opt.learn_residual)


        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def set_input(self, input):
        # we need to use single_dataset mode
        temp = self.input.clone()
        temp.resize_(input.size()).copy_(input)
        self.input = temp

    def test(self):
        with torch.no_grad():
            self.real_blur = Variable(self.input)
            self.fake_sharp = self.netG.forward(self.real_blur)['output']


    def get_current_visuals(self):
        real_blur = util.tensor2im(self.real_blur.data)
        fake_sharp = util.tensor2im(self.fake_sharp.data)
        return OrderedDict([('real_blur', real_blur), ('fake_sharp', fake_sharp)])

    def fusion(self, deblooming, blur, mask_path):
        deblooming = cv2.cvtColor(np.asarray(deblooming),cv2.COLOR_RGB2BGR) 
        blur = cv2.cvtColor(np.asarray(blur),cv2.COLOR_RGB2BGR) 

        debloominggray = cv2.cvtColor(deblooming, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(debloominggray, 50, 255, cv2.THRESH_BINARY)
        blur = cv2.bitwise_and(blur, blur, mask=mask)

        cv2.imwrite(mask_path, blur)

        output = cv2.addWeighted(deblooming,0.5,blur,0.5,0)
        output = Image.fromarray(cv2.cvtColor(output,cv2.COLOR_BGR2RGB)) 
        return output

    def save_image(self, deblooming_img_np, blur_img_np, image_path, mask_path):
        deblooming_img_pil = None
        blur_img_pil = None
        if deblooming_img_np.shape[2] == 1:
            deblooming_img_np = np.reshape(deblooming_img_np, (deblooming_img_np.shape[0],deblooming_img_np.shape[1]))
            deblooming_img_pil = Image.fromarray(deblooming_img_np, 'L')
        else:
            deblooming_img_pil = Image.fromarray(deblooming_img_np)

        if blur_img_np.shape[2] == 1:
            blur_img_np = np.reshape(blur_img_np, (blur_img_np.shape[0],blur_img_np.shape[1]))
            blur_img_pil = Image.fromarray(blur_img_np, 'L')
        else:
            blur_img_pil = Image.fromarray(blur_img_np)

        output = self.fusion(deblooming_img_pil, blur_img_pil, mask_path)
        output.save(image_path)
