import os.path
import random
import torchvision.transforms as transforms
import torch
from data_code.base_dataset import BaseDataset
from data_code.image_folder import make_dataset, make_test_dataset
from PIL import Image


class CreatContentDataset(BaseDataset):
    def __init__(self, opt):
        #super(CreatDataset, self).__init__()
        self.opt = opt
        self.paths_real_sharp = sorted(make_dataset(opt.dataroot_real_sharp))

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        #归一化为tensor，并且调整值为[-1,1]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        path_real_sharp = self.paths_real_sharp[index]
        img_real_sharp = Image.open(path_real_sharp).convert('RGB')
        img_real_sharp = img_real_sharp.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        img_real_sharp = self.transform(img_real_sharp)
        
        return img_real_sharp

    def __len__(self):
        return len(self.paths_real_sharp)

    def name(self):
        return 'ContentDataset'

class CreatStyleDataset(BaseDataset):
    def __init__(self, opt):
        #super(CreatDataset, self).__init__()
        self.opt = opt
        self.paths_real_blur = sorted(make_dataset(opt.dataroot_real_blur))

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        #归一化为tensor，并且调整值为[-1,1]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        path_real_blur = self.paths_real_blur[index]
        img_real_blur = Image.open(path_real_blur).convert('RGB')
        img_real_blur = img_real_blur.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        img_real_blur = self.transform(img_real_blur)
        return img_real_blur

    def __len__(self):
        return len(self.paths_real_blur)

    def name(self):
        return 'StyleDataset'

class CreatDebloomingDataset(BaseDataset):
    def __init__(self, opt):
        #super(CreatDataset, self).__init__()
        self.opt = opt
        self.paths_fake_blur = sorted(make_dataset(opt.dataroot_fake_blur))
        self.paths_real_sharp = sorted(make_dataset(opt.dataroot_real_sharp))

        #transform_list = [transforms.ToTensor()]#归一化为tensor
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        #归一化为tensor，并且调整值为[-1,1]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        path_fake_blur = self.paths_fake_blur[index]
        img_fake_blur = Image.open(path_fake_blur).convert('RGB')
        img_fake_blur = img_fake_blur.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        img_fake_blur = self.transform(img_fake_blur)
        
        path_real_sharp = self.paths_real_sharp[index]
        img_real_sharp = Image.open(path_real_sharp).convert('RGB')
        img_real_sharp = img_real_sharp.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        img_real_sharp = self.transform(img_real_sharp)
        return {'fake_blur': img_fake_blur, 'real_sharp': img_real_sharp}

    def __len__(self):
        return len(self.paths_fake_blur)

    def name(self):
        return 'DebloomingDataset'

class CreatTestDataset(BaseDataset):
    def __init__(self, opt):
        #super(CreatDataset, self).__init__()
        self.opt = opt
        self.paths_test_img = make_test_dataset(opt.test_dir)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        #归一化为tensor，并且调整值为[-1,1]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        path_test_img = self.paths_test_img[index]
        img_test_img = Image.open(path_test_img).convert('RGB')
        img_test_img = img_test_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        img_test_img = self.transform(img_test_img)
        return img_test_img

    def __len__(self):
        return len(self.paths_test_img)

    def name(self):
        return 'TestDataset'


