from operator import truediv
import torch.utils.data
from data_code.creat_dataset import CreatContentDataset, CreatStyleDataset, CreatDebloomingDataset, CreatTestDataset

class ContentDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreatContentDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

class StyleDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreatStyleDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

class DebloomingDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreatDebloomingDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=int(opt.nThreads)
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

class TestDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreatTestDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=int(opt.nThreads)
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
