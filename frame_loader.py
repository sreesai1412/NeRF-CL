import sys
import os
import numpy as np
import torch 

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from datasets import dataset_dict

class FrameLoader(Dataset):
  def __init__(self, hparams, batch_size):
    self.hparams = hparams
    self.batch_size = batch_size

    self.indxs = []
    self.current_frame = None

    self.wh = tuple(self.hparams.img_wh)
    self.frame_size_in_rays = self.wh[0] * self.wh[1]

    kwargs = {'root_dir': self.hparams.root_dir,
              'img_wh': tuple(self.hparams.img_wh)}
    self.dataset = dataset_dict[self.hparams.dataset_name](self.hparams, split='train',**kwargs)

    self.train_indxs =  torch.randperm(self.hparams.num_chunks * self.hparams.num_train_in_each_chunk)
    self.fill_frame(0)

  def __getitem__(self, index):
        return self.current_frame

  def __len__(self):
        return self.hparams.num_iters_per_epoch * self.batch_size

  def fill_frame(self, frame_num):
    self.current_frame = self.dataset[self.train_indxs[frame_num]]

  def get_batch(self):
    inds = torch.randint(0,self.frame_size_in_rays,(self.batch_size,))
    
    batch = {}
    batch['rays'] = self.current_frame['rays'][inds].cuda()
    batch['rgbs'] = self.current_frame['rgbs'][inds].cuda()
    return batch

  def shuffle_dataset(self):
    self.train_indxs = torch.randperm(self.hparams.num_chunks * self.hparams.num_train_in_each_chunk)
    print("Shuffled the dataset")

     

