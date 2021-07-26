import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import dataset_dict


class FrameBuffer(Dataset):
  def __init__(self, hparams, batch_size, epoch):
    self.hparams = hparams
    self.batch_size = batch_size
    self.indxs = []
    self.current_chunk = []

    self.wh = tuple(self.hparams.img_wh)
    self.chunk_size_in_rays = self.hparams.chunk_size * self.wh[0] * self.wh[1]

    self.set_dataset()

    self.fill_chunk(epoch)

  def __getitem__(self, index):
        return self.current_chunk[0]

  def __len__(self):
        return self.hparams.num_iters_per_chunk * self.batch_size

  def set_dataset(self):

    kwargs = {'root_dir': self.hparams.root_dir,
              'img_wh': tuple(self.hparams.img_wh)}
    self.online_dataset = dataset_dict[self.hparams.dataset_name](self.hparams, split='train',**kwargs)

  def add_data(self, i, sample):
    """
    sample is a python dictionary containing data and target
    """
    self.current_chunk.append(sample)
    self.indxs.append(i)

  def fill_chunk(self, chunk_num):
    
    self.current_chunk.clear()
    self.online_dataset.read_chunk_from_disk(chunk_num)
    self.buffer_size = len(self.online_dataset)
    print("Updating chunk ", chunk_num)
    for i in range(self.buffer_size):
      self.add_data(i, self.online_dataset[i])
    print("Updating done")

  def get_batch(self):

    inds = torch.randint(0,len(self.current_chunk),(self.batch_size,))

    current_batch = []
    for i in inds:
      current_batch.append(self.current_chunk[i])
    
    batch = {}
    batch['rays'] = torch.stack([sample['rays'] for sample in current_batch]).cuda()
    batch['rgbs'] = torch.stack([sample['rgbs'] for sample in current_batch]).cuda()
    batch['idx'] = torch.stack([sample['idx'] for sample in current_batch]).cuda()
    return batch

  def get_test_frame(self, epoch_num, frame_num):
    return self.online_dataset.get_test_frame(epoch_num, frame_num)

  def get_filling_batch(self, i):
    inds = torch.linspace(i*4096*8, i*4096*8 + 4095*8 ,4096).squeeze(0).to(torch.int)    
    current_batch = []
    for idx in inds:
      current_batch.append(self.current_chunk[idx])
    
    batch = {}
    batch['rays'] = torch.stack([sample['rays'] for sample in current_batch]).cuda()
    batch['rgbs'] = torch.stack([sample['rgbs'] for sample in current_batch]).cuda()
    return batch

     

