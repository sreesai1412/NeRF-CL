import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
import os
import numpy as np
from datasets import dataset_dict

class FrameBuffer(Dataset):
  """
  Implements the experience replay buffer"
  """
  def __init__(self, hparams, batch_size, epoch):
    self.hparams = hparams
    self.batch_size = batch_size

    self.data_samples = []
    self.indxs = []
    self.current_chunk = []

    self.wh = tuple(self.hparams.img_wh)
    self.chunk_size_in_rays = self.hparams.chunk_size * self.wh[0] * self.wh[1]

    self.get_all_data()

    self.fill_chunk(epoch)

  def __getitem__(self, index):
        return self.data_samples[0]

  def __len__(self):
        return self.hparams.num_iters_per_chunk * self.batch_size

  def get_all_data(self):

    kwargs = {'root_dir': self.hparams.root_dir,
              'img_wh': tuple(self.hparams.img_wh)}
    self.online_dataset = dataset_dict[self.hparams.dataset_name](self.hparams, split='train',**kwargs)

    self.buffer_size = len(self.online_dataset)

    for i in range(self.buffer_size):
      self.add_data(i, self.online_dataset[i])

  def add_data(self, i, sample):
    """
    sample is a python dictionary containing data and target
    """
    self.data_samples.append(sample)
    self.indxs.append(i)

  def fill_chunk(self, chunk_num):
    self.current_chunk.clear()
    
    print("Updating chunk")
    print(chunk_num)

    s = chunk_num * self.chunk_size_in_rays
    e = s + self.chunk_size_in_rays

    for i in range(s, e):
      self.current_chunk.append(self.data_samples[i])
    print("Done")

  def get_batch(self):

    inds = torch.randint(0,self.chunk_size_in_rays,(self.batch_size,))

    current_batch = []
    for i in inds:
      current_batch.append(self.current_chunk[i])
    
    batch = {}
    batch['rays'] = torch.stack([sample['rays'] for sample in current_batch]).cuda()
    batch['rgbs'] = torch.stack([sample['rgbs'] for sample in current_batch]).cuda()
    batch['idx'] = torch.stack([torch.tensor(sample['idx']) for sample in current_batch]).cuda()
    return batch

  def get_frame_batch(self, epoch_num, frame_num):
    current_batch = []

    s = (epoch_num * self.chunk_size_in_rays) + (frame_num * self.wh[0] * self.wh[1])
    e = s + (self.wh[0] * self.wh[1])

    for i in range(s, e):
      current_batch.append(self.data_samples[i])
    
    batch = {}
    batch['rays'] = torch.stack([sample['rays'] for sample in current_batch]).cuda()
    batch['rgbs'] = torch.stack([sample['rgbs'] for sample in current_batch]).cuda()
    batch['idx'] = torch.stack([torch.tensor(sample['idx']) for sample in current_batch]).cuda()
    return batch

     

