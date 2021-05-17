from operator import index
import torch 
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from datasets import dataset_dict
import sys
import os

class Buffer(Dataset):
  """
  Implements the experience replay buffer"
  """
  def __init__(self, hparams, batch_size, epoch):
    self.hparams = hparams
    self.batch_size = batch_size
    wh = tuple(self.hparams.img_wh)
    self.chunk_size_in_rays = self.hparams.chunk_size * wh[0] * wh[1]

    self.data_samples = []
    self.indxs = []

    self.get_all_data()

    if epoch == 0:
      self.losses = [0 for i in range(self.buffer_size)] 
    else:
      print("Restoring buffer")
      losses = torch.load('./buffer_losses.pt')
      self.losses = losses.tolist()   

  def get_all_data(self):

    kwargs = {'root_dir': self.hparams.root_dir,
              'img_wh': tuple(self.hparams.img_wh)}
    self.online_dataset = dataset_dict[self.hparams.dataset_name](self.hparams, split='train',**kwargs)

    self.buffer_size = len(self.online_dataset)

    for i in range(self.buffer_size):
      self.add_data(i, self.online_dataset[i])

  def __getitem__(self, index):
        return self.data_samples[index]

  def __len__(self):
        return len(self.data_samples)

  def add_data(self, i, sample):
    """
    sample is a python dictionary containing data and target
    """
    self.data_samples.append(sample)
    self.indxs.append(i)

  def update_buffer(self, replay_inds, replay_losses):

    for i in range(self.batch_size):
      self.losses[replay_inds[i]] = replay_losses[i]

    losses = torch.tensor(self.losses).clone().detach()
    torch.save(losses, './buffer_losses.pt')

  def get_batch(self, current_epoch, exploration_ratio):

    losses = torch.tensor(self.losses)[0:current_epoch*self.chunk_size_in_rays]
    denom = torch.sum(losses)
    weights = (losses/denom).squeeze(-1)

    n_explore = int(self.batch_size * exploration_ratio)
    n_weigted = self.batch_size - n_explore

    weighted_inds = torch.multinomial(weights, n_weigted, replacement=False)

    inds = weighted_inds
    
    if exploration_ratio > 0.0:
      index_mask = torch.ones((current_epoch*self.chunk_size_in_rays))
      index_mask[weighted_inds] = 0
      summat = torch.sum(index_mask)
      index_mask = index_mask/summat 
      explore_inds = torch.multinomial(index_mask, n_explore, replacement=False)
      inds = torch.cat([weighted_inds, explore_inds])

    current_batch = []
    current_indices = []
    for i in inds:
      current_batch.append(self.data_samples[i])
      current_indices.append(i)

    batch = {}
    batch['rays'] = torch.stack([sample['rays'] for sample in current_batch]).cuda()
    batch['rgbs'] = torch.stack([sample['rgbs'] for sample in current_batch]).cuda()
    batch['idx']  = torch.tensor(current_indices).cuda()
    
    return batch
