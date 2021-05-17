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
  def __init__(self, hparams, buffer_size, batch_size):
    self.hparams = hparams

    self.buffer_size =buffer_size
    self.batch_size = batch_size

    self.data_samples = []
    self.indxs = []
    self.losses = []

  def fill_buffer(self):
    kwargs = {'root_dir': self.hparams.root_dir,
              'img_wh': tuple(self.hparams.img_wh)}
    
    if self.hparams.train_view == 'left':
        self.replay_dataset = dataset_dict[self.hparams.dataset_name](self.hparams, split='train', view='right',**kwargs)
    else:
        self.replay_dataset = dataset_dict[self.hparams.dataset_name](self.hparams, split='train', view='left',**kwargs)
    
    fill_mode = self.hparams.buffer_fill_mode
    
    if fill_mode == 'random':
      indxs = torch.randint(0,len(self.replay_dataset),(self.buffer_size,))
      for i in indxs:
            self.add_data(i, self.replay_dataset[i])

    elif fill_mode == 'uniform':
      stride = len(self.replay_dataset) // self.buffer_size
      indxs = [i*stride for i in range(self.buffer_size)]
      for i in indxs:
            self.add_data(i, self.replay_dataset[i])

    elif fill_mode == 'topk':
      all_losses = self.replay_dataset.all_losses
      _, indices = torch.sort(all_losses, descending=True, dim=0)

      indxs = indices.squeeze(-1)[:self.buffer_size]
      for i in indxs:
        self.add_data(i, self.replay_dataset[i])

      loss_vals = torch.tensor(self.losses)
      denom = torch.sum(loss_vals)
      self.weights = (loss_vals/denom).squeeze(-1)

      if self.hparams.use_imgcentre_weight:
        gauss = self.get_gauss_weights()
        self.gauss_weights = torch.index_select(gauss, 0, indxs)
        denom = torch.sum(self.gauss_weights)
        self.gauss_weights = (self.gauss_weights/denom).squeeze(-1)

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

    if self.hparams.buffer_fill_mode == 'topk':
        self.losses.append(sample['losses'])

  def get_batch(self, exploration_ratio):
    n_explore = int(self.batch_size * exploration_ratio)
    n_weigted = self.batch_size - n_explore

    if self.hparams.use_imgcentre_weight:
      imgcentre_ratio = self.hparams.imgcentre_ratio
      sampling_weights = (self.gauss_weights*(imgcentre_ratio)) + (self.weights*(1-imgcentre_ratio))
    else:
      sampling_weights = self.weights

    weighted_inds = torch.multinomial(sampling_weights, n_weigted, replacement=False)

    inds = weighted_inds
    
    if exploration_ratio > 0.0:
      index_mask = torch.ones(self.buffer_size)
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
    if self.hparams.use_soft_targets_for_replay:
      batch['soft_rgbs'] = torch.stack([sample['soft_rgbs'] for sample in current_batch]).cuda()
    
    return batch

  def get_topk_batch(self):
    loss_vals = torch.tensor(self.losses)
    _, indices = torch.sort(loss_vals, descending=True, dim=0)
    inds = indices.squeeze(-1)[:self.batch_size]

    current_batch = []
    current_indices = []
    for i in inds:
      current_batch.append(self.data_samples[i])
      current_indices.append(i)

    batch = {}
    batch['rays'] = torch.stack([sample['rays'] for sample in current_batch]).cuda()
    batch['rgbs'] = torch.stack([sample['rgbs'] for sample in current_batch]).cuda()
    batch['idx']  = torch.tensor(current_indices).cuda()
    if self.hparams.use_soft_targets_for_replay:
      batch['soft_rgbs'] = torch.stack([sample['soft_rgbs'] for sample in current_batch]).cuda()

    return batch


  def update_buffer(self, replay_inds, replay_losses):

    for i in range(self.batch_size):
      self.losses[replay_inds[i]] = replay_losses[i]

    loss_vals = torch.tensor(self.losses)
    denom = torch.sum(loss_vals)
    self.weights = (loss_vals/denom).squeeze(-1)

  def get_gauss_weights(self):
    wh = tuple(self.hparams.img_wh)
    x, y = np.meshgrid(np.linspace(-1,1,wh[0]), np.linspace(-1,1,wh[1]))
    dst = np.sqrt(x*x+y*y)
    sigma = 1
    muu = 0.000
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    num_imgs = len(self.replay_dataset) / (wh[0]*wh[1])
    num_imgs = int(num_imgs)
    gauss = torch.from_numpy(gauss).unsqueeze(0).view(1, -1).permute(1, 0).squeeze(-1).repeat(num_imgs)
    return gauss