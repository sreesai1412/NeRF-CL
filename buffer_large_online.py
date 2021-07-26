import os
import sys
from operator import index

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from datasets import dataset_dict


class Buffer(Dataset):
  """
  Implements the experience replay buffer"
  """
  @torch.no_grad()
  def __init__(self, hparams, buffer_size, batch_size, epoch):
    self.hparams = hparams
    self.max_buffer_size = buffer_size
    self.batch_size = batch_size

    wh = tuple(self.hparams.img_wh)
    self.chunk_size_in_rays = self.hparams.chunk_size * wh[0] * wh[1]

    self.current_size = 0
    self.buffer_size = int(self.max_buffer_size/self.hparams.num_chunks)

    self.mem_per_chunk = int(self.max_buffer_size/self.hparams.num_chunks)

    self.data_samples = []
    self.ray_data = torch.zeros(self.max_buffer_size, 8).cuda()
    self.rgb_data = torch.zeros(self.max_buffer_size, 3).cuda()
    self.loss_data = torch.zeros(self.max_buffer_size,).cuda()
    self.indxs = torch.zeros(self.max_buffer_size,).cuda()
    self.inserted_at = torch.zeros(self.max_buffer_size,) - 1

    self.writer = SummaryWriter(log_dir = os.path.join('./runs/', self.hparams.exp_name))

  
  def __getitem__(self, index):
        return self.data_samples[index]

  def __len__(self):
        return len(self.data_samples)

  @torch.no_grad()
  def add_data(self, rays, rgbs, inds, losses, step):
    for i in range(inds.size(0)):
        ins= self.current_size + i
        self.ray_data[ins].copy_(rays[i])
        self.rgb_data[ins].copy_(rgbs[i])
        self.loss_data[ins].copy_(losses[i])
        self.indxs[ins].copy_(inds[i])
        self.inserted_at[ins].copy_(step)
    self.current_size += inds.size(0)

  @torch.no_grad()
  def expand_buffer_size(self, epoch):
      self.buffer_size = int((epoch+1)*(self.max_buffer_size/self.hparams.num_chunks))

  @torch.no_grad()
  def update_buffer(self, rays, rgbs, replay_inds, replay_losses, step):

    step = torch.tensor(step)

    if self.current_size < self.buffer_size:
        self.add_data(rays, rgbs, replay_inds, replay_losses, step)
        self.writer.add_scalar('num_replacements', 0, step)

    else:

        losses = self.loss_data[0:self.buffer_size]

        if self.hparams.online_buffer_fill_mode == 'highest_loss':
          _, select_inds = torch.sort(torch.cat([losses, replay_losses]), descending=True)
        if self.hparams.online_buffer_fill_mode == 'lowest_loss':
          _, select_inds = torch.sort(torch.cat([losses, replay_losses]))

        new = select_inds[0:self.buffer_size][select_inds[0:self.buffer_size]>(self.buffer_size-1)].tolist()
        old = select_inds[self.buffer_size:][select_inds[self.buffer_size:]<=(self.buffer_size-1)].tolist()

        self.writer.add_scalar('num_replacements', len(new), step)

        for i in range(len(new)):
            self.ray_data[old[i]].copy_(rays[new[i]-self.buffer_size])
            self.rgb_data[old[i]].copy_(rgbs[new[i]-self.buffer_size])
            self.loss_data[old[i]].copy_(replay_losses[new[i]-self.buffer_size])
            self.indxs[old[i]].copy_(replay_inds[new[i]-self.buffer_size])
            self.inserted_at[old[i]].copy_(step)

    if step % 1 ==0:
      ages = self.inserted_at[self.inserted_at != -1]
      ages = torch.abs(ages - step)

      median = torch.median(ages).item()
      self.writer.add_scalar('median_age', median, step)

  @torch.no_grad()
  def update_losses(self, inds, replay_losses):
      for i in range(inds.size(0)):
        self.loss_data[inds[i]] = replay_losses[i]
  
  @torch.no_grad()
  def get_batch(self, exploration_ratio):

    if self.hparams.online_buffer_sample_mode == 'weighted_random':
        losses = (self.loss_data)[0:self.buffer_size-self.mem_per_chunk]
        denom = torch.sum(losses)
        weights = (losses/denom).squeeze(-1)

        n_explore = int(self.batch_size * exploration_ratio)
        n_weigted = self.batch_size - n_explore

        weighted_inds = torch.multinomial(weights, n_weigted, replacement=False)

        inds = weighted_inds
        
        if exploration_ratio > 0.0:
          index_mask = torch.ones((self.buffer_size-self.mem_per_chunk)).cuda()
          index_mask[weighted_inds] = 0
          summat = torch.sum(index_mask)
          index_mask = index_mask/summat 
          explore_inds = torch.multinomial(index_mask, n_explore, replacement=False)
          inds = torch.cat([weighted_inds, explore_inds])

    if self.hparams.online_buffer_sample_mode == 'random':
        inds = torch.randint(0, self.buffer_size-self.mem_per_chunk, (self.batch_size,))

    current_rays = []
    current_rgbs = []
    current_indices = []
    for i in inds:
      current_rays.append(self.ray_data[i])
      current_rgbs.append(self.rgb_data[i])
      current_indices.append(i)

    batch = {}
    batch['rays'] = torch.stack(current_rays)
    batch['rgbs'] = torch.stack(current_rgbs)
    batch['idx']  = torch.stack(current_indices)

    return batch

@torch.no_grad()
def fill_buffer(self, frame_buf, losses):
    print("Filling buffer")
    for i in range(self.mem_per_chunk):
      sample = frame_buf.current_chunk[i*8]
      ins= self.current_size + i
      self.ray_data[ins].copy_(sample['rays'])
      self.rgb_data[ins].copy_(sample['rgbs'])
      self.indxs[ins].copy_(sample['idx'])
      self.loss_data[ins].copy_(losses[i])
    self.current_size += self.mem_per_chunk
    print("Buffer fill done")

