from os.path import split
import torch
from torch.utils.data import Dataset
import json
import operator
import numpy as np
import os
import sys
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *

class BlenderDatasetLargeOnline(Dataset):
    def __init__(self, hparams, root_dir, split='train', img_wh=(800, 800)):
        self.hparams = hparams

        self.root_dir = root_dir
        self.mode = 'training'

        if split=='infer_train':
            split = 'train'
            self.mode = 'infer'

        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh

        self.define_transforms()

        self.read_meta()

        self.read_trajectory()

        if self.split == 'val':
          self.read_frames()
        
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            

    def read_trajectory(self):
        if self.split == 'train':
            self.chunks = []
            num_test_in_each_chunk = self.hparams.chunk_size - self.hparams.num_train_in_each_chunk
            for i in range(self.hparams.num_chunks):
                mask = torch.ones((self.hparams.chunk_size,))
                self.test_inds = torch.multinomial(mask, num_test_in_each_chunk)
                mask[self.test_inds] = 0
                self.train_inds = torch.nonzero(mask).squeeze(-1)

                self.train_inds = (self.train_inds + i*self.hparams.chunk_size).tolist()
                self.test_inds = (self.test_inds + i*self.hparams.chunk_size).tolist()

                dictionary = {'train':self.train_inds, 'test':self.test_inds}
                self.chunks.append(dictionary)

        elif self.split == 'val':
            val_inds = [0, 28, 49, 91, 50, 33, 64, 100, 48, 65, 44, 98, 19, 16, 85, 31, 38, 69,
                    72, 14, 39, 55, 67, 59, 74, 9, 27, 24, 82, 11, 8, 71, 17, 94, 1, 58, 87, 
                    84, 90, 99, 88, 21, 60, 7, 43, 86, 77, 46, 30, 10, 93, 3, 15, 23, 78, 18, 
                    34, 79, 96, 51, 81, 20, 45, 54, 57, 6, 75, 42, 37, 76, 97, 80, 4, 22, 66, 
                    63, 26, 5, 83, 53, 95, 73, 2, 40, 70, 92, 52, 61, 89, 41, 35, 29, 47, 56, 
                    25, 13, 68, 62, 12, 32, 36]
            self.trajectory = val_inds

    def read_frames(self):
        self.meta_traj = []
        self.trajectory = [i-1 for i in self.trajectory]
        self.trajectory.pop(-1)
        for i in self.trajectory:
            self.meta_traj.append(self.meta['frames'][i])

    def read_chunk_from_disk(self, chunk_num):
        print("Reading Chunk " , chunk_num)
        
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []

        for i in self.chunks[chunk_num]['train']:
            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, h, w)
            img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

            self.all_rays += [torch.cat([rays_o, rays_d, 
                                                self.near*torch.ones_like(rays_o[:, :1]),
                                                self.far*torch.ones_like(rays_o[:, :1])],
                                                1)] # (h*w, 8)
            
        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_inds = torch.linspace(0, (self.hparams.num_train_in_each_chunk * self.img_wh[0] * self.img_wh[1]), (self.hparams.num_train_in_each_chunk * self.img_wh[0] * self.img_wh[1]))
        self.all_inds = self.all_inds + chunk_num*(self.hparams.num_train_in_each_chunk * self.img_wh[0] * self.img_wh[1])

        print("Reading from disk done")

    def get_test_frame(self, epoch, frame_num):

        idx = self.chunks[epoch]['test'][frame_num]
        frame = self.meta['frames'][idx]
        c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

        img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img) # (4, H, W)
        img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
        img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

        rays_o, rays_d = get_rays(self.directions, c2w)

        rays = torch.cat([rays_o, rays_d, 
                            self.near*torch.ones_like(rays_o[:, :1]),
                            self.far*torch.ones_like(rays_o[:, :1])],
                            1) # (H*W, 8)

        sample = {'rays': rays.cuda(),
                  'rgbs': img.cuda()}

        return sample

    def get_frame_by_idx(self, idx):
        frame = self.meta['frames'][idx]
        c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

        img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img) # (4, H, W)
        img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
        img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

        rays_o, rays_d = get_rays(self.directions, c2w)

        rays = torch.cat([rays_o, rays_d, 
                            self.near*torch.ones_like(rays_o[:, :1]),
                            self.far*torch.ones_like(rays_o[:, :1])],
                            1) # (H*W, 8)

        sample = {'rays': rays,
                  'rgbs': img}

        return sample

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train' and self.mode != 'infer':
            return len(self.all_rays)
        
        if self.split == 'val':
            return 1
        if self.split == 'test':
            return len(self.trajectory)

        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train' and self.mode != 'infer': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'idx' : self.all_inds[idx]}
        
        else:
            frame = self.meta_traj[idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample