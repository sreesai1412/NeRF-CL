import json
import operator
import os
import sys
from os.path import split

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import *


class BlenderDatasetLarge(Dataset):
    def __init__(self, hparams, root_dir, split='train', img_wh=(800, 800)):
        self.hparams = hparams

        self.root_dir = root_dir
        self.mode = 'training'

        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh

        # WARNING: Call the following 'make_train_test_sets()' function to generate the train and test split indices. 
        # Call ONLY IF saved tensor files containing the indices are not already present.
        # It is essential that the function is called after noting the experiment seed set in the opt.py file.
        # Changing the seed will change the train and test splits generated.
        
        # Uncomment below line after reading the warning above.
        # self.make_train_test_sets()
        
        self.define_transforms()

        self.read_meta()

        self.split_train_test()

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
            

    def make_train_test_sets(self):
        self.trajectory = torch.linspace(0,self.hparams.num_frames,self.hparams.num_frames).tolist()
        test_inds_saving = []
        train_inds_saving = []

        num_test_frames_in_each_chunk = self.hparams.chunk_size - self.hparams.num_train_in_each_chunk
        
        for i in range(self.hparams.num_chunks):
            mask = torch.ones((self.hparams.chunk_size,))
            self.test_inds = torch.multinomial(mask, num_test_frames_in_each_chunk)
            mask[self.test_inds] = 0
            self.train_inds = torch.nonzero(mask).squeeze(-1)

            self.train_inds = (self.train_inds + i*self.hparams.chunk_size).tolist()
            self.test_inds = (self.test_inds + i*self.hparams.chunk_size).tolist()

            test_inds_saving.append(self.test_inds)
            train_inds_saving.append(self.train_inds)

            test_inds_saving = torch.cat(test_inds_saving, dim=0)
            torch.save(test_inds_saving, 'test_inds_saving.pt')
            
            train_inds_saving = torch.cat(train_inds_saving, dim=0)
            torch.save(train_inds_saving, 'train_inds_saving.pt')

    def split_train_test(self):
        # Set paths to the saved tensor files containing the test and train indices 
        self.test_inds  = torch.load('/content/drive/MyDrive/NERF/NeRF-CL-dev2/test_inds_saving.pt').tolist()
        self.train_inds = torch.load('/content/drive/MyDrive/NERF/NeRF-CL-dev2/train_inds_saving.pt').tolist()

        self.meta_train = []
        self.meta_test = []

        for i in self.train_inds:
            self.meta_train.append(self.meta['frames'][i])

        for i in self.test_inds:
            self.meta_test.append(self.meta['frames'][i])

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.meta_train)
        
        if self.split == 'val':
            return 1
        if self.split == 'test':
            return len(self.meta_test)

        return len(self.meta['frames'])

    def get_test_selected(self,idx):
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

    def __getitem__(self, idx):
        frame = self.meta_train[idx] if self.split=='train' else self.meta_test[idx]
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
