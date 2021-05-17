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

class BlenderDatasetOnline(Dataset):
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

        self.read_frames()
        
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
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
        train_inds = [0, 41, 42, 67, 75, 26, 27, 64, 61, 1, 11, 81, 50, 79, 51, 74, 39, 57, 38, 60, 15, 12, 82, 
                    47, 34, 78, 4, 22, 56, 37, 59, 62, 88, 18, 77, 25, 31, 7, 49, 30, 14, 48, 99, 58, 21, 91, 
                    43, 80, 5, 66, 3, 33, 16, 44, 92, 36, 86, 90, 93, 24, 71, 6, 8, 83, 87, 29, 46, 73, 2, 32, 
                    53, 97, 84, 28, 68, 10, 52, 85, 69, 9, 20, 19, 54, 63, 95, 35, 13, 17, 40, 55, 72, 96, 70, 
                    94, 65, 23, 89, 45, 98, 76]
        
        val_inds = [0, 28, 49, 91, 50, 33, 64, 100, 48, 65, 44, 98, 19, 16, 85, 31, 38, 69,
                    72, 14, 39, 55, 67, 59, 74, 9, 27, 24, 82, 11, 8, 71, 17, 94, 1, 58, 87, 
                    84, 90, 99, 88, 21, 60, 7, 43, 86, 77, 46, 30, 10, 93, 3, 15, 23, 78, 18, 
                    34, 79, 96, 51, 81, 20, 45, 54, 57, 6, 75, 42, 37, 76, 97, 80, 4, 22, 66, 
                    63, 26, 5, 83, 53, 95, 73, 2, 40, 70, 92, 52, 61, 89, 41, 35, 29, 47, 56, 
                    25, 13, 68, 62, 12, 32, 36]

        test_inds = [0, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81,
                    80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 
                    57, 56, 55, 54, 154, 153, 155, 53, 152, 156, 52, 157, 158, 159, 160, 161, 162, 163, 164, 165, 
                    166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 
                    185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 150, 151, 
                    149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 
                    130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 
                    111, 110, 109, 108, 107, 106, 105, 104, 103, 102]

        if self.split == 'train':
            self.trajectory = train_inds
        elif self.split == 'val':
            self.trajectory = val_inds
        elif self.split == 'test':
            self.trajectory = test_inds

    def read_frames(self):
        if self.split == 'train' and self.mode!='infer':
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []

            for i in self.trajectory:
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

        else:
            self.meta_traj = []
            self.trajectory = [i-1 for i in self.trajectory]
            self.trajectory.pop(-1)
            # print(self.trajectory)
            for i in self.trajectory:
                self.meta_traj.append(self.meta['frames'][i])


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train' and self.mode != 'infer':
            return len(self.all_rays)
        
        if self.split == 'val':
            return len(self.trajectory)
        if self.split == 'test':
            return len(self.trajectory)

        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train' and self.mode != 'infer': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'idx' : idx}
        
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