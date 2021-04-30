import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import sys

from .ray_utils import *

class BlenderDatasetNew1(Dataset):
    def __init__(self, root_dir, split='train',subsplit=None, img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        self.subsplit =subsplit

        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
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

        if self.split == 'test' or self.split == 'val':
          self.meta_r = []
          self.meta_l = []
          for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.FloatTensor(pose)
                
                if c2w[0][3] > 0:
                  self.meta_r.append(frame)
                else:
                  self.meta_l.append(frame)
          print("Separated all " + self.split + " frames")
            
        if self.subsplit == 'train_right': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.soft_rgbs = []  # for using predictions of earlier tasks inplace of targets

            for i, frame in enumerate(self.meta['frames']):
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)
                
                if c2w[0][3] > 0:

                  image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                  self.image_paths += [image_path]
                  img = Image.open(image_path)
                  img = img.resize(self.img_wh, Image.LANCZOS)
                  img = self.transform(img) # (4, h, w)
                  img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                  img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                  self.all_rgbs += [img]

                  image_path = os.path.join(self.root_dir, './soft_train2/r_'+str(i)+'.png')
                  img = Image.open(image_path)
                  img = img.resize(self.img_wh, Image.LANCZOS)
                  img = self.transform(img) # (3, h, w)
                  img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                  self.soft_rgbs += [img]
                  
                  rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                  self.all_rays += [torch.cat([rays_o, rays_d, 
                                              self.near*torch.ones_like(rays_o[:, :1]),
                                              self.far*torch.ones_like(rays_o[:, :1])],
                                              1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.soft_rgbs = torch.cat(self.soft_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            print("Gathered all right view information")

        if self.subsplit == 'train_left': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.soft_rgbs = []  # for using predictions of earlier tasks inplace of targets

            for i, frame in enumerate(self.meta['frames']):
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)
                
                if c2w[0][3] < 0:

                  image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                  self.image_paths += [image_path]
                  img = Image.open(image_path)
                  img = img.resize(self.img_wh, Image.LANCZOS)
                  img = self.transform(img) # (4, h, w)
                  img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                  img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                  self.all_rgbs += [img]

                  image_path = os.path.join(self.root_dir, './soft_train2/r_'+str(i)+'.png')
                  img = Image.open(image_path)
                  img = img.resize(self.img_wh, Image.LANCZOS)
                  img = self.transform(img) # (3, h, w)
                  img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                  self.soft_rgbs += [img]
                  
                  rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                  self.all_rays += [torch.cat([rays_o, rays_d, 
                                              self.near*torch.ones_like(rays_o[:, :1]),
                                              self.far*torch.ones_like(rays_o[:, :1])],
                                              1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.soft_rgbs = torch.cat(self.soft_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            print("Gathered all left view information")

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
          if self.subsplit == 'train_all_1by1': 
            # Don't use cached rays (this subssplit can be used for inference on train set)
            return len(self.meta['frames'])
          else:
            return len(self.all_rays)

        if self.split == 'val':
            return 8 

        if self.split == 'test':
            if(self.subsplit) == 'test_right':
                return len(self.meta_r)
            elif(self.subsplit) == 'test_left':
                return len(self.meta_l)

        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train' and self.subsplit!='train_all_1by1': 
          # use data in the buffers
          sample = {'rays': self.all_rays[idx],
                    'rgbs': self.all_rgbs[idx],
                    'soft_rgbs': self.soft_rgbs[idx]}

        else: 
          # create data for each image separately
          if self.subsplit=='train_all_1by1':
            frame = self.meta['frames'][idx]
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

          elif self.split=='test':
            frame = self.meta_r[idx] if self.subsplit=='test_right' else self.meta_l[idx]
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

          elif self.split=='val':
            sample = {}

            frame = self.meta_r[idx]
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

            sample1 = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}
            
            sample['r']= sample1

            frame = self.meta_l[idx]
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

            sample2 = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}
            
            sample['l']= sample2

        return sample