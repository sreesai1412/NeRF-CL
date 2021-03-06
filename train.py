import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger

import pytorch_lightning as pl

from buffer import Buffer

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()
        self.loss_vector = loss_dict[hparams.loss_type](reduction='none')

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

        if self.hparams.use_replay_buf:
            self.replay_buf = Buffer(self.hparams, buffer_size=self.hparams.buffer_size, 
                                    batch_size=self.hparams.batch_size)
            
            self.replay_buf.fill_buffer()


    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        
        self.train_dataset = dataset(self.hparams, split='train', view = self.hparams.train_view, **kwargs)
        self.val_dataset = dataset(self.hparams, split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=2,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)
  
        if self.hparams.use_replay_buf and self.hparams.buffer_sample_mode == 'random':
            replay_loader = DataLoader(self.replay_buf,
                                        shuffle=True,
                                        num_workers=2,
                                        batch_size=self.hparams.batch_size,
                                        pin_memory=True)
            return {'new': loader, 'replay':replay_loader}

        return loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=2,
                          batch_size=1,
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        self.log('lr', get_learning_rate(self.optimizer))

        if self.hparams.use_replay_buf:
            if self.hparams.buffer_sample_mode == 'random':
                new_batch = batch['new']
                replay_batch = batch['replay']
            elif self.hparams.buffer_sample_mode == 'weighted_random':
                new_batch = batch
                replay_batch = self.replay_buf.get_batch(self.hparams.exploration_ratio)
            elif self.hparams.buffer_sample_mode == 'topk':
                new_batch = batch
                replay_batch = self.replay_buf.get_topk_batch()
        else:
          new_batch = batch
          
        rays, rgbs = self.decode_batch(new_batch)
        results = self(rays)
        loss = self.loss(results, rgbs)

        if self.hparams.continual_mode:

            if self.hparams.distillation:
                soft_rgbs = new_batch['soft_rgbs']
                distill_loss = self.loss(results, soft_rgbs)
                loss += distill_loss
                self.log('train/distill_loss', distill_loss, prog_bar=True)

            if self.hparams.use_replay_buf:
                replay_rays, replay_rgbs = self.decode_batch(replay_batch)
                replay_results = self(replay_rays)
                if self.hparams.use_soft_targets_for_replay:
                    replay_rgbs = replay_batch['soft_rgbs']
                
                if self.hparams.dynamic_buffer_update:
                    replay_inds = replay_batch['idx']
                    replay_losses = self.loss_vector(replay_results, replay_rgbs).mean(dim=1)
                    replay_loss = replay_losses.mean()
                    self.replay_buf.update_buffer(replay_inds.detach().cpu(),replay_losses.detach().cpu())
                else:
                    replay_loss = self.loss(replay_results, replay_rgbs)
                
                loss += replay_loss
                self.log('train/replay_loss', replay_loss, prog_bar=True)


        self.log('train/loss', loss, prog_bar=True)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch['r'])
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        loss_r = self.loss(results, rgbs)
        self.log('val/loss_r', loss_r,prog_bar=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        self.log('val/psnr_r', psnr(results[f'rgb_{typ}'], rgbs),prog_bar=True)

        rays, rgbs = self.decode_batch(batch['l'])
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        loss_l = self.loss(results, rgbs)
        self.log('val/loss_l', loss_l, prog_bar=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        self.log('val/psnr_l', psnr(results[f'rgb_{typ}'], rgbs),prog_bar=True)
    
        # if batch_nb == 0:
        #     W, H = self.hparams.img_wh
        #     img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
        #     img = img.permute(2, 0, 1) # (3, H, W)
        #     img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        #     depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
        #     stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
        #     self.logger.experiment.add_images('val/GT_pred_depth',
        #                                        stack, self.global_step)

        self.log('val/loss', (loss_r + loss_l)/2, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts()
    pl.utilities.seed.seed_everything(seed=hparams.exp_seed)

    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'ckpts/{hparams.exp_name}'),
                                          filename='{epoch:d}',
                                          monitor=hparams.monitor,
                                          mode='min',
                                          save_top_k=5)

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=hparams.num_gpus==1)

    trainer.fit(system)