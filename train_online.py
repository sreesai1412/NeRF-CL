from frame_buffer import FrameBuffer
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

from buffer_online import Buffer
from frame_buffer import FrameBuffer

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

        if self.hparams.online_cl_mode:
            self.frame_buf = FrameBuffer(self.hparams, batch_size=self.hparams.batch_size, epoch=self.hparams.resume)

        if self.hparams.use_replay_buf:
            self.replay_buf = Buffer(self.hparams, batch_size=self.hparams.batch_size, epoch=self.hparams.resume)

        if hparams.save_plots:
            self.plots_dir = f'./plots/{self.hparams.exp_name}'
            os.makedirs(self.plots_dir , exist_ok=True)

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
        
        self.train_dataset = dataset(self.hparams, split='train', **kwargs)
        self.val_dataset = dataset(self.hparams, split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        loader = DataLoader(self.frame_buf,
                          shuffle=False,
                          num_workers=2,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)
        return loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=2,
                          batch_size=1,
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        self.log('lr', get_learning_rate(self.optimizer))
        num_iters_per_chunk = self.hparams.num_iters_per_chunk

        step = self.global_step

        if step % num_iters_per_chunk == 0:
          self.frame_buf.fill_chunk(int(step / num_iters_per_chunk))
        
        new_batch = self.frame_buf.get_batch()

        rays, rgbs = self.decode_batch(new_batch)
        results = self(rays)
        loss = self.loss(results, rgbs)

        if self.hparams.use_replay_buf:
            inds = new_batch['idx']
            losses = self.loss_vector(results, rgbs).mean(dim=1)
            self.replay_buf.update_buffer(inds.detach().cpu(),losses.detach().cpu())

            if self.current_epoch > 0:
                replay_batch = self.replay_buf.get_batch(self.current_epoch, self.hparams.exploration_ratio)        
                rays_replay, replay_rgbs = self.decode_batch(replay_batch)
                replay_inds = replay_batch['idx']
                results_replay = self.forward(rays_replay)
                replay_losses = self.loss_vector(results_replay, replay_rgbs).mean(dim=1)

                replay_loss_total = replay_losses.mean()
                loss += (replay_loss_total)

                self.replay_buf.update_buffer(replay_inds.detach().cpu(),replay_losses.detach().cpu())
                self.log('train/replay_loss', replay_loss_total, prog_bar=True)


        self.log('train/loss', loss, prog_bar=True)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            self.log('train/psnr', psnr_, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self, outs):
      if self.hparams.save_plots:
          epoch = self.current_epoch
          seq_ids = [i for i in range(self.hparams.num_frames)]
          losses = [0 for i in range(self.hparams.num_frames)]
          psnrs = [0 for i in range(self.hparams.num_frames)]
        
          for i in range(0,epoch+1):
            for j in range(self.hparams.chunk_size):
              frame = self.frame_buf.get_frame_batch(i,j)
              rays, rgbs = self.decode_batch(frame)
              rays = rays.squeeze() 
              rgbs = rgbs.squeeze() 
              results = self(rays) 
              losses[(self.hparams.chunk_size*i) + j]=self.loss(results, rgbs).detach().item()
              psnrs[(self.hparams.chunk_size*i) + j]=psnr(results['rgb_fine'], rgbs).detach().item()
              del rays, rgbs, results

          print(losses)
          
          import matplotlib.pyplot as plt
          plt.clf()
          plt.ylim(0,0.25)
          plt.bar(seq_ids, losses)
          plt.xlabel('Frame IDs')
          plt.ylabel('Loss')
          plt.savefig(os.path.join(self.plots_dir,'plotloss_'+str(epoch)+'.png'))

          plt.clf()
          plt.ylim(0,40)
          plt.bar(seq_ids, psnrs)
          plt.xlabel('Frame IDs')
          plt.ylabel('PSNR')
          plt.savefig(os.path.join(self.plots_dir,'plotpsnr_'+str(epoch)+'.png'))
          print("Saved plot")

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        self.log('val/loss', self.loss(results, rgbs),prog_bar=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        self.log('val/psnr', psnr(results[f'rgb_{typ}'], rgbs),prog_bar=True)
    
        # if batch_nb == 0:
        #     W, H = self.hparams.img_wh
        #     img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
        #     img = img.permute(2, 0, 1) # (3, H, W)
        #     img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        #     depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
        #     stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
        #     self.logger.experiment.add_images('val/GT_pred_depth',
        #                                        stack, self.global_step)


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
                      check_val_every_n_epoch=hparams.val_after_n_epochs,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=hparams.num_gpus==1)

    trainer.fit(system)