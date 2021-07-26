import os, sys
import torch
from tqdm import tqdm
from opt import get_opts
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


from frame_loader import FrameLoader
from frame_buffer import FrameBuffer
from buffer_large_online import Buffer

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

        if not self.hparams.online_mode:
            self.frame_loader = FrameLoader(self.hparams, batch_size=self.hparams.batch_size)

        if self.hparams.online_mode:
            self.frame_buf = FrameBuffer(self.hparams, batch_size=self.hparams.batch_size, epoch=self.hparams.resume)
            
            colors = ['blue', 'black', 'red', 'yellow', 'green', 'violet', 'cyan', 'pink', 'peru', 'midnightblue']
            self.color_plots = []
            for i in range(self.hparams.num_chunks):
                for j in range(self.hparams.chunk_size - self.hparams.num_train_in_each_chunk):
                    self.color_plots.append(colors[i])

        if self.hparams.use_replay_buf:
            self.replay_buf = Buffer(self.hparams, buffer_size=self.hparams.online_buffer_size, batch_size=self.hparams.batch_size, epoch=self.hparams.resume)

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
        
        if not self.hparams.online_mode:
            self.train_dataset = dataset(self.hparams, split='train', **kwargs)
        else:
            self.train_dataset = dataset(self.hparams, split='val', **kwargs)

        self.val_dataset = dataset(self.hparams, split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        loader = DataLoader(self.frame_loader,
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

        if not self.hparams.online_mode:
            num_total_train = self.hparams.num_chunks * self.hparams.num_train_in_each_chunk
            if step % num_total_train == 0:
                self.frame_loader.shuffle_dataset()
            if step % num_iters_per_chunk == 0:
                self.frame_loader.fill_frame(int(step / num_iters_per_chunk) % num_total_train)
            
            new_batch = self.frame_loader.get_batch()

        if self.hparams.online_mode:
            if step % num_iters_per_chunk == 0:
                if self.hparams.use_replay_buf and step!=0:
                    self.replay_buf.expand_buffer_size(self.current_epoch)
                    if self.hparams.online_buffer_fill_mode == 'uniform':
                        self.eval_and_fill_buffer()
                    
                self.frame_buf.fill_chunk(int(step / num_iters_per_chunk))
            
            new_batch = self.frame_buf.get_batch()


        rays, rgbs = self.decode_batch(new_batch)
        results = self(rays)
        loss = self.loss(results, rgbs)

        if self.hparams.use_replay_buf:
            if self.hparams.online_buffer_fill_mode == ('highest_loss' or 'lowest_loss'):
                with torch.no_grad():
                    inds = new_batch['idx']
                    losses = self.loss_vector(results, rgbs).mean(dim=1)
                    self.replay_buf.update_buffer(rays.detach(), rgbs.detach(), inds.detach(), losses.detach(), step)

            if self.current_epoch > 0:
                replay_batch = self.replay_buf.get_batch(self.hparams.exploration_ratio)        
                rays_replay, replay_rgbs = self.decode_batch(replay_batch)
                results_replay = self.forward(rays_replay)
                replay_loss_total = self.loss(results_replay, replay_rgbs)
                loss += (replay_loss_total)
                    
                with torch.no_grad():
                    replay_inds = replay_batch['idx']
                    replay_losses = self.loss_vector(results_replay, replay_rgbs).mean(dim=1)
                    self.replay_buf.update_losses(replay_inds.detach(),replay_losses.detach())
                    self.log('train/replay_loss', replay_loss_total, prog_bar=True)


        self.log('train/loss', loss, prog_bar=True)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def eval_and_fill_buffer(self):
      with torch.no_grad():
        print("\nEvaluating rays to find losses")
        losses = torch.zeros((204800,))
        
        for i in tqdm(range(50)):
          sample = self.frame_buf.get_filling_batch(i)
          rays = sample['rays']
          rgbs = sample['rgbs']
          losses[(i*4096): (i*4096+4096)] = self.loss_vector(self.forward(rays), rgbs).mean(dim=1)

        self.replay_buf.fill_buffer(self.frame_buf, losses)

    def on_train_epoch_end(self, outs):
        with torch.no_grad():
            if self.hparams.online_mode:

                num_test_in_each_chunk = self.hparams.chunk_size - self.hparams.num_train_in_each_chunk
                num_total_test = self.hparams.num_chunks * num_test_in_each_chunk
                
                print("\nTesting")
                loss_test = 0
                psnr_test = 0
                typ = 'fine' #if 'rgb_fine' in results else 'coarse'
                for i in tqdm(range(num_test_in_each_chunk)):
                    batch = self.frame_buf.get_test_frame(self.current_epoch, i)
                    rays, rgbs = self.decode_batch(batch)
                    rays = rays.squeeze() # (H*W, 3)
                    rgbs = rgbs.squeeze() # (H*W, 3)
                    results = self(rays)
                    loss_test += self.loss(results, rgbs).detach().item()
                    psnr_test += psnr(results[f'rgb_{typ}'], rgbs).detach().item()
                
                self.log('test/loss', loss_test/100, prog_bar=True)
                self.log('test/psnr', psnr_test/100, prog_bar=True)

                print("\nSaving Plots")
                if self.hparams.save_plots:
                    epoch = self.current_epoch
                    seq_ids = [i for i in range(num_total_test)]
                    losses = [0 for i in range(num_total_test)]
                    psnrs = [0 for i in range(num_total_test)]
                
                    for i in range(0,epoch+1):
                        for j in tqdm(range(num_test_in_each_chunk)):
                            frame = self.frame_buf.get_test_frame(i,j)
                            rays, rgbs = self.decode_batch(frame)
                            rays = rays.squeeze() 
                            rgbs = rgbs.squeeze() 
                            results = self(rays) 
                            losses[(num_test_in_each_chunk*i) + j]=self.loss(results, rgbs).detach().item()
                            psnrs[(num_test_in_each_chunk*i) + j]=psnr(results['rgb_fine'], rgbs).detach().item()
                            del rays, rgbs, results
                    
                    torch.save(torch.tensor(losses), os.path.join(self.plots_dir, f'losslist_{epoch:d}.pt'))
                    torch.save(torch.tensor(psnrs), os.path.join(self.plots_dir, f'psnrlist_{epoch:d}.pt'))
                    
                    import matplotlib.pyplot as plt
                    plt.clf()
                    plt.ylim(0,0.5)
                    plt.bar(seq_ids, losses, color=self.color_plots)
                    plt.xlabel('Frame IDs')
                    plt.ylabel('Loss')
                    plt.savefig(os.path.join(self.plots_dir,'plotloss_'+str(epoch)+'.png'))

                    plt.clf()
                    plt.ylim(0,50)
                    plt.bar(seq_ids, psnrs, color=self.color_plots)
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
    

if __name__ == '__main__':
    hparams = get_opts()
    pl.utilities.seed.seed_everything(seed=hparams.exp_seed)

    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'ckpts/{hparams.exp_name}'),
                                          filename='{epoch:d}',
                                          monitor=hparams.monitor,
                                          mode='min',
                                          save_top_k=10)

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
