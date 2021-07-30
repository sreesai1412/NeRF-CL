# NeRF-CL

## Training

See [opt.py](opt.py) for details of all configuration parameters.

### Vanilla NeRF training on the large trajectory

```
python train_large.py \
   --dataset_name blender_large \
   --root_dir '/content/drive/MyDrive/Continual NeRF/data' \
   --N_importance 64 --img_wh 64 64 --noise_std 0 \
   --num_epochs 400 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --num_frames 5000 \
   --chunk_size 1 \
   --num_iters_per_chunk 1 \
   --val_after_n_epochs 1 \
   --exp_seed 42 \
   --monitor 'val/loss' \
   --exp_name exp_train_vanilla_5K \
```

### Online NeRF training with no Continual Learning technique  

Experiment to demonstrate Catastrophic Forgetting

```
python train_large.py \
   --dataset_name blender_large_online \
   --root_dir '/content/drive/MyDrive/Continual NeRF/data' \
   --N_importance 64 --img_wh 64 64 --noise_std 0 \
   --num_epochs 10 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --online_mode \
   --num_frames 5000 \
   --chunk_size 500 \
   --num_iters_per_chunk 2000 \
   --val_after_n_epochs 1 \
   --exp_seed 42 \
   --monitor 'val/loss' \
   --exp_name exp_online \
   --save_plots
```

### Online NeRF training with Experience Replay

```
python train_large.py \
   --dataset_name blender_large_online \
   --root_dir '/content/drive/MyDrive/Continual NeRF/data' \
   --N_importance 64 --img_wh 64 64 --noise_std 0 \
   --num_epochs 10 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --online_mode \
   --use_replay_buf \
   --num_frames 5000 \
   --chunk_size 500 \
   --num_train_in_each_chunk 400 \
   --online_buffer_size 2048000 \
   --online_buffer_fill_mode 'uniform' \
   --online_buffer_sample_mode 'random' \
   --num_iters_per_chunk 2000 \
   --val_after_n_epochs 1 \
   --exp_seed 42 \
   --monitor 'val/loss' \
   --exp_name exp_online_replay \
   --save_plots
```

### Online NeRF training with the A-GEM technique

```
python train_gem.py \
   --dataset_name blender_large_online \
   --root_dir '/content/drive/MyDrive/Continual NeRF/data' \
   --N_importance 64 --img_wh 64 64 --noise_std 0 \
   --num_epochs 10 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --online_mode \
   --use_replay_buf \
   --num_frames 5000 \
   --chunk_size 500 \
   --num_train_in_each_chunk 400 \
   --online_buffer_size 2048000 \
   --online_buffer_fill_mode 'uniform' \
   --online_buffer_sample_mode 'random' \
   --num_iters_per_chunk 2000 \
   --val_after_n_epochs 1 \
   --exp_seed 42 \
   --monitor 'val/loss' \
   --exp_name exp_online_gem \
   --save_plots
```
