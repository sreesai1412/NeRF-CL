# NeRF-CL

## Data download

Download `nerf_synthetic` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

## Training model

### Offline CL

#### Stage 1

```
python train.py \
   --dataset_name blender \
   --root_dir '/content/drive/MyDrive/nerf_synthetic/lego' \
   --N_importance 64 --img_wh 64 64 --noise_std 0 \
   --num_epochs 128 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --continual_mode \
   --train_view 'right' \
   --exp_seed 42 \
   --monitor 'val/loss_r' \
   --val_after_n_epochs 1 \
   --exp_name exp_train_r
```

#### Stage 2

```
python train.py \
   --dataset_name blender \
   --root_dir '/content/drive/MyDrive/nerf_synthetic/lego' \
   --N_importance 64 --img_wh 64 64 --noise_std 0 \
   --num_epochs 256 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --continual_mode \
   --train_view 'left' \
   --use_replay_buf \
   --buffer_size 20480 \
   --buffer_fill_mode 'topk' \
   --buffer_sample_mode 'weighted_random' \
   --exploration_ratio 0.2 \
   --dynamic_buffer_update \
   --exp_seed 42 \
   --monitor 'val/loss_l' \
   --val_after_n_epochs 1 \
   --exp_name exp_train_l \
   --ckpt_path '/content/drive/MyDrive/NERF/nerf_clean/ckpts/exp_train_r/epoch=128.ckpt'
```

#### Evaluation

On test set.\
This will create folder `results/{dataset_name}/{scene_name}/{save_dir_name}` and run inference on the test data, finally create a gif out of them.
```
python eval.py \
   --root_dir '/content/drive/MyDrive/nerf_synthetic/lego' \
   --dataset_name blender --scene_name lego \
   --img_wh 64 64 --N_importance 64 \
   --split 'test' \
   --test_view 'right' \
   --ckpt_path '/content/drive/MyDrive/NERF/nerf_clean/ckpts/exp_train_r/epoch=128.ckpt' \
   --save_dir_name 'results'
```

Running inference on train set to generate soft tragets and Stage 1 losses.\
This will create folders `{root_dir}/soft_targets` and `{root_dir}/stage1_losses`
```
python eval.py \
   --root_dir '/content/drive/MyDrive/nerf_synthetic/lego' \
   --dataset_name blender --scene_name lego \
   --img_wh 64 64 --N_importance 64 \
   --split 'infer_train' \
   --test_view 'NA' \
   --ckpt_path '/content/drive/MyDrive/NERF/nerf_clean/ckpts/exp_train_r/epoch=128.ckpt' \
   --save_dir_name 'results'
```

See [opt.py](opt.py) for all configurations.

You can monitor the training process by `tensorboard --logdir logs/` and go to `localhost:6006` in your browser.

### Online CL 

```
python train_online.py \
   --dataset_name blender_online \
   --root_dir '/content/drive/MyDrive/nerf_synthetic/lego' \
   --N_importance 64 --img_wh 64 64 --noise_std 0 \
   --num_epochs 50 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --online_cl_mode \
   --num_frames 100 \
   --use_replay_buf \
   --chunk_size 2 \
   --num_iters_per_chunk 32 \
   --exp_seed 42 \
   --monitor 'val/loss' \
   --val_after_n_epochs 50 \
   --exp_name exp_online
```   

### Vanilla NeRF implementation is borrowed from [here](https://github.com/kwea123/nerf_pl)
