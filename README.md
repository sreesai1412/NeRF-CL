# Continual Learning for NeRF

Implicit representations (such as a NeRF) offer advantages such as lower memory requirements, and the ability to complete regions of the scene where sensor observations are missing. When a NeRF is used as a scene representation, real-time SLAM can be posed as an online continual learning problem.

Please see my presenetation on [Continual Learning for Neural Coordinate Maps](https://docs.google.com/presentation/d/1av8o65LiR_aHS-C5FX1DwWEPWYWEDRdb3qYNT-qf0HM/edit#slide=id.gdd7c03a931_0_34) for an overview of the key experiments performed and results obtained.

##### The main branch contains code for experiments on offline continual learning.

##### The [online-cl](https://github.com/sreesai1412/NeRF-CL/tree/online-cl) branch contains code for experiments on online continual learning.

## Offline Continual Learning

### Experiments: 

#### Demonstration of Catastrophic Forgetting

1. Use the “Lego” (truck) scene
2. Create two “tasks” -- Task 1 reconstructs only the right side of a truck, Task 2 reconstructs only  the left side of a front facing truck
3. Train a NeRF  on task 1 (right views of the truck) until a reasonable performance is achieved
4. Use the trained weights at this point to initialize a NeRF
5. Train on task 2 (left views of the truck) until reasonable performance (on task 2) is achieved.
6. Compute performance  drop on task 1

#### Regularization based Continual Learning Method

#### Replay based Continual Learning Method

### Results

Results on the validation set:

Results on the test set:
Test on a held out set of 100 “right” view images:
1. Train on “right”: 
   Test on  “right” : Mean PSNR = 27.79

2. Train on “right”, then train on “left”: 
   Test on  “right” : Mean PSNR = 17.75 (- 10.04 from 1 → Catastrophic Forgetting!)

3. Train on “right”, then train on “left” (with Regularization):
   Test on “right”:  Mean PSNR = 22.94  ( - 4.85 from 1)

4. Train on “right”, then train on “left” (with Replay):
   Test on “right”:  Mean PSNR = 26.53  ( - 1.26 from 1)

### Other Findings

1. Filling the Replay Memory using Uniformly or Randomly drawn samples from Task 1 led to higher forgetting as compared to the Top K loss based method.
2. Use of a smaller Replay Memory size also led to higher forgetting as compared to larger sizes.
3. Use of Weighted Random Sampling for sampling from the Replay memory during Task 2.
 
   In each forward pass of Task 2:

      a) sample rays from Task 2 data (left views)
      b) sample rays from Replay memory using Weighted Random Sampling
         i) Weights based on loss associated with each ray in replay memory.
         ii) Dynamically updated based on the most recent loss 
      c) train the NeRF using both samples 

	Here, forgetting (drop in performance (PSNR) of  Task 1 after Task 2) reduced to - 1.21

## Data download

Download `nerf_synthetic` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

## Training

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

On the **test set**.\
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

Running inference on the **train set** to generate **soft tragets** and **stage 1 losses**.\
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
