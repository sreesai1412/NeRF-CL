# Continual Learning for NeRF

Implicit representations (such as a NeRF) offer advantages such as lower memory requirements, and the ability to complete regions of the scene where sensor observations are missing. When a NeRF is used as a scene representation, real-time SLAM can be posed as an online continual learning problem.

Please see my presenetation on [Continual Learning for Neural Coordinate Maps](https://docs.google.com/presentation/d/1av8o65LiR_aHS-C5FX1DwWEPWYWEDRdb3qYNT-qf0HM/edit#slide=id.gdd7c03a931_0_34) for an overview of the key experiments performed and results obtained.

#### This [online-cl](https://github.com/sreesai1412/NeRF-CL/tree/online-cl) branch contains code for experiments on online continual learning.

#### The [main](https://github.com/sreesai1412/NeRF-CL/tree/main) branch contains code for experiments on offline continual learning.

## Online Continual Learning
1. Use the “Lego” (truck) scene ![lego_200k_256w](https://user-images.githubusercontent.com/48653063/145560347-f1f0fba6-6bcc-4059-9f40-dd5aea23df32.gif)
2. Create a trajectory of 5000 frames using Blender
3. Split the trajectory into 10 chunks of 500 images each as (0 to 499), (500 to 999)........ In each chunk sample 100 test images, 400 train images.
4. 10 images representative of the each of the 10 chunks are shown below.
![Screenshot_2021-05-27_at_2 11 59_PM](https://user-images.githubusercontent.com/48653063/145557876-42201f9e-45b8-441c-83ce-6aea03319441.png)

It is important to note here that there are no task boundaries in this setting and that this is a better simulation of an online setting. Unlike the offline continual learning experiments where the network was aware of the task boundary and used continual learning techniques specifically during Task 2, the network is not aware of the transition between any of the 10 chunks and must adapt its training strategy automatically (for instance, this includes dynamically deciding what samples to retain in the the replay memory and what to discard if the memory is exhausted during training)

## Experiments
#### Demonstration of Catastrophic Forgetting

1. Move chunk by chunk, train on rays from 400 train images for 2000 iterations, (with batch size=1024 rays)
2. After training on each chunk, perform evaluation on test frames of current and previous chunks.

Performance on data from earlier chunks degrades over time.
![plot_psnr_no_replay](https://user-images.githubusercontent.com/48653063/145557985-b44391a5-c801-4cd3-9668-54ce87a469ff.gif)

Tracking the evolution of a test image in the 3rd chunk.
![Screenshot_2021-05-27_at_2 43 31_PM](https://user-images.githubusercontent.com/48653063/145559420-a6c87de5-fd04-4042-ac13-ed5366cf58dd.png)


#### Applying a form of Continual Learning
1. Use a replay memory whose maximum size is M = 10,24,000 rays. (half of the max buffer size as all previous experiments)
2. Move chunk by chunk. After training on chunk "i" is done,
    1. Sample 50 frames uniformly as frame 1, frame 8, frame 16, ......., frame 400
    2. For each of the 50 frames,
        1. divide frame into 4 x 4 grid 
        2. find loss in each of the 16 regions
        3. normalise losses to get L1, L2, L3,......L16 (all these losses sum to 1)
        4. Sample L(i)*2048 rays randomly from each of the 16 regions
        5. So in total 2048 rays are sampled from the given frame 
    3. Add 50 * 2048 = 102400 rays (Max Memory/10) to memory  
3. When training on chunk "i+1",  randomly sample new data from the "i+1"th chunk
4. Also randomly sample data from the replay memory which contains rays from chunks 0, 1, ....i.
5. Include loss due to both samples.
6. Train for 2000 iterations per chunk.

Performance on data from earlier chunks is maintained.
![plot_psnr_er_grid_investig](https://user-images.githubusercontent.com/48653063/145558645-4a468b14-5b7b-44d9-9ac1-92e08a649937.gif) 

Tracking the evolution of a test image in the 3rd chunk.
![Screenshot_2021-05-31_at_2 29 57_PM](https://user-images.githubusercontent.com/48653063/145559160-8411d996-f5b1-4268-a7dd-670fcaa1b6ed.png)


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
