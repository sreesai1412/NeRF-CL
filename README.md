# Continual Learning for NeRF

Implicit representations (such as a NeRF) offer advantages such as lower memory requirements, and the ability to complete regions of the scene where sensor observations are missing. When a NeRF is used as a scene representation, real-time SLAM can be posed as an online continual learning problem.

Please see my presenetation on [Continual Learning for Neural Coordinate Maps](https://docs.google.com/presentation/d/1av8o65LiR_aHS-C5FX1DwWEPWYWEDRdb3qYNT-qf0HM/edit#slide=id.gdd7c03a931_0_34) for an overview of the key experiments performed and results obtained.

#### The [main](https://github.com/sreesai1412/NeRF-CL/tree/main) branch contains code for experiments on offline continual learning.

#### The [online-cl](https://github.com/sreesai1412/NeRF-CL/tree/online-cl) branch contains code for experiments on online continual learning.

## Offline Continual Learning

### Experiments: 

#### Demonstration of Catastrophic Forgetting

1. Use the “Lego” (truck) scene
   ![lego_200k_256w](https://user-images.githubusercontent.com/48653063/145554209-5c303c2a-8cdf-4b04-96e8-e964cf69ee58.gif)
2. Create two “tasks” -- Task 1 reconstructs only the right side of a truck, Task 2 reconstructs only  the left side of a front facing truck<br />
   ![Screenshot 2021-12-10 at 3 22 52 PM](https://user-images.githubusercontent.com/48653063/145554412-5fee0bbc-07d1-44be-acd3-9a82dfe75133.png)
4. Train a NeRF  on task 1 (right views of the truck) until a reasonable performance is achieved
5. Use the trained weights at this point to initialize a NeRF
6. Train on task 2 (left views of the truck) until reasonable performance (on task 2) is achieved.
7. Compute performance  drop on task 1

#### Regularization based Continual Learning Method
![Screenshot 2021-12-10 at 2 56 01 PM](https://user-images.githubusercontent.com/48653063/145554498-4ae70175-6fe0-4a14-bfc6-b7db8034bd54.png)

#### Replay based Continual Learning Method
![Screenshot 2021-12-10 at 2 57 43 PM](https://user-images.githubusercontent.com/48653063/145554506-d50168e0-6464-44ec-a59f-252d2e5289f7.png)

### Results

Results on the validation set:<br />
![Screenshot 2021-12-10 at 2 59 39 PM](https://user-images.githubusercontent.com/48653063/145554515-0a2b3765-e928-4b3d-82d4-fe0f8e72f95e.png)

Results on the test set:
Test on a held out set of 100 “right” view images:
1. Train on “right”<br />
   Test on  “right” : Mean PSNR = 27.79
   
2. Train on “right”, then train on “left”<br />
   Test on  “right” : Mean PSNR = 17.75 (- 10.04 from 1 → Catastrophic Forgetting!)

3. Train on “right”, then train on “left” (with Regularization)<br />
   Test on “right”:  Mean PSNR = 22.94  ( - 4.85 from 1)

4. Train on “right”, then train on “left” (with Replay)<br />
   Test on “right”:  Mean PSNR = 26.53  ( - 1.26 from 1)

Other Findings
1. Filling the Replay Memory using Uniformly or Randomly drawn samples from Task 1 led to higher forgetting as compared to the Top K loss based method.
2. Use of a smaller Replay Memory size also led to higher forgetting as compared to larger sizes.
3. Use of Weighted Random Sampling for sampling from the Replay memory during Task 2.<br />
   In each forward pass of Task 2:<br />
      &nbsp;&nbsp;&nbsp;&nbsp;a) sample rays from Task 2 data (left views)<br />
      &nbsp;&nbsp;&nbsp;&nbsp;b) sample rays from Replay memory using Weighted Random Sampling<br />
         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i) Weights based on loss associated with each ray in replay memory.<br />
         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii) Dynamically updated based on the most recent loss <br />
      &nbsp;&nbsp;&nbsp;&nbsp;c) train the NeRF using both samples <br />
    Here, forgetting (drop in performance (PSNR) of  Task 1 after Task 2) reduced to - 1.21
4. Use of Weighted Random Sampling for sampling from the Replay memory during Task 2.<br />
   Weights are based on a combination of:<br />
	&nbsp;&nbsp;&nbsp;&nbsp;i) Loss associated with  each ray<br /> 
	&nbsp;&nbsp;&nbsp;&nbsp;ii) Image centre weight (rays at image centre have a higher weight)<br />
   Here, forgetting (drop in performance (PSNR) of  Task 1 after Task 2) reduced to - 1.14<br />
5. Use of a very large Replay Memory size (All 172,032 Task 1 (right view) rays) and Weighted Random Sampling.<br />Here, forgetting (drop in performance (PSNR) of  Task 1 after Task 2) reduced to   - 0.72

## Data download

Download `nerf_synthetic` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

## Training

See [opt.py](opt.py) for all configurations.

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
You can monitor the training process by `tensorboard --logdir logs/` and go to `localhost:6006` in your browser.

## Online Continual Learning 
1. Use the “Lego” (truck) scene
2. Create a trajectory of 5000 frames using Blender
3. Split the trajectory into 10 chunks of 500 images each as (0 to 499), (500 to 999)........ In each chunk sample 100 test images, 400 train images.
4. 10 images representative of the each of the 10 chunks are shown below.
![Screenshot_2021-05-27_at_2 11 59_PM](https://user-images.githubusercontent.com/48653063/145557876-42201f9e-45b8-441c-83ce-6aea03319441.png)

It is important to note here that there are no task boundaries in this setting and that this is a better simulation of an online setting. Unlike the first set of experiments where the network was aware of the task boundary and used continual learning techniques specifically during Task 2, the network is not aware of the transition between any of the 10 chunks and must adapt its training strategy automatically (for instance, this includes dynamically deciding what samples to retain in the the replay memory and what to discard if the memory is exhausted during training)

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


