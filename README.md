# NeRF-CL

### Most of the code is taken from [here](https://github.com/kwea123/nerf_pl)

### Data download

Download `nerf_llff_data.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

### Training model

Run (example)
```
python train.py \
   --dataset_name llff \
   --root_dir $LLFF_DIR \
   --N_importance 64 --img_wh 504 378 \
   --num_epochs 30 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name exp
```

These parameters are chosen to best mimic the training settings in the original repo. See [opt.py](opt.py) for all configurations.

You can monitor the training process by `tensorboard --logdir logs/` and go to `localhost:6006` in your browser.

### Testing

Use [eval.py](eval.py) to create the whole sequence of moving views.
E.g.
```
python eval.py \
   --root_dir $BLENDER \
   --dataset_name blender --scene_name lego \
   --img_wh 400 400 --N_importance 64 --ckpt_path $CKPT_PATH
```
It will create folder `results/{dataset_name}/{scene_name}` and run inference on all test data, finally create a gif out of them.
