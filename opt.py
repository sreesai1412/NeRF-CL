import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'blender_online'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
        
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse'],
                        help='loss to use')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    ################### params for Continual Learning ################################
    parser.add_argument('--continual_mode', action='store_true', default=False)

    parser.add_argument('--train_view', type=str, choices=['right', 'left'], 
                        help='which training views to train on')

    parser.add_argument('--distillation', action='store_true', default=False, 
                        help='whether to use distsillation as a CL technique in task 2')

    parser.add_argument('--use_replay_buf', action='store_true', 
                        help='whether to use a replay memory as a CL technique in task 2')

    parser.add_argument('--buffer_size', type=int, default=20480, 
                        help ='size of replay memory if a replay memory is used')

    parser.add_argument('--buffer_fill_mode', type=str,
                        choices=['topk', 'uniform', 'random'], 
                        help='how to fill the replay memory with task 1 examples at the beginning of task 2')

    parser.add_argument('--buffer_sample_mode', type=str,
                        choices=['random', 'weighted_random', 'topk'],
                        help='how to sample from replay memory during each iteration of task 2')

    parser.add_argument('--use_soft_targets_for_replay', action='store_true', 
                        help='whether to use soft targets for samples drawn from replay memory')

    parser.add_argument('--exploration_ratio', default=0.0, type=float, 
                        help='value between 0 to 1 indicating the fraction of the batch of\
                             replay samples drawn uniformly in each iteration of task 2. [used only when buffer_sample_mode==weighted_random]')

    parser.add_argument('--dynamic_buffer_update', action='store_true',
                        help='whether to dynamically update weights for replay samples\
                            based on most recent loss obtained in each iteration of task 2')

    parser.add_argument('--use_imgcentre_weight', action='store_true',
                        help='whether to use image centre weight for sampling replay rays')

    parser.add_argument('--imgcentre_ratio', type=float, default=0.3,
                        help='value between 0 to 1, for combining imgcentre weight and loss based weights')
    ###################################################################

    ######################## Online CL args ##########################
    parser.add_argument('--online_cl_mode', action='store_true')

    parser.add_argument('--num_frames', type=int,
                       help='number of frames in online trajectory')
    
    parser.add_argument('--chunk_size', type=int,
                       help='size of chunk (in number of frames) to train for at a given time')

    parser.add_argument('--num_iters_per_chunk', type=int,
                       help='number of iterations to train on a single chunk')

    parser.add_argument('--save_plots', action='store_true',
                       help='whether to save plots after each epoch for making a GIF')

    parser.add_argument('--resume', type=int, default=0, 
                        help='frame in the trajectory from which to resume, in case training is stopped abruptly')
    ######################################################################

    parser.add_argument('--val_after_n_epochs', type=int, required=True,
                       help='number of epochs after which to run validation')    
    
    parser.add_argument('--exp_name', type=str, required=True,
                        help='experiment name')

    parser.add_argument('--exp_seed', type=int, required=True, help='seeding everything for repeatability')

    parser.add_argument('--monitor', type=str, default='val/loss', 
                        choices=['val/loss', 'val/loss_l', 'val/loss_r'], required=True)

    return parser.parse_args()
