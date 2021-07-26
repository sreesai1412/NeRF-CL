import torch
import os,sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

from losses import loss_dict

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff','blender_new1','blender_new2','blender_large', 'blender_large_online'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test', required=True,
                        choices=['test', 'infer_train'], 
                        help='infer_train runs inference on the train set one image at a time for getting soft tragets and stage1 losses')
 
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
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--save_dir_name', type=str, required=True,
                        help='directory to save predictions')


    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    parser.add_argument('--test_view', type=str, choices=['right', 'left', 'NA'],
                        help='which views from the test set to test for')

    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--expname', type=str, required=True)

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=False)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    # loss_fn = nn.MSELoss(reduction='none')
    loss_fn = loss_dict['mse']()

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](args, **kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    nerf_fine = NeRF()
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()

    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]

    colors = ['blue', 'black', 'red', 'yellow', 'green', 'violet', 'cyan', 'pink', 'peru', 'midnightblue']
    color_plots = []
    for i in range(10):
        for j in range(200):
              color_plots.append(colors[i])

    train_inds = torch.load('/content/drive/MyDrive/NERF/NeRF-CL-dev3/train_inds_saving.pt')
    test_inds = torch.load('/content/drive/MyDrive/NERF/NeRF-CL-dev3/test_inds_saving.pt')

    plots_dir = f'plots_train/{args.expname}/{args.epoch}'
    os.makedirs(plots_dir , exist_ok=True)

    print("\nSaving Plots")
    epoch = args.epoch
    seq_ids = [i for i in range(2000)]
    losses = [0 for i in range(2000)]
    psnrs = [0 for i in range(2000)]
      
    for i in range(0,3):
      for j in tqdm(range(200)):
        sample = dataset.get_frame_by_idx(train_inds[(400*i) + (2*j)])
        rays = sample['rays'].cuda()
        rgbs = sample['rgbs'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back)

        losses[(200*i) + (j)]= loss_fn(results, rgbs).detach().item()
        psnrs[(200*i) + (j)] = metrics.psnr(results['rgb_fine'], rgbs).detach().item()

        if j==100:
          print(metrics.psnr(results['rgb_fine'], rgbs).detach().item())
          img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
          img_pred_ = (img_pred*255).astype(np.uint8)
          imageio.imwrite(os.path.join(plots_dir, f'r_{(400*i)+(2*j):d}.png'), img_pred_)

          sample = dataset.get_frame_by_idx(test_inds[(100*i) + 50])
          rays = sample['rays'].cuda()
          rgbs = sample['rgbs'].cuda()
          results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back)
          img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
          img_pred_ = (img_pred*255).astype(np.uint8)
          imageio.imwrite(os.path.join(plots_dir, f'r_test_{(100*i)+50:d}.png'), img_pred_)

        del rays, rgbs, results

        # print(losses)
        
    import matplotlib.pyplot as plt
    plt.clf()
    plt.ylim(0,0.5)
    plt.bar(seq_ids, losses, color=color_plots)
    plt.xlabel('Frame IDs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(plots_dir,'plotloss_'+str(epoch)+'.png'))

    plt.clf()
    plt.ylim(0,50)
    plt.bar(seq_ids, psnrs, color=color_plots)
    plt.xlabel('Frame IDs')
    plt.ylabel('PSNR')
    plt.savefig(os.path.join(plots_dir,'plotpsnr_'+str(epoch)+'.png'))
    print("Saved plot")

    



    

    # imgs = []
    # psnrs = []
    # dir_name = f'results/{args.dataset_name}/{args.scene_name}/{args.save_dir_name}'
    # os.makedirs(dir_name, exist_ok=True)

    # if args.split == 'infer_train':
    #     soft_targets_dir = f'{args.root_dir}/soft_targets'
    #     os.makedirs(soft_targets_dir , exist_ok=True)

    #     losses_dir = f'{args.root_dir}/stage1_losses'
    #     os.makedirs(losses_dir, exist_ok=True)

    # indxs = torch.load('/content/drive/MyDrive/NERF/NeRF-CL-dev2/test_inds.pt')

    # for i in tqdm(range(len(indxs))):
    #     sample = dataset.get_test_selected(i)
    #     rays = sample['rays'].cuda()
    #     rgbs = sample['rgbs'].cuda()
    #     results = batched_inference(models, embeddings, rays,
    #                                 args.N_samples, args.N_importance, args.use_disp,
    #                                 args.chunk,
    #                                 dataset.white_back)

    #     img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
        
    #     if args.save_depth:
    #         depth_pred = results['depth_fine'].view(h, w).cpu().numpy()
    #         depth_pred = np.nan_to_num(depth_pred)
    #         if args.depth_format == 'pfm':
    #             save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
    #         else:
    #             with open(f'depth_{i:03d}', 'wb') as f:
    #                 f.write(depth_pred.tobytes())

    #     img_pred_ = (img_pred*255).astype(np.uint8)
    #     imgs += [img_pred_]
        
    #     if args.split == 'infer_train':
    #         imageio.imwrite(os.path.join(soft_targets_dir, f'r_{i:d}.png'), img_pred_)
    #     # else:
    #     #     imageio.imwrite(os.path.join(dir_name, f'r_{i:d}.png'), img_pred_)


    #     if 'rgbs' in sample:
    #         rgbs = sample['rgbs']
    #         img_gt = rgbs.view(h, w, 3)
    #         psnrs += [metrics.psnr(img_gt, img_pred).item()]
            
    #         if args.split == 'infer_train': 
    #             loss = loss_fn(torch.from_numpy(img_pred), img_gt).mean(dim=2)
    #             loss = loss.clone().detach()
    #             torch.save(loss, os.path.join(losses_dir, f'loss_{i:d}.pt'))
        
    # # imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)
    
    # if psnrs:
    #     mean_psnr = np.mean(psnrs)
    #     print(f'Mean PSNR : {mean_psnr:.2f}')