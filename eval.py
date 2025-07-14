

import os 
import numpy as np
import logging
import argparse

import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import dataloading as dl
import model as mdl
from utils_poses.comp_ate import compute_ATE, compute_rpe
from model.common import backup,  mse2psnr
from utils_poses.align_traj import align_ate_c2b_use_a2b
from collections import defaultdict 
import json
import yaml
from matplotlib import pyplot as plt
from model import EdgePreservingSmoothnessLoss, SmoothnessLoss
from model import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, MotionNetwork, NeuSRenderer
from model import PoseRetriever
from utils_poses.pose_refinement import PoseRefineDataset, compute_loss_and_warp_image, setup_pose_refinement, perform_pose_refinement
from train import Trainer
import cv2
from model.common import arange_pixels
import lpips as lpips_lib
import torch.nn.functional as F
from third_party import pytorch_ssim
from co3d_metric import psnr as co3d_psnr
from co3d_metric import ssim as co3d_ssim
from co3d_metric import lpips as co3d_lpips

class Evaluator(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_views = self.train_dataset['img'].N_imgs
        self.pose_retriever_train = PoseRetriever(self.n_views, init_c2w=None)
        self.pose_retriever_train.load_state_dict(torch.load(f"{self.out_dir}/models/refine_pose.pt"))
 
    def eval_optimization(self):
        # Initialize the pose_triever_test for test view
        test_idx = self.test_dataset['img'].i_test
        test_init_c2w_idx = [list(self.train_dataset['img'].i_train).index(init_idx) for init_idx in (test_idx-1)] 
        test_init_c2w = torch.stack([self.pose_retriever_train(idx) for idx in test_init_c2w_idx])
        # Initialize poses to be optimizer
        num_epoch = self.cfg["eval"]["eval_pose_epoch"]
        self.pose_retriever_test = PoseRetriever(len(test_idx), init_c2w=test_init_c2w)
        self.pose_retriever_test = self.pose_retriever_test.cuda()
        pose_optimizer = torch.optim.Adam(self.pose_retriever_test.parameters(), lr=self.cfg["eval"]["eval_pose_lr"])
        # scheduler_eval_pose = torch.optim.lr_scheduler.MultiStepLR(pose_optimizer, milestones=list(range(0, int(num_epoch), 100)), gamma=self.cfg["eval"]["eval_pose_scheduler_gamma"])    
        scheduler_eval_pose = torch.optim.lr_scheduler.MultiStepLR(pose_optimizer, milestones=list(range(0, int(num_epoch), int(num_epoch/5))), gamma=self.cfg["eval"]["eval_pose_scheduler_gamma"])    

        if not os.path.isfile(f'{self.out_dir}/models/weights/model_eval_pose.pt'):
            print("Cannot find the optimized poses for the test images ==> Optimizing it...")
            for epoch_i in tqdm(range(num_epoch)):
                L2_loss_epoch = []
                psnr_epoch = []
                for batch in self.test_loader:
                    image_idx = batch.get('img.idx')
                    # Step 1: Get gt_color, rays_o, rays_d in the world coordinate system (Nope-Nerf)
                    world_mat = self.pose_retriever_test(list(self.test_dataset['img'].i_test).index(image_idx))
                    # world_mat: world to cam
                    (img, ref_img, sampled_pixel, normalized_sampled_pixel, 
                        rays_o, rays_d, rays_d_norm, rgb_gt, camera_mat, scale_mat) = self.model.process_data(batch, world_mat, self.it, self.epoch_it)
                    # Rendering
                    near, far = self.model.near_far_from_sphere(rays_o, rays_d)
                    render_out = self.model.renderer(rays_o, rays_d, rays_d_norm, torch.tensor([self.world_time_step]).float().cuda().repeat(len(self.cfg['training']['gpu_ids'])), 
                                                near, far, background_rgb=None, cos_anneal_ratio=1.0, it=self.it, eval=False)
                    rgb_pred = render_out['color_fine']
                    # Loss computation
                    # model.rgb_weight = 1.0
                    loss_dict = self.model.compute_loss(batch, render_out['color_fine'], rgb_gt,  0.0, 0.0, 
                                                    0.0, 0.0, 0.0, 0.0, it=self.it, epoch=self.epoch_it)
                    loss = loss_dict['loss_rgb']
                    # Back propagation
                    pose_optimizer.zero_grad()
                    loss.backward()
                    pose_optimizer.step()
                    L2_loss_epoch.append(loss_dict['l2_mean'].item())
                L2_loss_mean = np.mean(L2_loss_epoch)
                opt_pose_psnr = mse2psnr(L2_loss_mean)
                scheduler_eval_pose.step()
                if epoch_i % 10 == 0: print(f'PSNR: {opt_pose_psnr}')
            torch.save(self.pose_retriever_test.state_dict(), f'{self.out_dir}/models/weights/model_eval_pose.pt')
        else:
            print("Found the optimized poses for the test images")
            self.pose_retriever_test.load_state_dict(torch.load(f'{self.out_dir}/models/weights/model_eval_pose.pt'))
            print("Load poses succesfully!")
            print(f"Note: Remove the file {f'{self.out_dir}/models/weights/model_eval_pose.pt'} if optimizing new poses for the test set is desired")

    def render_eval(self):
        self.renderer.eval()
        gt_img_list = []
        gt_depth_list = []

        rgb_pred_list = []
        depth_pred_list = []
        normal_pred_list = []
        flow_pred_list = []
        image_idx_list  = []
        depth_highest_weight_pred_list = []
        for data in tqdm(self.test_loader):
            (img, camera_mat, scale_mat, img_idx) = self.model.process_data_dict(data)
            image_idx = data.get('img.idx')
            ref_image_idx = image_idx + 1
            time_step = (image_idx / (self.total_nb_images-1) * 2 - 1).cuda()
            if len(self.model.gt_depths) != 0: 
                gt_depth = self.model.gt_depths[image_idx]
                if cfg['dataloading']['crop_size'] != 0:
                    depth_h, depth_w = gt_depth.shape
                    crop_h = cfg['dataloading']['crop_size']
                    crop_w = int(crop_h * depth_w/depth_h)
                    gt_depth = cv2.resize(gt_depth, (648,484), interpolation=cv2.INTER_NEAREST)
                    gt_depth = gt_depth[crop_h:depth_h-crop_h, crop_w:depth_w-crop_w]
            else: gt_depth = -1

            h, w = img.shape[2], img.shape[3]
            p_idx = torch.arange(h*w).float().cuda()
            p_loc, pixels = arange_pixels(resolution=(h, w))

            pixels = pixels.float().cuda()
            with torch.no_grad():
                rgb_pred = []
                depth_pred = []
                normal_pred = []
                depth_highest_weight_pred = []
                world_mat = self.pose_retriever_test(list(self.train_dataset['img'].i_test).index(image_idx))

                for i in range(0, pixels.shape[1]//1024+1):
                    pixels_i = pixels[:,i*1024:(i+1)*1024,:]
                    ray_o_i, ray_d_i, rays_d_norm_i = self.model.get_world_cameraOrigin_cameraRay(pixels_i, camera_mat, world_mat, scale_mat)
                    near, far = self.model.near_far_from_sphere(ray_o_i, ray_d_i)
                    # Model predictions
                    render_out = self.model.renderer(ray_o_i, ray_d_i, rays_d_norm_i, torch.tensor(self.world_time_step).repeat(len(cfg['training']['gpu_ids'])), near, far,  
                                                    background_rgb=None, cos_anneal_ratio=self.model.get_cos_anneal_ratio(self.it, self.anneal_end), it=self.it, eval=True)
                    rgb_pred_i = render_out['color_fine']
                    depth_pred_i = render_out['depth_pred']
                    pts = render_out['sampled_points'].view(-1,3)
                    # Get depth corresponding to the points having weight for each ray
                    weights = render_out['weights'] # [1024, 128]
                    _, max_idx = torch.max(weights, dim=1)
                    max_idx = torch.stack([torch.from_numpy(np.arange(len(weights))), max_idx.detach().cpu()])
                    depth_highest_weight = -render_out['sampled_points'][:,:,-1][tuple(max_idx)] # [1024, 128, 3]
                    depth_highest_weight = torch.clip(depth_highest_weight, 0,5)
                    depth_highest_weight_pred.append(depth_highest_weight.detach().cpu())

                    rgb_pred.append(rgb_pred_i.detach().cpu())
                    depth_pred.append(depth_pred_i.detach().cpu())
                    n_samples = self.model.renderer.module.n_samples + self.model.renderer.module.n_importance
                    normal_i = render_out['normals'] * render_out['weights'][:, :n_samples, None]
                    # normal_i = normal_i * render_out['inside_sphere'][..., None]
                    normal_i = normal_i.sum(dim=1).detach().cpu()
                    normal_pred.append(normal_i) # torch.Size([129600, 3])

                rgb_pred = torch.cat(rgb_pred, dim=0)
                depth_pred = torch.cat(depth_pred, dim=0)
                depth_highest_weight_pred = torch.cat(depth_highest_weight_pred, dim=0)
                normal_pred = torch.cat(normal_pred, dim=0) # torch.Size([129600, 3])
            
                rgb_pred = rgb_pred.view(h, w, 3).detach().cpu()
                depth_pred_out = depth_pred.view(h, w).detach().cpu()
                depth_highest_weight_pred = depth_highest_weight_pred.view(h, w).detach().cpu()

                # depth_pred_out = depth_pred_out/depth_pred_out.max()
                disp_pred_out = 1/depth_pred_out
                disp_pred_out = disp_pred_out / disp_pred_out.max()
                disp_highest_weight_pred = 1/depth_highest_weight_pred
                disp_highest_weight_pred = disp_highest_weight_pred / disp_highest_weight_pred.max()

                normal_img = normal_pred
                rot = torch.eye(3).detach().cpu() #torch.inverse(self.pose_param_net(img_idx)[:3,:3].detach().cpu())
                normal_img = (torch.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([h, w, 3, -1]) * 128 + 128).clip(0, 255)
                normal_img = (normal_img/255.0)[:,:,:,0]

                # Store the prediction
                rgb_pred_list.append(rgb_pred)
                depth_pred_list.append(depth_pred_out)
                normal_pred_list.append(normal_img)
                depth_highest_weight_pred_list.append(depth_highest_weight)
                image_idx_list.append(image_idx)
                # Store ground-truth
                gt_img_list.append(img[0].permute(1,2,0))
                gt_depth_list.append(gt_depth)
        return gt_img_list, gt_depth_list, rgb_pred_list, depth_pred_list, normal_pred_list, flow_pred_list, image_idx_list

    def image_eval(self, gt_img_list, rgb_pred_list):

        eval_psnr_list = []
        eval_ssim_list = []
        eval_lpips_list = []

        lpips_metric = lpips_lib.LPIPS(net='vgg').cuda()
        for i in range(len(rgb_pred_list)):
            pred_img = rgb_pred_list[i].float().cuda() # h w 3
            gt_img =  gt_img_list[i].float().cuda()

            pred_img = pred_img.permute(2, 0, 1)
            gt_img = gt_img.permute(2, 0, 1)

            psnr = co3d_psnr(pred_img, gt_img).mean().double()
            ssim = co3d_ssim(pred_img, gt_img).mean().double()
            lpips = co3d_lpips(pred_img, gt_img, net_type="vgg").mean().double()
            eval_psnr_list.append(psnr.item())
            eval_ssim_list.append(ssim.item())
            eval_lpips_list.append(lpips.item())

        mean_psnr = np.mean(eval_psnr_list)
        mean_ssim = np.mean(eval_ssim_list)
        mean_lpips = np.mean(eval_lpips_list)
        # print('--------------------------')
        # print('PSNR: {1:.2f}, SSIM: {2:.2f}, LPIPS {3:.2f}'.format(mean_psnr, mean_ssim, mean_lpips))    
        image_error_dict = {
            "PSNR": mean_psnr, 
            "SSIM": mean_ssim, 
            "LPIPS": mean_lpips
        }
        return image_error_dict

    def depth_eval(self, gt_depth_list, pred_depth_list, min_depth=0.1, max_depth=80):
        if np.all(np.array(gt_depth_list)==-1): return None
        depth_errors = []
        # depth_max = torch.quantile(torch.stack(gt_depth_list), 0.90)
        for i in tqdm(range(len(pred_depth_list))):
            gt_depth = torch.clone(torch.tensor(gt_depth_list)[i]).detach().cpu().numpy()
            if cfg['dataloading']['crop_size'] != 0: # Crop the gt depth of scene0079_00 in Scannet dataset
                gt_depth = gt_depth[6:-6, 8:-8]
            # gt_depth[:(gt_depth.shape[0]//4)] = max_depth + 1
            pred_depth = torch.clone(pred_depth_list[i]).detach().cpu().numpy()
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid_depth_mask = (gt_depth >= min_depth) & (gt_depth <= max_depth)
            gt_depth = gt_depth[valid_depth_mask]
            pred_depth = pred_depth[valid_depth_mask]
            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth = pred_depth * ratio
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth
            depth_errors.append(self.compute_depth_errors(gt_depth, pred_depth))
        mean_errors = np.array(depth_errors).mean(0)              
        depth_error_dict = dict(zip(["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"], mean_errors.tolist()))
        return depth_error_dict

    def pose_eval(self):
        # Pose evaluation
        pred_poses = torch.stack([self.pose_retriever_train(i) for i in range(len(self.pose_retriever_train.r))])
        pred_poses = torch.inverse(pred_poses)
        _, rpe_trans, rpe_rot, ate = self.compute_pose_error(pred_poses, self.gt_poses)
        pose_error_dict = {
            "rpe_trans": rpe_trans, 
            "rpe_rot": rpe_rot, 
            "ate": ate
        }
        return pose_error_dict

    def eval(self, store_output=True):
        self.eval_optimization()
        gt_img_list, gt_depth_list, rgb_pred_list, depth_pred_list, normal_pred_list, flow_pred_list, image_idx_list = self.render_eval()
        image_error_dict = self.image_eval(gt_img_list, rgb_pred_list)
        depth_error_dict = self.depth_eval(gt_depth_list, depth_pred_list)
        pose_error_dict = self.pose_eval()
        result = {}
        result.update(image_error_dict)
        result.update(pose_error_dict)
        if depth_error_dict is not None: result.update(depth_error_dict)
        with open(f'{self.out_dir}/results.txt', 'w') as file:
            for key, value in result.items():
                file.write(f"{key}: {value}\n")
        print(result)
        print(f"The evaluation result is stored in {self.out_dir}/results.txt")
        if store_output:
            os.makedirs(f'{self.out_dir}/extraction/images_gt', exist_ok=True)
            os.makedirs(f'{self.out_dir}/extraction/images', exist_ok=True)
            os.makedirs(f'{self.out_dir}/extraction/depths', exist_ok=True)
            os.makedirs(f'{self.out_dir}/extraction/depths_raw', exist_ok=True)
            os.makedirs(f'{self.out_dir}/extraction/normal', exist_ok=True)
            os.makedirs(f'{self.out_dir}/extraction/disparity_highest_weight', exist_ok=True)
            for i in tqdm(range(len(rgb_pred_list))):
                frame_idx = str(image_idx_list[i].item()).zfill(6)
                plt.imsave(f'{self.out_dir}/extraction/images_gt/{frame_idx}.jpg', np.clip(gt_img_list[i].detach().cpu().numpy(), 0, 1))
                plt.imsave(f'{self.out_dir}/extraction/images/{frame_idx}.jpg', np.clip(rgb_pred_list[i].detach().cpu().numpy(), 0, 1))
                plt.imsave(f'{self.out_dir}/extraction/depths/{frame_idx}.jpg', (depth_pred_list[i]).detach().cpu().numpy(),cmap='gray')
                np.savez(f'{self.out_dir}/extraction/depths_raw/depth_{frame_idx}.npz', pred=(depth_pred_list[i]).detach().cpu().numpy())
                plt.imsave(f'{self.out_dir}/extraction/normal/{frame_idx}.jpg', np.clip(normal_pred_list[i].detach().cpu().numpy(), 0, 1))


if __name__=='__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('config_path', type=str, help='Config file path')
    args = parser.parse_args()
    # Load config
    cfg = dl.load_config(args.config_path, 'configs/default.yaml')
    # Save the backup of the config file
    evaluator = Evaluator(cfg)
    evaluator.eval()

