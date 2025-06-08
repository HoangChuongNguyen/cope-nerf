
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

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dir = cfg['training']['out_dir']
        self.device = "cuda"
        seed = cfg['training']['seed']
        # 0. Create folder for visualization
        self.render_path = os.path.join(self.out_dir, 'rendering')
        os.makedirs(self.render_path, exist_ok=True)
        # 1. Define network/optimizer
        self.mode = cfg['training']['mode']
        params_to_train = []
        self.nerf_outside = NeRF(**cfg['neus_nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**cfg['neus_sdf_network']).to(self.device)
        if cfg['training']['pretrained_sdf_path'] is not None: 
            self.sdf_network.load_state_dict(torch.load(cfg['training']['pretrained_sdf_path']))
            print("Load SDF network from pretrained model succesfully.")
        self.motion_network = MotionNetwork(**cfg['motion_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**cfg['neus_variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**cfg['neus_rendering_network']).to(self.device)
        self.renderer = NeuSRenderer(self.nerf_outside,
                                self.sdf_network,
                                self.deviation_network,
                                self.color_network,
                                self.motion_network,
                                **cfg['neus_renderer'])
        self.renderer = self.renderer.to(device=self.device)
        self.renderer = torch.nn.DataParallel(self.renderer, device_ids=cfg['training']['gpu_ids'])
        self.renderer = self.renderer.to(device=self.device)
        # Optimizer
        lr = cfg['training']['learning_rate']
        motion_lr = cfg['training']['pose_learning_rate']
        self.optimizer = torch.optim.Adam(list(self.sdf_network.parameters())+list(self.deviation_network.parameters())+list(self.color_network.parameters()), lr=lr)
        self.motion_optimizer = torch.optim.Adam(self.motion_network.parameters(), lr=motion_lr)
        self.anneal_end = cfg['neus_training']['neus_anneal_end']
        # 2. Define dataset
        self.train_loader, self.train_dataset = dl.get_dataloader(cfg, mode='train', shuffle=cfg['dataloading']['shuffle'], seed=seed)
        self.test_loader, self.test_dataset = dl.get_dataloader(cfg, mode='eval', shuffle=cfg['dataloading']['shuffle'], seed=seed)
        self.iter_test = iter(self.test_loader)
        self.n_views = self.train_dataset['img'].N_imgs
        self.total_nb_images = len([image_file for image_file in os.listdir(f"{cfg['dataloading']['path']}/{cfg['dataloading']['scene'][0]}/images") if image_file.endswith('jpg') ])
        self.gt_poses = self.train_dataset['img'].c2ws.float().to(self.device) 
        # 3. Define loss params
        self.sdf_weight = cfg['training']['sdf_weight']
        self.end_sdf_weight_increase_iteration = cfg['training']['end_sdf_weight_increase_iteration']
        self.sdf_consistency_weight = cfg['training']['sdf_consistency_weight']
        self.end_consistency_weight_increase_iteration = cfg['training']['end_consistency_weight_increase_iteration']
        self.end_smooth_epoch = cfg['training']['end_smooth_epoch']
        self.patch_size = cfg['training']['patch_size'] 
        self.compute_smoothness_loss = SmoothnessLoss(self.patch_size)
        self.compute_edge_smoothness_loss = EdgePreservingSmoothnessLoss(self.patch_size)
        # # 4. Define training params
        self.nb_sample_timestep = cfg['training']['nb_sample_timestep'] 
        self.nerf_lr_warm_up_it = cfg['training']['nb_warm_up_it']
        self.start_query_world_epoch = cfg['training']['start_query_world_epoch']
        self.freeze_camera_pose_period = cfg['training']['freeze_camera_pose_period']
        self.coarse_to_fine_scheduler = cfg['training']['coarse_to_fine_scheduler']
        # Get the world_idx
        if cfg['training']['world_idx'] == 'mid': self.world_cam_idx = self.total_nb_images // 2
        else: self.world_cam_idx = cfg['training']['world_idx']
        while not (self.world_cam_idx in self.train_dataset['img'].i_train): 
            print(f"The current world_cam_idx {self.world_cam_idx} is not in the training set ==> Decrease the world_cam_idx by 1")
            self.world_cam_idx=self.world_cam_idx-1
        assert self.world_cam_idx in self.train_dataset['img'].i_train, 'World cam index is not in the list of training indices'
        self.world_time_step = (self.world_cam_idx / (self.total_nb_images-1) * 2 - 1)
        print("World_cam_idx: ", self.world_cam_idx)
        # # 5. Define logging/checkpoints params
        self.checkpoint_io = mdl.CheckpointIO(self.out_dir, model=self.renderer, optimizer=self.optimizer, motion_optimizer=self.motion_optimizer)
        self.load_dir = cfg['training']['load_dir']
        self.logger = SummaryWriter(os.path.join(self.out_dir, 'logs'))
        self.print_every = cfg['training']['print_every']
        self.visualize_every = cfg['training']['depth_bound_update_every_milestones'][0]
        self.checkpoint_every = cfg['training']['checkpoint_every']
        self.eval_pose_every = cfg['training']['eval_pose_every']
        # # 6. Define a final model and lr scheduler
        self.model = mdl.Trainer(self.renderer, self.optimizer, self.motion_optimizer, cfg=cfg['training'], 
                            device=self.device, total_nb_images=self.total_nb_images, 
                            gt_depths=self.train_dataset['img'].gt_depths, cfg_all=self.cfg, logger=self.logger)
        # 7. Load current checkpoint
        try: 
            load_dict = self.checkpoint_io.load(self.load_dir, load_model_only=cfg['training']['load_ckpt_model_only'])
            print("Checkpoint found ==> Continue training from the existing checkpoint")
        except FileExistsError: 
            print("No checkpoint found ==> Train from scratch")
            load_dict = dict()
        self.epoch_it = load_dict.get('epoch_it', -1)
        self.it = load_dict.get('it', -1)
        # self.load_depth_range = load_dict.get('depth_range', cfg["rendering"]["depth_range"])
        # 8. Define lr scheduler
        self.scheduling_start = cfg['training']['scheduling_start']
        self.scheduling_epoch = cfg['training']['scheduling_epoch'] 
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                milestones=list(range(self.scheduling_start, self.scheduling_epoch+self.scheduling_start, 10)),
                                                gamma=cfg['training']['scheduler_gamma'], last_epoch=self.epoch_it)
        self.motion_scheduler = optim.lr_scheduler.MultiStepLR(self.motion_optimizer, 
                                                milestones=list(range(self.scheduling_start, self.scheduling_epoch+self.scheduling_start, 10)),
                                                gamma=cfg['training']['motion_scheduler_gamma'], last_epoch=self.epoch_it)

    def print_log(self, out, loss_dict, initializing_pose=False):
        for l, num in loss_dict.items():
            self.logger.add_scalar('loss/' + l, num.detach().cpu(), self.it)
        if not initializing_pose:
            weights_key = ['weight_sum', 'weight_max', 'weights', 'weight_inside', 'weight_outside']
            for key in weights_key:
                self.logger.add_scalar(f'weight/{key}', torch.mean(out[key]).item(), self.it)
            stats_keys = ['s_val', 'cdf_fine']
            for key in stats_keys:
                self.logger.add_scalar(f'stats/{key}', torch.mean(out[key]).item(), self.it)
        self.logger.add_scalar('lr/model', self.optimizer.param_groups[0]['lr'], self.it)
        self.logger.add_scalar('lr/motion_net', self.motion_optimizer.param_groups[0]['lr'], self.it)

    def print_log_epoch(self, epoch_loss_dict):
        for key in epoch_loss_dict: 
            if 'loss' in key or 'l2' in key:
                self.logger.add_scalar(f'loss_epoch/{key}', np.mean(epoch_loss_dict[key]), self.epoch_it)

    def visualize_log(self, data, world_mat, query_cam_idx, vis_idx):
        out_render_path = os.path.join(self.render_path, '%04d_vis' % self.it)
        os.makedirs(out_render_path, exist_ok=True)
        # Assuming that `render_visdata` is a method of self.model
        render_pkl = self.model.render_visdata(
            data,
            world_mat,
            query_cam_idx,
            self.cfg['training']['vis_resolution'], 
            self.it,
            self.anneal_end,
            out_render_path,
            idx=vis_idx)
        # Optionally, you could log or return render_pkl if needed

    def save_checkpoint(self):
        print('Saving checkpoint')
        self.checkpoint_io.save(
            'model.pt', lastest_checkpoint=True,
            epoch_it=self.epoch_it, it=self.it,
            depth_range=self.model.depth_range)
        self.checkpoint_io.save(
            'model.pt', lastest_checkpoint=False,
            epoch_it=self.epoch_it, it=self.it,
            depth_range=self.model.depth_range)

    def compute_pose_error(self, pred_poses, gt_poses):
        # Align the gt and predicted poses
        c2ws_est_aligned = align_ate_c2b_use_a2b(pred_poses.detach().cpu(), gt_poses.detach().cpu())  # (N, 4, 4)
        # Compute ATE
        ate = compute_ATE(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        trans_errors, rot_errors = compute_rpe(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        rpe_trans, rpe_rot = np.mean(trans_errors), np.mean(rot_errors)
        rpe_trans = rpe_trans * 100
        rpe_rot = rpe_rot * 180 / np.pi
        return c2ws_est_aligned, rpe_trans, rpe_rot, ate

    def compute_depth_errors(self, gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def eval_pose(self, pred_poses, gt_poses):
        # Note: This method uses self.epoch_it for logging
        _, rpe_trans, rpe_rot, ate = self.compute_pose_error(pred_poses, gt_poses)
        eval_dict = {
            'ate_trans': ate,
            'rpe_trans': rpe_trans,
            'rpe_rot': rpe_rot
        }
        for l, num in eval_dict.items():
            self.logger.add_scalar('eval/' + l, num, self.epoch_it)

    def pose_evaluation(self):
        with torch.no_grad():
            # Compute the relative camera pose
            _, c2c = self.motion_network.compute_relative_camera_pose(
                target_cam_idx=0,
                final_ref_cam_idx=self.total_nb_images - 1,
                total_nb_images=self.total_nb_images,
                nb_sample_timestep=self.nb_sample_timestep)
            w2c = self.motion_network.compute_w2c_mappings(c2c)[self.train_dataset['img'].i_train, :, :]
            pred_poses = torch.inverse(w2c)
            aligned_pred_pose, rpe_trans, rpe_rot, ate = self.compute_pose_error(pred_poses, self.gt_poses)
            self.logger.add_scalar('eval_pose/rpe_trans', rpe_trans, self.epoch_it)
            self.logger.add_scalar('eval_pose/rpe_rot', rpe_rot, self.epoch_it)
            self.logger.add_scalar('eval_pose/ate', ate, self.epoch_it)
            return aligned_pred_pose, rpe_trans, rpe_rot, ate

    def vis_pose_2d(self, aligned_pred_pose):
        # Visualize the cameras in the XY plane
        figure = plt.figure()
        plt.scatter(aligned_pred_pose[:, 0, -1].detach().cpu(), aligned_pred_pose[:, 1, -1].detach().cpu())
        plt.scatter(self.gt_poses[:, 0, -1].detach().cpu(), self.gt_poses[:, 1, -1].detach().cpu())
        plt.legend(['Pred', 'Gt'])
        plt.title(f'Epoch: {self.epoch_it}')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        os.makedirs(f'{self.out_dir}/poses_vis', exist_ok=True)
        plt.savefig(f'{self.out_dir}/poses_vis/{self.epoch_it}.jpg', bbox_inches='tight')
        plt.close()

    def warp_pixel(self, src_frame, uv, normalize_pix=True):
        _, _, height, width = src_frame.shape
        warp_x, warp_y = uv[:, 0], uv[:, 1]
        if normalize_pix:
            warp_x = warp_x / ((width - 1) / 2) - 1
            warp_y = warp_y / ((height - 1) / 2) - 1
        coord = torch.stack([warp_x, warp_y], dim=-1)
        warped_image = torch.nn.functional.grid_sample(
            src_frame, coord, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_image

    def scalar_annealing(self, it, start_anneal, end_anneal, start_weight, end_weight):
        it = np.clip(it, start_anneal, end_anneal)
        return start_weight + (end_weight - start_weight) * np.clip(
            (it - start_anneal) / (end_anneal - start_anneal + 1e-10), 0, 1)

    def loss_weight_scalar_annealing(self, it):
        # Linearly increase the weight of sdf consistency loss
        if self.end_consistency_weight_increase_iteration != -1:
            sdf_consistency_weight_it = self.scalar_annealing(
                it, 0.0, self.end_consistency_weight_increase_iteration,
                self.sdf_consistency_weight[0], self.sdf_consistency_weight[1])
            self.model.sdf_consistency_weight = sdf_consistency_weight_it
        # Linearly increase the weight of sdf loss
        if self.end_sdf_weight_increase_iteration != -1:
            sdf_weight_it = self.scalar_annealing(
                it, 0.0, self.end_sdf_weight_increase_iteration,
                self.sdf_weight[0], self.sdf_weight[1])
            self.model.sdf_weight = sdf_weight_it

    def neus_warmup_learning_rate(self, it):
        learning_factor = np.clip(it / self.nerf_lr_warm_up_it, 0, 1)
        for g in self.optimizer.param_groups:
            g['lr'] = self.cfg['training']['learning_rate'] * learning_factor
        # Optionally, update motion optimizer learning rate:
        # for g in self.motion_optimizer.param_groups:
        #     g['lr'] = self.cfg['training']['pose_learning_rate'] * learning_factor

    def update_training_resolution(self, current_epoch):
        # Reinitialize the dataset with the corresponding scale from the coarse-to-fine scheduler
        for s, interval in self.coarse_to_fine_scheduler.items():
            if current_epoch >= interval[0] and current_epoch <= interval[1]:
                break
        new_resolution = [
            self.cfg['training']['original_resolution'][0] // s,
            self.cfg['training']['original_resolution'][1] // s
        ]
        self.cfg['training']['resolution'] = new_resolution
        # Retrieve the new dataloader and dataset based on the updated resolution
        train_loader, train_dataset = dl.get_dataloader(
            self.cfg, mode=self.mode, shuffle=self.cfg['dataloading']['shuffle'], seed=self.cfg['training']['seed'])
        return train_loader, train_dataset, new_resolution, s

    def render_train_views(self):
        anneal_end = self.cfg['neus_training']['neus_anneal_end']
        os.makedirs(f'{self.out_dir}/extraction_stage1/depths', exist_ok=True)
        os.makedirs(f'{self.out_dir}/extraction_stage1/images', exist_ok=True)
        resolution = np.array(self.cfg['training']['resolution'])
        with torch.no_grad():
            for o, batch in enumerate(tqdm(self.train_loader)):
                image_idx = query_cam_idx = self.train_dataset['img'].i_train[o]
                render_pkl = self.model.render_visdata(
                    batch, None, query_cam_idx, resolution, self.it,
                    anneal_end, idx=image_idx, render_only=True)
                render_depth = render_pkl['render_depth']
                render_image = render_pkl['render_image']
                np.savez(f'{self.out_dir}/extraction_stage1/depths/depth_{str(image_idx).zfill(6)}.npz', pred=render_depth)
                plt.imsave(f'{self.out_dir}/extraction_stage1/depths/{str(image_idx).zfill(6)}.jpg',
                           render_depth / np.max(render_depth), cmap='gray')
                plt.imsave(f'{self.out_dir}/extraction_stage1/images/{str(image_idx).zfill(6)}.png',
                           np.clip(render_image, 0, 1))

    def prepare_training(self):
        self.current_epoch = self.epoch_it+1 if self.epoch_it != -1 else 0
        self.current_it = self.it if self.it != -1 else -1
        # if self.current_it != -1: self.model.depth_range = load_depth_range
        self.query_in_canonical_space = self.current_epoch >= self.start_query_world_epoch
        # For coarse-to-fine training
        if len(self.coarse_to_fine_scheduler) != 0:
            self.update_resolution_milestones = [interval[0] for interval in self.coarse_to_fine_scheduler.values()]
            self.train_loader, self.train_dataset, current_resolution, s = self.update_training_resolution(self.current_epoch)
        else:
            self.s = 1
            self.coarse_to_fine_scheduler = {1: [0,int(10e10)]}
            current_resolution = cfg['training']['resolution']
        if self.current_epoch > self.end_smooth_epoch:
            self.model.smoothness_weight = cfg['training']['smoothness_weight'][1]
            self.model.edge_aware_smoothness_weight = cfg['training']['edge_aware_smoothness_weight'][1]
            self.patch_size = 1
            print(f'At epoch {self.current_epoch}, set smooth_weight={self.model.edge_aware_smoothness_weight} and patch_size={self.patch_size}')
        # Load the pre-computed pose
        if self.epoch_it > self.start_query_world_epoch: 
            print("Load the pre-computed camera poses")
            self.pose_retriever = PoseRetriever(len(self.train_dataset['img'].i_train), learn_R=False, learn_t=False, init_c2w=None)
            self.pose_retriever = self.pose_retriever.to(device=self.device)
            self.pose_retriever.load_state_dict(torch.load(f"./{out_dir}/models/refine_pose.pt"))
        return current_resolution

    def train(self):
        current_resolution = self.prepare_training()
        print(f"Continue training at epoch = {self.current_epoch}, it = {self.it}")
        print(f"Current depth range: {self.model.depth_range}")
        print(f"Current training resolution: {current_resolution}")
        print(f"Query SDF in {'*World' if self.query_in_canonical_space else '*Local Camera'} Coordinate System*")
        print()

        for epoch_it in tqdm(range(self.current_epoch, self.scheduling_start + self.scheduling_epoch)):
            epoch_loss_dict = defaultdict(lambda: [])
            self.epoch_it = epoch_it
            # Drop learning rate by half
            if epoch_it in self.cfg['training']['lr_drop_half_epoch']:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] / 2
                self.scheduler._last_lr = [g['lr'] for g in self.optimizer.param_groups]
                for g in self.motion_optimizer.param_groups:
                    g['lr'] = g['lr'] / 2
                self.motion_scheduler._last_lr = [g['lr'] for g in self.motion_optimizer.param_groups]
                print(f"At epoch {epoch_it}, drop lr by half.")

            # Update the training resolution according to the scheduler
            if len(self.coarse_to_fine_scheduler) >= 2 and epoch_it in self.update_resolution_milestones:
                self.train_loader, self.train_dataset, new_resolution, self.s = self.update_training_resolution(epoch_it)
                print(f"Change the training resolution to: {new_resolution}")

            # Stage 2: Switch to querying SDF in the world coordinate system (optionally refine camera poses)
            if epoch_it == self.start_query_world_epoch:
                self.query_in_canonical_space = True
                # Reset the learning rate
                for g in self.optimizer.param_groups:
                    g['lr'] = self.cfg['training']['learning_rate']
                self.scheduler._last_lr = [g['lr'] for g in self.optimizer.param_groups]
                for g in self.motion_optimizer.param_groups:
                    g['lr'] = 0.0
                self.motion_scheduler._last_lr = [g['lr'] for g in self.motion_optimizer.param_groups]
                # Pose refinement
                if self.cfg['training']['do_refine_pose']:
                    print("Rendering depth maps of train views")
                    self.render_train_views()
                    (pose_refine_dataset, pose_refine_dataloader, relative_pose_retriever,
                     relative_pose_optimizer, relative_pose_optimizer_scheduler, uv) = setup_pose_refinement(
                        self.cfg, self.motion_network, self.train_dataset, self.total_nb_images, self.nb_sample_timestep)
                    print("Perform pose refinement")
                    pred_poses = perform_pose_refinement(
                        self.model, self.motion_network, self.gt_poses,
                        pose_refine_dataset, pose_refine_dataloader,
                        relative_pose_retriever, relative_pose_optimizer, relative_pose_optimizer_scheduler,
                        uv, self.cfg, self.warp_pixel, self.compute_pose_error)
                    del pose_refine_dataset, pose_refine_dataloader, relative_pose_retriever, relative_pose_optimizer, relative_pose_optimizer_scheduler, uv
                    torch.cuda.empty_cache()
                else:
                    pred_poses = []
                    for image_idx in self.train_dataset['img'].i_train:
                        with torch.no_grad():
                            _, c2c = self.motion_network.compute_relative_camera_pose(
                                target_cam_idx=0,
                                final_ref_cam_idx=self.total_nb_images - 1,
                                total_nb_images=self.total_nb_images,
                                nb_sample_timestep=self.nb_sample_timestep)
                            w2c = self.motion_network.compute_w2c_mappings(c2c)[self.train_dataset['img'].i_train, :, :]
                            pred_poses = torch.inverse(w2c)
                pred_poses = torch.inverse(pred_poses) @ pred_poses[list(self.train_dataset['img'].i_train).index(self.world_cam_idx)].unsqueeze(0)
                self.pose_retriever = PoseRetriever(len(self.train_dataset['img'].i_train), learn_R=False, learn_t=False, init_c2w=pred_poses)
                self.pose_retriever = self.pose_retriever.to(device=self.device)
                torch.save(self.pose_retriever.state_dict(), f"./{self.out_dir}/models/refine_pose.pt")
                print(f"Start querying in the canonical space at epoch {epoch_it}")

            if epoch_it == self.end_smooth_epoch:
                self.model.smoothness_weight = self.cfg['training']['smoothness_weight'][1]
                self.model.edge_aware_smoothness_weight = self.cfg['training']['edge_aware_smoothness_weight'][1]
                self.patch_size = 1
                print(f'At epoch {epoch_it}, set smooth_weight={self.model.edge_aware_smoothness_weight} and patch_size={self.patch_size}')

            for batch in self.train_loader:
                self.it += 1
                # Loss weight annealing
                self.loss_weight_scalar_annealing(self.it)
                # Warmup learning rate
                if self.it <= self.nerf_lr_warm_up_it:
                    self.neus_warmup_learning_rate(self.it)

                image_idx = batch.get('img.idx')
                ref_image_idx_list = torch.cat(batch.get('img.ref_idxs'))
                ref_image_list = torch.cat(batch.get('img.ref_image_list'))
                ref_camera_mat_list = torch.cat(batch.get('img.ref_camera_mat')).float().to(device=self.device)
                time_step = (image_idx / (self.total_nb_images - 1) * 2 - 1).to(device=self.device)
                next_time_step = (ref_image_idx_list / (self.total_nb_images - 1) * 2 - 1).to(device=self.device)
                nb_valid_next_time_step = len(next_time_step[next_time_step.cpu() <= 1.0])

                freeze_camera_pose = (epoch_it >= self.start_query_world_epoch) and (epoch_it <= (self.start_query_world_epoch + self.freeze_camera_pose_period))
                with torch.set_grad_enabled(not freeze_camera_pose):
                    if self.query_in_canonical_space and (self.world_cam_idx != image_idx.item()):
                        _idx = list(self.train_dataset['img'].i_train).index(image_idx.item())
                        world_mat = self.pose_retriever(_idx)
                    else:
                        world_mat = torch.eye(4).float().to(self.device)
                    if freeze_camera_pose:
                        world_mat = world_mat.detach()

                (img, ref_img, sampled_pixel, normalized_sampled_pixel, 
                 rays_o, rays_d, rays_d_norm, rgb_gt, camera_mat, scale_mat) = self.model.process_data(
                    batch, world_mat, self.it, epoch_it, self.scheduling_start, self.render_path, patch_size=self.patch_size)

                near, far = self.model.near_far_from_sphere(rays_o, rays_d)
                cos_anneal_ratio = self.model.get_cos_anneal_ratio(self.it, self.anneal_end)

                query_time_step = torch.tensor([self.world_time_step]).float().to(self.device) if self.query_in_canonical_space else time_step
                render_out = self.model.renderer(
                    rays_o, rays_d, rays_d_norm,
                    query_time_step.repeat(len(self.cfg['training']['gpu_ids'])),
                    near, far, background_rgb=None, cos_anneal_ratio=cos_anneal_ratio, it=self.it, eval=False)

                rgb_pred = render_out['color_fine']
                depth_pred = render_out['depth_pred']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                sdf = render_out['sdf']
                normals = render_out['normals'].view(-1, 3)
                sdf_flows = render_out['sdf_flows'].view(-1)
                weights = render_out['weights'].view(-1)
                pts = render_out['sampled_points'].view(-1, 3)

                flow_loss = torch.tensor(0.0).float().to(self.device)
                flow_rgb_loss = torch.tensor(0.0).float().to(self.device)
                sdf_consistency_loss = torch.tensor(0.0).float().to(self.device)
                depth_smoothness_loss = torch.tensor(0.0).float().to(self.device)
                edge_aware_smoothness_loss = torch.tensor(0.0).float().to(self.device)
                smoothness_loss = torch.tensor(0.0).float().to(self.device)
                depth_loss = torch.tensor(0.0).float().to(self.device)
                sdf_loss = torch.tensor(0.0).float().to(self.device)

                if not self.query_in_canonical_space:
                    pts = render_out['sampled_points'].view(-1, 3)
                    normals = render_out['normals'].view(-1, 3)
                    sdf_flows = render_out['sdf_flows'].view(-1)
                    weights = render_out['weights'].view(-1)
                    angular_velocity, velocity = self.motion_network(torch.tensor([query_time_step]).float().to(self.device).view(-1, 1))
                    angular_velocity = angular_velocity.repeat(pts.shape[0], 1)
                    velocity = velocity.repeat(pts.shape[0], 1)
                    scene_flow = torch.cross(angular_velocity, pts) + velocity
                    lhs = torch.sum(scene_flow * normals, dim=-1)
                    sdf_loss = torch.sum(torch.abs(lhs + sdf_flows) * weights.detach()) / (torch.sum(weights.detach()) + 1e-10)

                    if ((self.cfg['training']['flow_rgb_weight'][1] != 0) or (sum(self.cfg['training']['sdf_consistency_weight']) != 0)) and (ref_image_idx_list[0] > image_idx):
                        time_interval, c2c = self.motion_network.compute_relative_camera_pose(
                            target_cam_idx=image_idx, final_ref_cam_idx=ref_image_idx_list[nb_valid_next_time_step - 1],
                            total_nb_images=self.total_nb_images, nb_sample_timestep=self.nb_sample_timestep)
                        w2c = self.motion_network.compute_w2c_mappings(c2c)[(ref_image_idx_list - image_idx)[:nb_valid_next_time_step], :, :]
                        flow_fw_pred_list = []
                        for t in range(len(w2c)):
                            ref_camera_mat = ref_camera_mat_list[[t]]
                            pts_map = (w2c[t, :3, :3] @ pts.T + w2c[t, :3, [-1]]).T
                            weighted_pts_map = torch.sum(weights.view(rgb_pred.shape[0], -1, 1) * pts_map.view(rgb_pred.shape[0], -1, 3), dim=1)
                            pixels_map = (scale_mat[0, :3, :3] @ ref_camera_mat[0, :3, :3] @ weighted_pts_map.T).T
                            pixels_map = pixels_map[:, :2] / pixels_map[:, [-1]]
                            flow_fw_pred = pixels_map - normalized_sampled_pixel
                            flow_fw_pred[:, 0] = flow_fw_pred[:, 0] * (img.shape[3] / 2)
                            flow_fw_pred[:, 1] = flow_fw_pred[:, 1] * (img.shape[2] / 2)
                            flow_fw_pred_list.append(flow_fw_pred)
                        if sum(self.cfg['training']['sdf_consistency_weight']) != 0 and image_idx.item() != self.world_cam_idx:
                            with torch.set_grad_enabled(self.cfg['training']['sdf_consistency_enable_pose_grad']):
                                cam_idx_pair = [self.world_cam_idx, image_idx.item()]
                                _, relative_pose = self.motion_network.compute_relative_camera_pose(
                                    target_cam_idx=min(cam_idx_pair), final_ref_cam_idx=max(cam_idx_pair),
                                    total_nb_images=self.total_nb_images, nb_sample_timestep=self.nb_sample_timestep)
                                c2c = self.motion_network.compute_w2c_mappings(relative_pose)[-1]
                                cw2 = torch.inverse(c2c) if self.world_cam_idx <= image_idx else c2c
                                pts_world = (cw2[:3, :3] @ pts.T + cw2[:3, [-1]]).T
                            sdf_w = self.sdf_network.sdf(torch.cat([pts_world, torch.ones_like(pts_map[:, [0]]) * (self.world_time_step)], dim=1))
                            sdf_consistency_loss = torch.mean(torch.abs(sdf_w - sdf))
                        if sum(self.cfg['training']['flow_rgb_weight']) != 0:
                            for t_idx in range(nb_valid_next_time_step):
                                flow_fw_pred = flow_fw_pred_list[t_idx]
                                ref_img = ref_image_list[t_idx].unsqueeze(0).float().to(self.device)
                                flow_correspondences = sampled_pixel + flow_fw_pred
                                with torch.no_grad():
                                    valid_pixel_mask = ((flow_correspondences >= 0) & (flow_correspondences < 
                                                           torch.tensor([ref_img.shape[3], ref_img.shape[2]]).float().to(self.device))).all(dim=1, keepdim=True)
                                rgb_warped = self.warp_pixel(src_frame=ref_img, uv=flow_correspondences.T.unsqueeze(0).unsqueeze(-1)).squeeze().T
                                flow_rgb_loss_t = torch.sum(torch.abs(rgb_warped - rgb_gt) * valid_pixel_mask) / (torch.sum(valid_pixel_mask) + 1e-10)
                                flow_rgb_loss += flow_rgb_loss_t
                            flow_rgb_loss = flow_rgb_loss / 3.0

                if sum(self.cfg['training']['edge_aware_smoothness_weight']) != 0 and self.patch_size > 1:
                    rgb_gt_grid = rgb_gt.view(-1, self.patch_size, self.patch_size, 3)
                    disp = depth_pred.view(-1, self.patch_size, self.patch_size, 1)
                    disp_norm = disp
                    edge_aware_smoothness_loss = 1 / (2 ** self.s) * self.compute_edge_smoothness_loss(disp_norm, rgb_gt_grid)
                if sum(self.cfg['training']['smoothness_weight']) != 0 and self.patch_size > 1:
                    smoothness_loss = 1 / (2 ** self.s) * self.compute_smoothness_loss(disp_norm)
                gradient_loss = torch.mean((torch.linalg.norm(normals.reshape(-1, 3), ord=2, dim=-1) - 1.0) ** 2)

                loss_dict = self.model.compute_loss(
                    batch, rgb_pred, rgb_gt, gradient_loss, sdf_loss, flow_rgb_loss, sdf_consistency_loss,
                    edge_aware_smoothness_loss, smoothness_loss,
                    it=self.it, epoch=epoch_it, scheduling_start=self.scheduling_start, out_render_path=self.render_path)
                self.model.backpropagation(loss_dict, train_motion_network= not freeze_camera_pose)

                if self.print_every > 0 and (self.it % self.print_every) == 0:
                    self.print_log(render_out, loss_dict)
                for m, milestones in enumerate(self.cfg['training']['depth_bound_scheduler_milestones']):
                    if self.it >= milestones:
                        visualize_every = self.cfg['training']['depth_bound_update_every_milestones'][m]
                if visualize_every > 0 and (self.it % visualize_every) == 0:
                    query_cam_idx = torch.tensor([self.world_cam_idx]).to(self.device) if self.query_in_canonical_space else None
                    try:
                        self.visualize_log(batch, world_mat, query_cam_idx, vis_idx=image_idx.item())
                    except Exception as e:
                        pass

                for key in loss_dict:
                    if 'loss' in key or 'l2' in key:
                        epoch_loss_dict[key].append(loss_dict[key].item())

            if (self.checkpoint_every > 0 and (epoch_it % self.checkpoint_every) == 0) and epoch_it > 0:
                self.save_checkpoint()
            self.logger.add_scalar('stats/psnr', mse2psnr(np.mean(epoch_loss_dict['l2_mean'])), epoch_it)
            if epoch_it % self.eval_pose_every == 0 and not self.query_in_canonical_space:
                try:
                    aligned_pred_pose, rpe_trans, rpe_rot, ate = self.pose_evaluation()
                    self.vis_pose_2d(aligned_pred_pose)
                except: pass
            self.print_log_epoch(epoch_loss_dict)
            self.scheduler.step()
            self.motion_scheduler.step()

if __name__=='__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('config_path', type=str, help='Config file path')
    args = parser.parse_args()
    # Load config
    cfg = dl.load_config(args.config_path, 'configs/default.yaml')
    # Save the backup of the config file
    out_dir = cfg['training']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    os.system(f"cp -p {args.config_path} {out_dir}")
    # Set seed reproducing the results
    seed = cfg['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Start training
    trainer = Trainer(cfg=cfg)
    trainer.train()

