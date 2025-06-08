import os
import torch
import logging
# from model.losses import Loss
import numpy as np
from PIL import Image
import imageio
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchvision.utils import flow_to_image
import cv2
from model.common import (get_tensor_values, arange_pixels,  project_to_cam)
from model.common import (image_points_to_world, origin_to_world, transform_to_world)
logger_py = logging.getLogger(__name__)
class Trainer(object):
    def __init__(self, renderer, optimizer, motion_optimizer, cfg, device=None, **kwargs):
        """model trainer

        Args:
            model (nn.Module): model
            optimizer (optimizer):pytorch optimizer object
            cfg (dict): config argument options
            device (device): Pytorch device option. Defaults to None.
        """
        self.total_nb_images = kwargs['total_nb_images']
        self.renderer = renderer
        self.optimizer = optimizer
        self.motion_optimizer = motion_optimizer
        self.cfg = cfg
        self.cfg_all = kwargs['cfg_all']
        self.depth_range = kwargs['cfg_all']["rendering"]["depth_range"]
        self.logger = kwargs['logger']
        self.gt_depths = kwargs['gt_depths']
        self.device = device
        self.n_training_points = cfg['n_training_points']

        self.rgb_weight = cfg['rgb_weight'][0]
        self.eikonal_weight = cfg['eikonal_weight'][0]
        self.sdf_weight = cfg['sdf_weight'][0]
        # self.flow_weight = cfg['flow_weight'][0]
        self.flow_rgb_weight = cfg['flow_rgb_weight'][0]
        self.sdf_consistency_weight = cfg['sdf_consistency_weight'][0]
        self.edge_aware_smoothness_weight = cfg['edge_aware_smoothness_weight'][0]
        self.smoothness_weight = cfg['smoothness_weight'][0]
        # self.depth_weight = cfg['depth_weight'][0]

        try:
            self.world_cam_idx = kwargs['world_cam_idx']
            self.train_dataset = kwargs['train_dataset']
        except: pass
        # self.weight_dist_2nd_loss = cfg['weight_dist_2nd_loss']
        # self.weight_dist_1st_loss = cfg['weight_dist_1st_loss']
        # self.depth_consistency_weight = cfg['depth_consistency_weight']
        # self.eikonal_weight_start = cfg['eikonal_weight_start']
        # self.eikonal_weight_stop_increase = cfg['eikonal_weight_stop_increase']
        # self.loss = Loss(cfg)

        # self.h, self.w = cfg['resolution']
        # if cfg['invalid_border_region'] != -1:
        #     # self.image_valid_region_mask[cfg['invalid_border_region']:-cfg['invalid_border_region']] = 1
        #     # self.image_valid_region_mask[:, cfg['invalid_border_region']:-cfg['invalid_border_region']] = 1
        #     self.image_valid_region_mask = torch.zeros(self.h, self.w) #.float().cuda()
        #     self.image_valid_region_mask[cfg['invalid_border_region']:-cfg['invalid_border_region'], cfg['invalid_border_region']:-cfg['invalid_border_region']] = 1
        # else:
        #     self.image_valid_region_mask = torch.ones(self.h, self.w) #.float().cuda()
        # self.image_valid_region_mask = self.image_valid_region_mask.float().cuda()

    def train_step(self, data, it=None, epoch=None,scheduling_start=None, render_path=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
            epoch(int): current number of epochs
            scheduling_start(int): num of epochs to start scheduling
        '''
        self.renderer.train()
        self.optimizer.zero_grad()
        if self.pose_param_net:
           self.pose_param_net.train()
           self.optimizer_pose.zero_grad()
        if self.focal_net:
            self.focal_net.train()
            self.optimizer_focal.zero_grad()
        if self.distortion_net:
            self.distortion_net.train()
            self.optimizer_distortion.zero_grad()
        loss_dict = self.compute_loss(data, it=it, epoch=epoch, scheduling_start=scheduling_start, out_render_path=render_path)
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        if self.optimizer_pose:
            self.optimizer_pose.step()
        if self.optimizer_focal:
            self.optimizer_focal.step()
        if self.optimizer_distortion:
            self.optimizer_distortion.step()
        return loss_dict

    
    def near_far_from_sphere(self, rays_o, rays_d):
        # The code assume that the object is bounded in a sphere.
        # Given the ray function is rays_d * t + ray_o, and a sphere function is x^T @ x = r^2. 
        # Find the intersection between they ray and the sphere, we have  (rays_d * t + ray_o)^T @ (rays_d * t + ray_o) = r^2
        # Rearrage the above equation into quadratic form: t^2 (rays_d@rays_d) + 2t (ray_o@ray_d) + (ray_o@ray_o - r^2) = 0.
        # The mid point is the middle point of the two intersection, which is equal to the average of the two solutions of the above equation. 
        # Thus mid point = - 2t (ray_o@ray_d) / (2 t^2 (rays_d@rays_d)),
            # which is equal to mid = 0.5 * (-b) / a in the code. 
        # The near and far point are points that are one unit away from the mid point. 
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        # TODO: Set dummy near far for now
        near = near * 0 + self.depth_range[0] # 0.01
        far = far * 0 + self.depth_range[1] # 10.0
        return near, far
    
    def get_cos_anneal_ratio(self, iter_step, anneal_end):
        if anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, iter_step / anneal_end])

    def compute_depth_errors(self, gt_depth_map, pred_depth_map, min_depth=0.1, max_depth=80.0):
        """Computation of error metrics between predicted and ground truth depths
        """

        pred_depth_map = cv2.resize(pred_depth_map, (gt_depth_map.shape[1], gt_depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_depth_map = (gt_depth_map >= min_depth) & (gt_depth_map <= max_depth)

        pred = pred_depth_map[valid_depth_map]
        gt = gt_depth_map[valid_depth_map]

        ratio = np.median(gt) / np.median(pred)
        pred = pred * ratio

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
    

    def render_visdata(self, data, world_mat, query_cam_idx, resolution, it, anneal_end, out_render_path=None, idx=None, render_only=False):
        (img, camera_mat, scale_mat, _) = self.process_data_dict(data)

        # image_idx = data.get('img.idx')
        if idx is None:
            # img_idx = image_idx = torch.tensor(np.random.randint(self.total_nb_images)).view(1)
            image_idx = data.get('img.idx')
        else:
            img_idx = image_idx = torch.tensor(idx).view(1)
        ref_image_idx = image_idx + (data.get('img.ref_idxs')[-1] - data.get('img.idx'))

        time_step = (image_idx / (self.total_nb_images-1) * 2 - 1).to(device=img.device)
        next_time_step = (ref_image_idx / (self.total_nb_images-1) * 2 - 1).to(device=img.device)
        if query_cam_idx is not None and (query_cam_idx!=image_idx.item()): 
            query_time_step = (query_cam_idx / (self.total_nb_images-1) * 2 - 1).to(device=img.device)
            if world_mat is None:
                with torch.no_grad():
                    cam_idx_pair = [query_cam_idx.item(), image_idx.item()]
                    _, relative_pose = self.renderer.module.motion_network.compute_relative_camera_pose(target_cam_idx=min(cam_idx_pair), final_ref_cam_idx=max(cam_idx_pair), 
                                                                                total_nb_images=self.total_nb_images, nb_sample_timestep=self.cfg['nb_sample_timestep'])
                    c2c = self.renderer.module.motion_network.compute_w2c_mappings(relative_pose)[-1]
                    world_mat = c2c if query_cam_idx <= image_idx else torch.inverse(c2c)
        else: 
            query_time_step = time_step
            world_mat = torch.eye(4).float().cuda()

        h, w = resolution
            
        p_idx = torch.arange(h*w).to(self.device)
        p_loc, pixels = arange_pixels(resolution=(h, w))

        pixels = pixels.to(self.device)
        # depth_input = dpt
        # depth_img_resized = F.interpolate(depth_input, size=(h, w) ,mode='nearest')
        # depth_img_resized = depth_img_resized.view(1, 1, -1).permute(0, 2, 1) 

        with torch.no_grad():
            rgb_pred = []
            depth_pred = []
            weighted_z_vals_pred = []
            flow_pred = []
            normal_pred = []
            depth_highest_weight_pred = []
            angular_velocity_list = []
            velocity_list = []

            # Predict motion
            nb_sample_timestep = self.cfg['nb_sample_timestep'] * (ref_image_idx[0]-image_idx[0]) 
            for time_step_t in torch.linspace(time_step.item(), next_time_step.item(), (nb_sample_timestep+1))[:-1]:
                angular_velocity_t, velocity_t = self.renderer.module.motion_network(time_step_t.view(-1,1).cuda())
                angular_velocity_list.append(angular_velocity_t)
                velocity_list.append(velocity_t)

            for i in range(0, pixels.shape[1]//1024+1):
                pixels_i = pixels[:,i*1024:(i+1)*1024,:]
                # color_valid_mask_i = self.color_valid_mask[i*1024:(i+1)*1024,:].unsqueeze(0).unsqueeze(-1)
                # print('pixels_i: ', pixels_i.shape, torch.sum(pixels_i))
                # depth_i = depth_img_resized[:,i*1024:(i+1)*1024,:]
                ray_o_i, ray_d_i, rays_d_norm_i = self.get_world_cameraOrigin_cameraRay(pixels_i, camera_mat, world_mat, scale_mat)

                # ray_o_i, ray_d_i, ray_d_i_norm, depth_gt_i, valid_depth_mask_i = self.get_world_cameraOrigin_cameraRay(pixels_i, depth_i, camera_mat, world_mat, scale_mat)

                near, far = self.near_far_from_sphere(ray_o_i, ray_d_i)

                # Model predictions
                render_out = self.renderer(ray_o_i, ray_d_i, rays_d_norm_i, query_time_step.repeat(len(self.cfg_all['training']['gpu_ids'])), near, far, 
                                                background_rgb=None, cos_anneal_ratio=self.get_cos_anneal_ratio(it, anneal_end), it=it, eval=True)
                
                # render_out = self.renderer.render(ray_o_i, ray_d_i, ray_d_i_norm, near, far,
                #                                 background_rgb=None,
                #                                 cos_anneal_ratio=self.get_cos_anneal_ratio(it, anneal_end), it=it, eval=True)
                rgb_pred_i = render_out['color_fine']
                depth_pred_i = render_out['depth_pred']
                weighted_z_vals_pred_i = render_out['weighted_z_vals']
                pts = render_out['sampled_points'].view(-1,3)
                sdf = render_out['sdf'].view(-1,1)
                weights = render_out['weights'] # [1024, 128]

                # Get depth corresponding to the points having weight for each ray
                _, max_idx = torch.max(weights, dim=1)
                max_idx = torch.stack([torch.from_numpy(np.arange(len(weights))), max_idx.detach().cpu()])
                if torch.all(world_mat == torch.eye(4).float().cuda()):
                    depth_highest_weight = -render_out['sampled_points'][:,:,-1][tuple(max_idx)] # [1024, 128, 3]
                else:
                    pc_transform = world_mat @ (torch.cat([pts, torch.ones_like(pts[:,[0]])],dim=-1)).T
                    pc_transform = pc_transform.T[:,:3].view(weights.shape[0], weights.shape[1],3)
                    depth_highest_weight = -pc_transform[:,:,-1][tuple(max_idx)] # [1024, 128, 3]

                # depth_highest_weight = torch.clip(depth_highest_weight, 0,5)
                depth_highest_weight_pred.append(depth_highest_weight.detach().cpu())

                rgb_pred.append(rgb_pred_i.detach().cpu())
                # depth_pred_i = out_dict['depth_pred']
                depth_pred.append(depth_pred_i.detach().cpu())
                weighted_z_vals_pred.append(weighted_z_vals_pred_i.detach().cpu())


                n_samples = self.renderer.module.n_samples + self.renderer.module.n_importance
                normal_i = render_out['normals'] * render_out['weights'][:, :n_samples, None]
                # normal_i = normal_i * render_out['inside_sphere'][..., None]
                normal_i = normal_i.sum(dim=1)
                if not torch.all(world_mat == torch.eye(4).float().cuda()):
                    normal_i = world_mat[:3,:3] @ normal_i.T
                    normal_i = normal_i.T
                normal_i = normal_i.detach().cpu()
                normal_pred.append(normal_i) # torch.Size([129600, 3])

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # !!!!!!!!!!!! Compute fw optical flow !!!!!!!!!!!!!!!!
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # For each point compute its scene flow

                pts_sf = torch.clone(pts)
                time_interval = (next_time_step-time_step)/(nb_sample_timestep)
                for t, time_step_t in enumerate(range(nb_sample_timestep)):
                    scene_flow_t = torch.cross(angular_velocity_list[t], pts_sf) + velocity_list[t]
                    pts_sf = pts_sf + time_interval*scene_flow_t
                weights = weights.view(rgb_pred_i.shape[0],-1,1)
                pts_sf = pts_sf.view(rgb_pred_i.shape[0],-1,3)
                pts_sf = torch.sum(weights*pts_sf, dim=1)
                # Compute pixel coordinate from pts_sf
                pixels_sf = (scale_mat[0,:3,:3] @ camera_mat[0,:3,:3] @ pts_sf.T).T
                pixels_sf = pixels_sf[:,:2] / pixels_sf[:,[-1]]
                flow_fw_pred_i = pixels_sf - pixels_i
                flow_pred.append(flow_fw_pred_i[0])

                del render_out
                
            rgb_pred = torch.cat(rgb_pred, dim=0)
            depth_pred = torch.cat(depth_pred, dim=0)
            weighted_z_vals_pred = torch.cat(weighted_z_vals_pred, dim=0)
            depth_highest_weight_pred = torch.cat(depth_highest_weight_pred, dim=0)
            flow_pred = torch.cat(flow_pred, dim=0)
            normal_pred = torch.cat(normal_pred, dim=0) # torch.Size([129600, 3])
     
            rgb_pred = rgb_pred.view(h, w, 3).detach().cpu().numpy()
            img_out = (rgb_pred * 255).astype(np.uint8)
            depth_pred_out = depth_pred.view(h, w).detach().cpu().numpy()
            weighted_z_vals_pred = weighted_z_vals_pred.view(h, w).detach().cpu().numpy()
            depth_highest_weight_pred = depth_highest_weight_pred.view(h, w).detach().cpu().numpy()

            # Process predicted flow
            flow_pred[:,0] = flow_pred[:,0] * (rgb_pred.shape[1]/2)
            flow_pred[:,1] = flow_pred[:,1] * (rgb_pred.shape[0]/2)
            flow_pred = flow_pred.view(h, w, 2)
            try: flow_pred = flow_to_image(flow_pred.permute(2,0,1)).permute(1,2,0).detach().cpu().numpy()
            except: flow_pred = torch.zeros_like(img_out)




            # depth_pred_out = depth_pred_out/depth_pred_out.max()
            disp_pred_out = 1/depth_pred_out
            disp_pred_out = disp_pred_out / disp_pred_out.max()
            disp_highest_weight_pred = 1/depth_highest_weight_pred
            disp_highest_weight_pred = disp_highest_weight_pred / disp_highest_weight_pred.max()

            normal_img = normal_pred
            rot = torch.eye(3).detach().cpu() #torch.inverse(self.pose_param_net(img_idx)[:3,:3].detach().cpu())
            normal_img = (torch.matmul(rot[None, :, :], normal_img[:, :, None])
                            .reshape([h, w, 3, -1]) * 128 + 128).clip(0, 255)
            normal_img = normal_img.numpy()[:,:,:,0]

            render_pkl = {
                'render_image': rgb_pred,
                'render_depth': depth_pred_out,
                'render_normal': normal_pred
            }

            if render_only:
                return render_pkl

            imageio.imwrite(os.path.join(out_render_path,'%04d_disparity.png'% img_idx), (disp_pred_out*255.0).astype(np.uint8))
            imageio.imwrite(os.path.join(out_render_path,'%04d_disparity_highest_weight.png'% img_idx), (disp_highest_weight_pred*255.0).astype(np.uint8))
            imageio.imwrite(os.path.join(out_render_path,'%04d_normal.png'% img_idx), normal_img.astype(np.uint8))
            
            img1 = Image.fromarray((img_out).astype(np.uint8)).convert("RGB")
            flow_pred = Image.fromarray((flow_pred).astype(np.uint8)).convert("RGB")

            img1.save(os.path.join(out_render_path, '%04d_img.png' % img_idx))
            flow_pred.save(os.path.join(out_render_path, '%04d_flow.png' % img_idx))

            # Update adaptive depth range
            for m, milestones in enumerate(self.cfg['depth_bound_scheduler_milestones']):
                if it >= milestones: depth_bound_lr = self.cfg['depth_bound_lr'][m]

            min_depth, max_depth = np.min(weighted_z_vals_pred), np.max(weighted_z_vals_pred)
            min_depth = max(self.cfg_all["rendering"]["depth_range"][0], min_depth * 0.9)
            max_depth = max_depth * 1.1

            # We do not update the depth lower bound
            # self.depth_range[0] = self.depth_range[0] * (1-depth_bound_lr) + min_depth * depth_bound_lr
            self.depth_range[1] = self.depth_range[1] * (1-depth_bound_lr) + max_depth * depth_bound_lr

            self.logger.add_scalar('stats/depth_running_min', self.depth_range[0], it)
            self.logger.add_scalar('stats/depth_running_max', self.depth_range[1], it)
            
            self.logger.add_scalar('stats/depth_sample_min', min_depth, it)
            self.logger.add_scalar('stats/depth_sample_max', max_depth, it)

            # Depth evaluation 
            if len(self.gt_depths) != 0:
                depth_metric_name = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
                gt_depth = self.gt_depths[image_idx]
                # gt_depth = np.load(f'{{self.cfg_all["dataloading"]["path"]}/{self.cfg_all["dataloading"]["scene"][0]}}/gt_depth/depth_{str(int(image_idx)).zfill(6)}.npz')['pred']
                if self.cfg_all['dataloading']['crop_size'] != 0:
                    depth_h, depth_w = gt_depth.shape
                    crop_h = self.cfg_all['dataloading']['crop_size']
                    crop_w = int(crop_h * depth_w/depth_h)
                    gt_depth = cv2.resize(gt_depth, (648,484), interpolation=cv2.INTER_NEAREST)
                    gt_depth = gt_depth[crop_h:depth_h-crop_h, crop_w:depth_w-crop_w]
                depth_error = self.compute_depth_errors(gt_depth, depth_pred_out)
                depth_highest_weight_error = self.compute_depth_errors(gt_depth, depth_highest_weight_pred)
                for d in range(len(depth_metric_name)): 
                    self.logger.add_scalar(f'depth_eval/{depth_metric_name[d]}', depth_error[d], it)
                    self.logger.add_scalar(f'depth_highW_eval/{depth_metric_name[d]}', depth_highest_weight_error[d], it)
                    
        return render_pkl


    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors
        Args:
            data (dictionary): data dictionary
        '''
        device = self.device
        img = data.get('img').to(device)
        depth = None
        img_idx = data.get('img.idx')
        # dpt = data.get('img.dpt').to(device).unsqueeze(1)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        # flow_fw_list = [flow.to(device) for flow in data.get('img.flow_fw')]
        # flow_bw_list = [flow.to(device) for flow in data.get('img.flow_bw')]
       
        return (img, camera_mat, scale_mat, img_idx)
    def process_data_reference(self, data):
        ''' Processes the data dictionary and returns respective tensors
        Args:
            data (dictionary): data dictionary
        '''
        device = self.device
        ref_imgs = data.get('img.ref_imgs').to(device)
        ref_dpts = None # data.get('img.ref_dpts').to(device).unsqueeze(1)
        ref_idxs = data.get('img.ref_idxs')
        return ( ref_imgs, ref_dpts, ref_idxs)
    def anneal(self, start_weight, end_weight, anneal_start_epoch, anneal_epoches, current):
        """Anneal the weight from start_weight to end_weight
        """
        if current <= anneal_start_epoch:
            return start_weight
        elif current >= anneal_start_epoch + anneal_epoches:
            return end_weight
        else:
            return start_weight + (end_weight - start_weight) * (current - anneal_start_epoch) / anneal_epoches
        
    def get_patch_indices(self, h, w, patch_size, n_points):
        n_patches = n_points // (patch_size ** 2)
        # Adjusted dimensions to ensure valid patch selection
        h_adjusted, w_adjusted = h - patch_size + 1, w - patch_size + 1

        # Number of patches across the adjusted width and height
        n_patches = min(n_patches, h_adjusted * w_adjusted)
        
        # Sample top-left corners
        top_left_corners = torch.randperm(h_adjusted * w_adjusted)[:n_patches]
        
        # Convert flat indices to 2D indices
        rows = top_left_corners // w_adjusted
        cols = top_left_corners % w_adjusted

        # Generate patch offsets
        patch_offsets = torch.arange(patch_size).repeat(patch_size, 1)
        patch_offsets =  (patch_offsets + patch_offsets.t() * w).flatten()
        # Calculate start indices of patches in the flattened image
        start_indices = (rows * w + cols).unsqueeze(1)  # Each patch's top-left corner in the flattened image
        
        # Calculate all indices for the patches in the flattened image
        patch_indices = start_indices + patch_offsets.view(-1)
        return patch_indices.flatten()


    def process_data(self, data, world_mat, eval_mode=False, it=None, epoch=None, scheduling_start=None, out_render_path=None, patch_size=1):
      
        n_points = self.n_training_points
        img, camera_mat_gt, scale_mat, _ = self.process_data_dict(data)   
        (ref_img, _, ref_idx) = self.process_data_reference(data)   
     
        device = self.device
        batch_size, _, h, w = img.shape
        kwargs = dict()
        
        camera_mat = camera_mat_gt
        # # Sample pixels
        # if patch_size == 1: 
        #     ray_idx = torch.randperm(h*w,device=device)[:n_points]
        # else:
        ray_idx = self.get_patch_indices(h, w, patch_size, n_points)

        img_flat = img.view(batch_size, 3, h*w).permute(0,2,1)
        # depth_flat = depth.view(batch_size, 1, h*w).permute(0,2,1)
        rgb_gt = img_flat[:,ray_idx]
        # depth_gt = depth_flat[:,ray_idx]

        # flow_fw_gt_list = [flow_fw_gt_list[f].view(batch_size,2,h*w).permute(0,2,1)[:, ray_idx][0] for f in range(len(flow_fw_gt_list))]
        # flow_bw_gt_list = [flow_bw_gt_list[f].view(batch_size,2,h*w).permute(0,2,1)[:, ray_idx][0] for f in range(len(flow_bw_gt_list))]

        p_full, p_full_normalize = arange_pixels((h, w), batch_size, device=device)
        p_normalize = p_full_normalize[:, ray_idx]
        p = p_full[:, ray_idx].float()

        # My change here !!!!! 
        # ray_o, ray_d, rays_d_norm = self.get_cameraOrigin_cameraRay(p_normalize, camera_mat, scale_mat)
        ray_o, ray_d, rays_d_norm = self.get_world_cameraOrigin_cameraRay(p_normalize, camera_mat, world_mat, scale_mat)
        return (img, ref_img, p[0], p_normalize[0], ray_o, ray_d, rays_d_norm, rgb_gt[0], camera_mat, scale_mat)
        
    
    def get_world_cameraOrigin_cameraRay(self, pixels, camera_mat, world_mat, scale_mat):
        # Get configs
        batch_size, n_points, _ = pixels.shape
        # Find surface points in world coorinate
        camera_world = origin_to_world(n_points, camera_mat, world_mat, scale_mat)
        # Prepare camera projection
        pixels_world = image_points_to_world(pixels, camera_mat, world_mat,scale_mat)
        ray_vector = (pixels_world - camera_world)
        ray_vector_norm = ray_vector.norm(2,2)
        ray_vector = ray_vector/ray_vector_norm.unsqueeze(-1) # normalised ray vector
        # Project depth to 3d poinsts
        ray_o = camera_world.reshape(-1, 3)
        ray_d = ray_vector.reshape(-1, 3)
        return ray_o, ray_d, ray_vector_norm.view(-1,1)
        

    def compute_loss(self, data, rendered_rgb, rgb_gt, 
                     gradient_loss, sdf_loss, flow_rgb_loss, sdf_consistency_loss, 
                     edge_aware_smoothness_loss, smoothness_loss,
                    #  depth_input, scale_input, shift_input,
                     it=None, epoch=None, scheduling_start=None, out_render_path=None):
        weights = {}
        weights_name_list = ['rgb_weight', 
                            #  'depth_weight', 
                             'eikonal_weight', 'sdf_weight', 
                            #  'flow_weight', 
                             'flow_rgb_weight', 'sdf_consistency_weight', 'edge_aware_smoothness_weight', 'smoothness_weight']
        weights_list = [getattr(self, w) for w in weights_name_list] # loss weights
        # use_ref_imgs = (weights['flow_weight']!=0.0)
        # (ref_img, depth_ref, ref_idx) = self.process_data_reference(data)
        weights = dict(zip(weights_name_list, weights_list))

        if weights['rgb_weight'] == 0.0: rgb_full_loss = torch.tensor(0.0).cuda().float()
        else: 
            rgb_full_loss = torch.sum(torch.abs(rendered_rgb - rgb_gt)) / float(rendered_rgb.shape[0])
            rgb_l2_mean = F.mse_loss(rendered_rgb, rgb_gt)
        # if weights['depth_weight'] == 0.0:
        #     depth_loss = torch.tensor(0.0).cuda().float()
        if weights['eikonal_weight']==0.0:
            gradient_loss = torch.tensor(0.0).cuda().float()
        if weights['sdf_weight']==0.0:
            sdf_loss = torch.tensor(0.0).cuda().float()
        # if weights['flow_weight']==0.0:
        #     flow_loss = torch.tensor(0.0).cuda().float()
        if weights['flow_rgb_weight']==0.0:
            flow_rgb_loss = torch.tensor(0.0).cuda().float()
        if weights['edge_aware_smoothness_weight']==0.0:
            edge_aware_smoothness_loss = torch.tensor(0.0).cuda().float()
        if weights['smoothness_weight']==0.0:
            smoothness_loss = torch.tensor(0.0).cuda().float()

        loss = weights['rgb_weight'] * rgb_full_loss + \
                    weights['eikonal_weight'] * gradient_loss +\
                        weights['sdf_weight'] * sdf_loss+\
                            weights['flow_rgb_weight'] * flow_rgb_loss+\
                                weights['sdf_consistency_weight'] * sdf_consistency_loss+\
                                    weights['edge_aware_smoothness_weight'] * edge_aware_smoothness_loss+\
                                        weights['smoothness_weight'] * smoothness_loss
        if torch.isnan(loss):
            assert False, "Nan loss found"

        loss_dict = {
            'loss': loss,
            'loss_rgb': rgb_full_loss,
            'loss_eikonal': gradient_loss,
            'l2_mean': rgb_l2_mean,
            'loss_sdf': sdf_loss,
            'loss_flow_rgb': flow_rgb_loss,
            'sdf_consistency_loss': sdf_consistency_loss,
            'edge_aware_smoothness_loss': edge_aware_smoothness_loss,
            'smoothness_loss': smoothness_loss,
        }
        # loss_dict = self.loss(rendered_rgb, rgb_gt, depth_loss,gradient_error, sdf_loss, flow_loss, 
        #                       flow_rgb_loss, sdf_consistency_loss, edge_aware_smoothness_loss, smoothness_loss,
        #                       weights)
        return loss_dict
    

    def backpropagation(self, loss_dict, train_motion_network):
        self.optimizer.zero_grad()
        if train_motion_network: self.motion_optimizer.zero_grad()
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        if train_motion_network: self.motion_optimizer.step()



    