
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from model import PoseRetriever
import os 
from tqdm import tqdm

# Dataset class
class PoseRefineDataset(Dataset):
    def __init__(self, image_list, depth_list, K_list, max_interval=1):
        self.image_list = image_list
        self.depth_list = depth_list
        self.K_list = K_list
        self.max_interval = max_interval

    def __len__(self):
        return len(self.image_list)-1

    def __getitem__(self, idx):
        next_idx = np.random.randint(idx+1, idx+self.max_interval+1)
        if next_idx >= len(self.image_list): next_idx = len(self.image_list) -1
        # Get image
        image = torch.from_numpy(self.image_list[idx]).float()
        next_image = torch.from_numpy(self.image_list[next_idx])
        # Get depth
        depth = torch.from_numpy(self.depth_list[idx]).float()
        next_depth = torch.from_numpy(self.depth_list[next_idx]).float()
        # Get intrinsic
        K = self.K_list[idx]
        next_K = self.K_list[next_idx]
        return idx, next_idx, image, next_image, depth, next_depth, K, next_K
    
def compute_loss_and_warp_image(images, next_images, depths, K_batch, uv_batch, relative_poses, warp_pixel_fn):
    # Step III: Perform mapping
    
    # 1. Project pixels to 3D space
    xyz = torch.inverse(K_batch) @ ((uv_batch * depths).view(len(images), 3, -1))
    
    # 2. Transform xyz to the coordinate system of the next time step
    transformed_xyz = relative_poses[:, :3, :3] @ xyz + relative_poses[:, :3, [-1]]
    
    # 3. Project the transformed_xyz back to the image space
    tranformed_uv = K_batch @ transformed_xyz
    transformed_depth = tranformed_uv[:, [-1]]
    tranformed_uv = tranformed_uv[:, :2] / transformed_depth
    tranformed_uv = tranformed_uv.view(uv_batch[:, :2].shape)

    # Check if both U and V coordinates are within the range [-1, 1]
    valid_u = (tranformed_uv[:, 0, :, :] >= -1) & (tranformed_uv[:, 0, :, :] <= 1)
    valid_v = (tranformed_uv[:, 1, :, :] >= -1) & (tranformed_uv[:, 1, :, :] <= 1)
    valid_mask = valid_u & valid_v
    valid_mask = valid_mask.float().unsqueeze(1)

    # Step IV: Get the warped image
    warped_images = warp_pixel_fn(src_frame=next_images, uv=tranformed_uv, normalize_pix=False)
    
    # Step V: Compute photometric loss
    loss = torch.sum(torch.abs(warped_images - images)*valid_mask) / torch.sum(valid_mask)
    
    return loss, warped_images

def setup_pose_refinement(cfg, motion_network, train_dataset, total_nb_images, nb_sample_timestep):
    out_dir = cfg['training']['out_dir']
    resolution = np.array(cfg['training']['resolution']) 
    # Initialize the dataset
    depth_dir = f'{out_dir}/extraction_stage1/depths'
    depth_file_list = sorted([file for file in os.listdir(depth_dir) if file.endswith('.npz')] )
    depth_list = np.stack([np.load(f'{depth_dir}/{depth_file}')['pred'] for depth_file in depth_file_list])
    pose_refine_dataset = PoseRefineDataset(image_list=train_dataset['img'].imgs, 
                                        depth_list=depth_list, 
                                        K_list=train_dataset['img'].K[train_dataset['img'].i_train][:,:3,:3])
    pose_refine_dataloader = DataLoader(pose_refine_dataset, batch_size=16, shuffle=False)
    # Initialize the relative camera poses
    if not cfg['training']['refine_from_scratch']:
        with torch.no_grad(): 
            c2c_list = []
            for idx_pair in zip(train_dataset['img'].i_train[:-1], train_dataset['img'].i_train[1:]):
                _, c2c = motion_network.compute_relative_camera_pose(target_cam_idx=idx_pair[0], final_ref_cam_idx=idx_pair[1], 
                                                                    total_nb_images=total_nb_images, nb_sample_timestep=nb_sample_timestep)
                c2c = motion_network.compute_w2c_mappings(c2c)[-1]
                c2c_list.append(c2c)
            c2c_list = torch.stack(c2c_list)
    else: c2c_list = None
    # Set up relative_pose_retriever
    relative_pose_retriever = PoseRetriever(len(train_dataset['img'].i_train)-1, learn_R=True, learn_t=True, init_c2w=c2c_list)
    relative_pose_retriever = relative_pose_retriever.cuda()
    relative_pose_optimizer = torch.optim.Adam(relative_pose_retriever.parameters(), lr=cfg['training']['pose_refine_lr'])
    relative_pose_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(relative_pose_optimizer, 
                                                    milestones=list(range(30, 10000, 10)),
                                                    gamma=0.9, last_epoch=-1)
    # Set a default pixel coordinate
    uv = torch.squeeze(torch.stack(torch.meshgrid(torch.arange(0, end=resolution[0], dtype=torch.float),
                                    torch.arange(0, end=resolution[1], dtype=torch.float),
                                    torch.tensor([1.0, ]))), dim=3)
    uv[[0,1],:,:] = uv[[1,0],:,:]
    uv = uv.float().cuda()
    # Normalize the uv cooridnate from -1 to 1
    uv[0] = uv[0] / ((resolution[1] - 1) / 2) - 1 
    uv[1] = uv[1] / ((resolution[0] - 1) / 2) - 1
    return pose_refine_dataset, pose_refine_dataloader, relative_pose_retriever, relative_pose_optimizer, relative_pose_optimizer_scheduler, uv


def perform_pose_refinement(model, motion_network, gt_poses, pose_refine_dataset, pose_refine_dataloader, relative_pose_retriever, relative_pose_optimizer, relative_pose_optimizer_scheduler, uv, cfg, warp_pixel_fn, compute_pose_error_fn):
    resolution = np.array(cfg['training']['resolution']) 
    running_loss_window = []
    for epoch in tqdm(range(cfg['training']['pose_refine_epochs'])):
        running_loss = 0
        for batch in pose_refine_dataloader:
            # I. Get idx, image and depths
            image_idx, next_image_idx, images, next_images, depths, next_depths, K, next_K = [item.float().cuda() for item in batch]
            depths, next_depths = depths.unsqueeze(1), next_depths.unsqueeze(1)
            depths = torch.nn.functional.interpolate(depths, (int(resolution[0]),int(resolution[1])))
            next_depths = torch.nn.functional.interpolate(next_depths, (int(resolution[0]),int(resolution[1])))
            # depths_resize = depths_resize * -1
            uv_batch = uv.unsqueeze(0).repeat(len(images),1,1,1)
            # II. Get the relative camera motion
            relative_poses = torch.stack([relative_pose_retriever(idx) for i,idx in enumerate(image_idx)])
            # III. Perform mapping
            # 1. Project pixels to 3D space
            pose_refine_loss_pos, warped_images_pos = compute_loss_and_warp_image(images, next_images, depths, K, uv_batch, relative_poses, warp_pixel_fn)
            pose_refine_loss_neg, warped_images_neg = compute_loss_and_warp_image(next_images, images, next_depths, K, uv_batch, torch.inverse(relative_poses), warp_pixel_fn)
            pose_refine_loss = (pose_refine_loss_pos + pose_refine_loss_neg)/2
            running_loss += pose_refine_loss.item()*len(images)
            # VI. Backpropagation
            relative_pose_optimizer.zero_grad()
            pose_refine_loss.backward()
            relative_pose_optimizer.step()
        relative_pose_optimizer_scheduler.step()
        # VII. Perform evaluation
        with torch.no_grad():   
            # Get all the relative camera poses
            all_relative_poses = torch.stack([relative_pose_retriever(idx) for idx in range(len(pose_refine_dataset))])
            # Compute the camera poses from the relative camera motions by choosing the first as the world coordinate system 
            world_to_cam = motion_network.compute_w2c_mappings(all_relative_poses)
            pred_poses = torch.inverse(world_to_cam)
            # Compute pose metrics
            _, rpe_trans, rpe_rot, ate = compute_pose_error_fn(pred_poses, gt_poses)
            model.logger.add_scalar(f'poseRefine/_loss', running_loss/len(pose_refine_dataset), epoch)
            model.logger.add_scalar(f'poseRefine/rpe_trans', rpe_trans, epoch)
            model.logger.add_scalar(f'poseRefine/rpe_rot', rpe_rot, epoch)
            model.logger.add_scalar(f'poseRefine/ate', ate, epoch)
            model.logger.add_scalar(f'poseRefine/lr', relative_pose_optimizer_scheduler.state_dict()['_last_lr'][0], epoch)
            if len(running_loss_window) >= 50: running_loss_window.pop(0)
            running_loss_window.append(running_loss/len(pose_refine_dataset)) 
            # Stop if converge criteria is met
            if len(running_loss_window)==50 and np.std(running_loss_window) <= 1e-5: 
                print("Converged. Stop refining poses.")
                return pred_poses
    return pred_poses