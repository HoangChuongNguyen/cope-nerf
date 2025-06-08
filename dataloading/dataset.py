import os
import glob
import random
import logging
import torch
from PIL import Image
import numpy as np
import imageio
import cv2
from dataloading.common import _load_data, recenter_poses, spherify_poses, load_depths_npz, load_gt_depths, load_flows
from utils_poses.pose_pytorch3d import *
logger = logging.getLogger(__name__)

class DataField(object):
    def __init__(self, model_path,
                #  transform=None, 
                 with_camera=False, 
                #  use_DPT=False, 
                 scene_name=[' '], mode='train', spherify=False, 
                 load_ref_img=False,
                #  customized_poses=False,
                #  customized_focal=False,
                 resize_factor=2, 
                #  depth_net='dpt',
                 crop_size=0, 
                #  random_ref=False,
                   random_ref_interval=False, 
                #  norm_depth=False,
                 load_gt_depth=True,
                 load_colmap_poses=True, sample_rate=8, 
                 load_flow=False, resolution=None, **kwargs):
        """load images, depth maps, etc.
        Args:
            model_path (str): path of dataset
            transform (class, optional):  transform made to the image. Defaults to None.
            with_camera (bool, optional): load camera intrinsics. Defaults to False.
            load_gt_depth (bool, optional): load gt depth maps (if available). Defaults to False.
            DPT (bool, optional): run DPT model. Defaults to False.
            scene_name (list, optional): scene folder name. Defaults to [' '].
            mode (str, optional): train/eval/all/render. Defaults to 'train'.
            spherify (bool, optional): spherify colmap poses (no effect to training). Defaults to False.
            load_ref_img (bool, optional): load reference image. Defaults to False.
            customized_poses (bool, optional): use GT pose if available. Defaults to False.
            customized_focal (bool, optional): use GT focal if provided. Defaults to False.
            resize_factor (int, optional): image downsample factor. Defaults to 2.
            depth_net (str, optional): which depth estimator use. Defaults to 'dpt'.
            crop_size (int, optional): crop if images have black border. Defaults to 0.
            random_ref (bool/int, optional): if use a random reference image/number of neaest images. Defaults to False.
            norm_depth (bool, optional): normalise depth maps. Defaults to False.
            load_colmap_poses (bool, optional): load colmap poses. Defaults to True.
            sample_rate (int, optional): 1 in 'sample_rate' images as test set. Defaults to 8.
        """
        # self.transform = transform
        self.with_camera = with_camera
        self.load_gt_depth = load_gt_depth
        self.load_flow = load_flow
        # self.use_DPT = use_DPT
        self.mode = mode
        self.ref_img = load_ref_img
        # self.random_ref = random_ref
        self.random_ref_interval = random_ref_interval
        self.sample_rate = sample_rate
        self.h, self.w = resolution[0], resolution[1]
        
        is_tank = load_colmap_poses = ('tanks' in model_path.lower())
        is_scannet = ('scannet' in model_path.lower())
        is_co3d = ('co3d' in model_path.lower())
        load_dir = os.path.join(model_path, scene_name[0])
        # if crop_size!=0:
        #     depth_net = depth_net + '_' + str(crop_size)
        poses, bds, imgs, img_names, crop_ratio, focal_crop_factor = _load_data(load_dir, factor=resize_factor, crop_size=crop_size, load_colmap_poses=load_colmap_poses)
        if is_tank:
            poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
            poses = np.moveaxis(poses, -1, 0).astype(np.float32)
            bds = np.moveaxis(bds, -1, 0).astype(np.float32)
            bd_factor = 0.75
            # Rescale if bd_factor is provided
            sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
            poses[:,:3,3] *= sc
            bds *= sc
            poses = recenter_poses(poses)
            if spherify:
                poses, render_poses, bds = spherify_poses(poses, bds)
            input_poses = poses.astype(np.float32)
            hwf = input_poses[0,:3,-1]
            self.hwf = input_poses[:,:3,:]
            input_poses = input_poses[:,:3,:4]
            H, W, focal = hwf
            H, W = int(H), int(W)
            poses_tensor = torch.from_numpy(input_poses)
            bottom = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
            bottom = bottom.repeat(poses_tensor.shape[0], 1, 1)
            c2ws_gt = c2ws_colmap = torch.cat([poses_tensor, bottom], 1)

        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        _,_,original_h, original_w = imgs.shape
        imgs = torch.nn.functional.interpolate(torch.from_numpy(imgs), (self.h, self.w)).numpy()
        _, _, h, w = imgs.shape


        # Load camera intrinsic
        if load_colmap_poses and is_tank:
            fx, fy = focal, focal
            fx = fx / focal_crop_factor
            fy = fy / focal_crop_factor
            self.H, self.W, self.focal = h, w, fx
            self.K = np.array([[2*fx/original_w, 0, 0, 0], 
                [0, -2*fy/original_h, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]]).astype(np.float32)
            self.K = np.stack([self.K for i in range(len(imgs))])
        elif is_scannet:
            intrinsic = torch.from_numpy(np.load(f'{load_dir}/intrinsic.npy')).float()
            fx,fy = intrinsic[0,0], intrinsic[1,1]
            fx = fx / focal_crop_factor
            fy = fy / focal_crop_factor
            self.H, self.W, self.focal = h, w, fx
            self.K = np.array([[2*fx/original_w, 0, 0, 0], 
                [0, -2*fy/original_h, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]]).astype(np.float32)
            self.K = np.stack([self.K for i in range(len(imgs))])
        elif is_co3d:
            self.K = []
            intrinsic_list = torch.from_numpy(np.load(f'{load_dir}/intrinsic.npy')).float()
            for intrinsic in intrinsic_list:
                fx,fy = intrinsic[0,0], intrinsic[1,1]
                fx = fx / focal_crop_factor
                fy = fy / focal_crop_factor
                self.H, self.W, self.focal = h, w, fx
                K_i = np.array([[2*fx/original_w, 0, 0, 0], 
                    [0, -2*fy/original_h, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]]).astype(np.float32)
                self.K.append(K_i)
            self.K = np.stack(self.K)

        ids = np.arange(imgs.shape[0])
        i_test = ids[int(sample_rate/2)::sample_rate]
        # assert False
        i_train = np.array([i for i in ids if i not in i_test])
        self.i_train = i_train
        self.i_test = i_test
        image_list_train = [img_names[i] for i in i_train]
        image_list_test = [img_names[i] for i in i_test]
        print('test set: ', image_list_test)

        # Load gt pose
        if is_scannet:
            c2ws_gt = torch.from_numpy(np.load(f'{load_dir}/pose.npy')).float()
            T = torch.tensor(np.array([[1, 0, 0, 0],[0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)) # ScanNet coordinate
            c2ws = c2ws_gt @ T
            c2ws_gt = c2ws_colmap = c2ws
        if is_co3d:
            c2ws_gt = torch.from_numpy(np.load(f'{load_dir}/pose.npy')).float()
            c2ws_gt = torch.stack([torch.inverse(pose) for pose in c2ws_gt])
            gt_R = torch.clone(c2ws_gt[:,:3,:3])
            gt_T = torch.clone(c2ws_gt[:,:3,-1])
            # Following CF3DGS, we scale the gt translation vector
            gt_T = gt_T - torch.mean(gt_T[i_train], dim=0)
            gt_T = gt_T / torch.norm(gt_T[i_train])
            # Transform the rotation angle (flip the Y-axis)
            gt_eucler_angles = matrix_to_euler_angles(gt_R, convention='XYZ')
            gt_eucler_angles[:,1:] = gt_eucler_angles[:,1:] * -1
            gt_R = euler_angles_to_matrix(gt_eucler_angles, convention='XYZ')
            # Transform the translation vector (flip the Y-axis)
            gt_T[:,1:] = gt_T[:,1:] * -1
            c2ws_gt = torch.eye(4).unsqueeze(0).repeat(len(gt_R),1,1).float().cuda()
            c2ws_gt[:,:3,:3] = gt_R
            c2ws_gt[:,:3,-1] = gt_T
        # Set dummy c2ws_colmap = c2ws (to avoid dataloading error)
        c2ws = c2ws_gt
        
        self.N_imgs_train = len(i_train)
        self.N_imgs_test = len(i_test)
        
        self.dpt_depth = None
        if mode in ('train','eval_trained', 'render'):
            idx_list = i_train
            self.img_list = image_list_train
        elif mode=='eval':
            idx_list = i_test
            self.img_list = image_list_test
        elif mode=='all':
            idx_list = ids
            self.img_list = img_names

        self.all_imgs = imgs
        # Set dummy value for test images if training mode. 
        if mode in ('train','eval_trained', 'render'):
            self.all_imgs[i_test] = np.zeros_like(self.all_imgs[0])
        self.imgs = imgs[idx_list]
        self.idx_list = idx_list

        self.N_imgs = len(idx_list)
        if c2ws is not None:
            self.c2ws = c2ws[idx_list]
        if load_colmap_poses:
            self.c2ws_colmap = c2ws_colmap[i_train]
        # Load gt_depth
        self.gt_depths = []
        if load_gt_depth and (is_scannet or is_co3d):
            self.gt_depths = [(np.load(f'{load_dir}/gt_depth/depth_{str(int(idx)).zfill(6)}.npz')['pred']) for idx in range(len(imgs))] 
        if len(self.gt_depths) >0: self.gt_depths = np.stack(self.gt_depths)
        # pred_depth_path = os.path.join(load_dir, depth_net)
        # if not use_DPT:
        #     self.dpt_depth = load_depths_npz(self.img_list, pred_depth_path, norm=norm_depth)
        # if load_gt_depth:
        #     self.depth = load_gt_depths(image_list_train, load_dir, crop_ratio=crop_ratio)
        # if load_flow:
        #     self.flow_fw, self.flow_bw = load_flows(image_list_train, load_dir, self.random_ref_interval, H=self.H, W=self.W, crop_ratio=crop_ratio)
        

    def load(self, input_idx_img=None):
        ''' Loads the field.
        '''
        return self.load_field(input_idx_img)

    def load_image(self, idx, data={}):
        image = self.imgs[idx]
        data[None] = image
        # if self.use_DPT:
        #     data_in = {"image": np.transpose(image, (1, 2, 0))}
        #     data_in = self.transform(data_in)
        #     data['normalised_img'] = data_in['image']
        # !!! My change here !!!
        # data['idx'] = idx
        data['idx'] = self.idx_list[idx]
            
    def load_ref_img(self, idx, data={}):
        target_image_idx = self.idx_list[idx]

        ref_image_list = []
        ref_idxs_list = []
        # if self.random_ref:
        random_ref_interval = self.random_ref_interval
        for r in range(len(random_ref_interval)):
            ref_interval = random_ref_interval[r]
            ref_idx = target_image_idx + ref_interval
            # if ref_idx in self.i_test: print("ref_idx in i_test: ", ref_idx)
            if ref_idx in self.i_test: continue
            ref_image = np.ones_like(self.all_imgs[0]) * 10e5 if ref_idx >= len(self.all_imgs) else self.all_imgs[ref_idx]
            ref_image_list.append(ref_image)
            ref_idxs_list.append(ref_idx)

        data['ref_image_list'] = ref_image_list
        # !!! Create a dummy data['ref_imgs'] so that dataloading will not crash. This is not used during training ==> Fix this one later. 
        data['ref_imgs'] = np.ones_like(self.all_imgs[0])
        data['ref_idxs'] = ref_idxs_list
        


    # def load_depth(self, idx, data={}):
    #     depth = self.depth[idx]
    #     data['depth'] = depth

    def load_optical_flow(self, idx, data={}):
        flow_fw_list = []
        flow_bw_list = []
        for ran_idx in range(len(self.random_ref_interval)):
            ref_interval = self.random_ref_interval[ran_idx]
            flow_fw = self.flow_fw[ran_idx][idx]
            flow_bw = self.flow_bw[ran_idx][idx]
            flow_fw_list.append(flow_fw)
            flow_bw_list.append(flow_bw)
        data['flow_fw'] = flow_fw_list
        data['flow_bw'] = flow_bw_list

        
    def load_DPT_depth(self, idx, data={}):
        depth_dpt = self.dpt_depth[idx]
        # ********* My change here: Scale gt depth to stablize the training *********
        # depth_dpt = depth_dpt / np.max(depth_dpt) * 2.0 
        data['dpt'] = depth_dpt


    def load_camera(self, idx, data={}):
        data['scale_mat'] = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]).astype(np.float32)
        target_image_idx = self.idx_list[idx]
        data['idx'] = target_image_idx
        data['camera_mat'] = self.K[target_image_idx]
    
        ref_K_list = []
        # if self.random_ref:
        for ran_idx in range(len(self.random_ref_interval)):
            ref_interval = self.random_ref_interval[ran_idx]
            ref_idx = target_image_idx + ref_interval
            ref_K = np.ones_like(self.K[0]) * 10e5 if ref_idx >= len(self.all_imgs) else self.K[ref_idx]
            ref_K_list.append(ref_K)

        data['ref_camera_mat'] = ref_K_list


    def load_field(self, input_idx_img=None):
        if input_idx_img is not None:
            idx_img = input_idx_img
        else:
            idx_img = 0
        # Load the data
        data = {}
        if not self.mode =='render':
            self.load_image(idx_img, data)
            # if self.load_flow:
            #     self.load_optical_flow(idx_img, data)
            if self.ref_img:
                self.load_ref_img(idx_img, data)
            # if self.load_gt_depth:
            #     self.load_depth(idx_img, data)
            # if self.dpt_depth is not None:
            #     self.load_DPT_depth(idx_img, data)

        if self.with_camera:
            self.load_camera(idx_img, data)
        
        return data





