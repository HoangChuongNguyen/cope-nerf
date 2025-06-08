import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.neus_embedder import get_embedder
from utils_poses.pose_pytorch3d import * 


# def euler_angles_to_matrix(euler_angles, convention: str):
#     if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
#         raise ValueError("Invalid input euler angles.")
#     if len(convention) != 3:
#         raise ValueError("Convention must have 3 letters.")
#     if convention[1] in (convention[0], convention[2]):
#         raise ValueError(f"Invalid convention {convention}.")
#     for letter in convention:
#         if letter not in ("X", "Y", "Z"):
#             raise ValueError(f"Invalid letter {letter} in convention string.")
#     matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
#     return functools.reduce(torch.matmul, matrices)
# def _sqrt_positive_part(x):
#     """
#     Returns torch.sqrt(torch.max(0, x))
#     but with a zero subgradient where x is 0.
#     """
#     ret = torch.zeros_like(x)
#     positive_mask = x > 0
#     ret[positive_mask] = torch.sqrt(x[positive_mask])
#     return ret
# def matrix_to_axis_angle(matrix):
#     return quaternion_to_axis_angle(matrix_to_quaternion(matrix))
# def matrix_to_quaternion(matrix):
#     if matrix.size(-1) != 3 or matrix.size(-2) != 3:
#         raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
#     m00 = matrix[..., 0, 0]
#     m11 = matrix[..., 1, 1]
#     m22 = matrix[..., 2, 2]
#     o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
#     x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
#     y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
#     z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
#     o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
#     o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
#     o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
#     return torch.stack((o0, o1, o2, o3), -1)
# def quaternion_to_axis_angle(quaternions):
#     norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
#     half_angles = torch.atan2(norms, quaternions[..., :1])
#     angles = 2 * half_angles
#     eps = 1e-6
#     small_angles = angles.abs() < eps
#     sin_half_angles_over_angles = torch.empty_like(angles)
#     sin_half_angles_over_angles[~small_angles] = (
#         torch.sin(half_angles[~small_angles]) / angles[~small_angles]
#     )
#     # for x small, sin(x/2) is about x/2 - (x/2)^3/6
#     # so sin(x/2)/x is about 1/2 - (x*x)/48
#     sin_half_angles_over_angles[small_angles] = (
#         0.5 - (angles[small_angles] * angles[small_angles]) / 48
#     )
#     return quaternions[..., 1:] / sin_half_angles_over_angles
# import functools
# def _axis_angle_rotation(axis: str, angle):
#     cos = torch.cos(angle)
#     sin = torch.sin(angle)
#     one = torch.ones_like(angle)
#     zero = torch.zeros_like(angle)

#     if axis == "X":
#         R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
#     if axis == "Y":
#         R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
#     if axis == "Z":
#         R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

#     return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class MotionNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(MotionNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        self.scale = scale

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = torch.nn.LeakyReLU(0.2) # nn.Leak(beta=100)

    def compute_consecutive_relative_pose(self, target_cam_idx, total_nb_images, nb_sample_timestep):
        ref_cam_idx = target_cam_idx + 1.0
        time_step = (target_cam_idx / (total_nb_images-1) * 2 - 1) # .to(device=device)
        next_time_step = (ref_cam_idx / (total_nb_images-1) * 2 - 1) #.to(device=device)
        total_nb_sample_timestep = int(nb_sample_timestep * (ref_cam_idx-target_cam_idx))
        time_step_list = torch.linspace(time_step, next_time_step, (total_nb_sample_timestep+1))[:-1]
        time_interval = time_step_list[1] - time_step_list[0]
        R = torch.eye(3).float().cuda()
        T = torch.zeros(3).float().cuda()
        angular_velocity_list, velocity_list = self.forward(time_step_list.view(-1,1).cuda())
        R_list = euler_angles_to_matrix(angular_velocity_list * time_interval, convention='XYZ')
        V_list = velocity_list*time_interval
        for t, time_step_t in enumerate(time_step_list):
            T = R_list[t] @ T.view(3,1) + V_list[t].view(3,1)
            R = R @ R_list[t]
        relative_pose = torch.eye(4).float().cuda()
        relative_pose[:3,:3] = R
        relative_pose[:3,-1] = T.view(1,3)
        # relative_pose = torch.concat([R, T.view(3,1)], dim=1)
        return time_interval, relative_pose

    def compute_relative_camera_pose(self, target_cam_idx, final_ref_cam_idx, total_nb_images, nb_sample_timestep):
        relative_camera_pose = []
        for cam_idx in range(target_cam_idx, final_ref_cam_idx):
            time_interval, pred_pose = self.compute_consecutive_relative_pose(target_cam_idx=cam_idx, total_nb_images=total_nb_images, nb_sample_timestep=nb_sample_timestep)
            relative_camera_pose.append(pred_pose)
        return time_interval, relative_camera_pose
        # relative_camera_pose = torch.stack(relative_camera_pose)

    def compute_w2c_mappings(self, relative_camera_pose):
        """
        This function compute the mapping from the world to each camera.
        The world is chosen to be the first camera in the input list. 
        """
        N = len(relative_camera_pose)
        # Initialize the list of mappings with the identity mapping for the world coordinate system
        w2c = [torch.eye(4).float().cuda()]
        for i in range(N):
            new_mapping = relative_camera_pose[i] @ w2c[-1] # 1 to w, w to 0 -> 1 to 0
            w2c.append(new_mapping)
        w2c = torch.stack(w2c)
        return w2c

    def forward(self, inputs):
        inputs = inputs # * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        x = x * self.scale
        angular_velocity, velocity = x[:,:3], x[:,3:]
        return angular_velocity, velocity


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 4:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :4], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 4):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.sdf(x)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True,
                 use_negative_ray_vector=False,
                 ):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        self.use_negative_ray_vector = use_negative_ray_vector
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.use_negative_ray_vector:
            view_dirs = -1 * view_dirs
            normals = -1 * normals
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).cuda() * torch.exp(self.variance * 10.0)
