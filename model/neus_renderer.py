import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic

epsilon = 1e-6
def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).cuda().split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).cuda().split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).cuda().split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    u = u.cuda()
    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def sample_pdf_naive(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    u = u.cuda()
    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer(nn.Module):
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 motion_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb, 
                 n_max_network_queries,
                 importance_sampling_start, 
                 naive_render):
        super(NeuSRenderer, self).__init__()
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.motion_network = motion_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.n_max_network_queries = n_max_network_queries
        self.importance_sampling_start = importance_sampling_start
        self.naive_render = naive_render

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).cuda()], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).cuda(), 1. - alpha + epsilon], -1), -1)[:, :-1]
        # weights = alpha * torch.cumprod(torch.cat([torch.ones((batch_size, 1), device=alpha.device), 1.-alpha + epsilon ], -1), -1)[:, :-1]

        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 10.0) | (radius[:, 1:] < 10.0)
        # TODO!!! : Not sure whether we need to have inside_sphere or not
        inside_sphere = torch.ones_like(inside_sphere)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]).cuda(), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # weights = alpha * torch.cumprod(torch.cat([torch.ones((batch_size, 1), device=alpha.device), 1.-alpha + epsilon ], -1), -1)[:, :-1]
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def up_sample_naive(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 10.0) | (radius[:, 1:] < 10.0)
        # TODO!!! : Not sure whether we need to have inside_sphere or not
        inside_sphere = torch.ones_like(inside_sphere)
        sdf = sdf.reshape(batch_size, n_samples)
        # prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        # prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        # mid_sdf = (prev_sdf + next_sdf) * 0.5
        # cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # # ----------------------------------------------------------------------------------------------------------
        # # Use min value of [ cos, prev_cos ]
        # # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # # robust when meeting situations like below:
        # #
        # # SDF
        # # ^
        # # |\          -----x----...
        # # | \        /
        # # |  x      x
        # # |---\----/-------------> 0 level
        # # |    \  /
        # # |     \/
        # # |
        # # ----------------------------------------------------------------------------------------------------------
        # prev_cos_val = torch.cat([torch.zeros([batch_size, 1]).cuda(), cos_val[:, :-1]], dim=-1)
        # cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        # cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        # cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        # dist = (next_z_vals - prev_z_vals)
        # prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        # next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        # prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        # next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        # alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        alpha = self.logistic_density(sdf, 1/inv_s)
        # alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # weights = alpha * torch.cumprod(torch.cat([torch.ones((batch_size, 1), device=alpha.device), 1.-alpha + epsilon ], -1), -1)[:, :-1]
        # print('a')
        # print(z_vals.shape, weights.shape)
        # print(n_importance)
        # assert False
        z_samples = sample_pdf_naive(z_vals, weights, n_importance, det=True).detach()

        return z_samples

    def cat_z_vals(self, rays_o, rays_d, time_step, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            pts = pts.reshape(-1, 3)
            time_step_ = time_step.unsqueeze(0).repeat(pts.shape[0],1)
            pts_time = torch.cat([pts, time_step_], dim=-1)
            new_sdf = self.sdf_network.sdf(pts_time).reshape(batch_size, n_importance)

            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)
        return z_vals, sdf

    def logistic_density(self, x, s):
        exp_term = torch.exp(-s * x)
        numerator = s * exp_term
        denominator = (1 + exp_term) ** 2
        return numerator / denominator


    def render_core(self,
                    rays_o,
                    rays_d,
                    rays_d_norm,
                    time_step,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    all_z_vals=None,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    it=-1, eval=False):
        batch_size, n_samples = z_vals.shape

        # # print(rays_o.shape) # torch.Size([512, 3])
        # # print(rays_d.shape) # torch.Size([512, 3])
        # # print(z_vals.shape) # torch.Size([512, 128])
        # # print(z_vals[:,0])
        # # print(sample_dist) # 0.03125

        # # Section length
        # dists = z_vals[..., 1:] - z_vals[..., :-1]
        # dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).cuda()], -1)
        # mid_z_vals = z_vals + dists * 0.5
        
        # # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).cuda()], -1)
        mid_z_vals = z_vals + dists * 0.5
        
        # # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        # # Predict SDF
        # !!! New changes here: concat time and pts and use them as input to the SDF network
        time_step_rep = time_step.unsqueeze(0).repeat(pts.shape[0],1)
        pts_time = torch.cat([pts, time_step_rep], dim=-1)
        # !!! New changes here: Use pts_time as input to predict sdf
        sdf_nn_output = sdf_network(pts_time)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts_time.detach()).squeeze() # Gradient of sdf w.r.t points and time # nb_points, 4
        normals, sdf_flows = gradients[:,:3], gradients[:,[-1]]
        sampled_color = color_network(pts_time, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1/1e3,1/1e-3)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        # QUESTION: is this correct? 
        # true_cos = (dirs * gradients).sum(-1, keepdim=True)
        true_cos = (dirs * normals).sum(-1, keepdim=True)

        # # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -( F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                      F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # # Estimate signed distances at section points 
        # # This part is the first order approximation using Taylor expansion. 
        # #  f(x+t)≈f(x)+t⋅∇f(x)⋅d
        # #   with ∇f(x)⋅d is the true_cos, annealed by computing iter_cos
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        
        # # Compute alpha equation 13 in the paper
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 10.0).float().detach()
        # TODO!!! : Not sure whether we need to have inside_sphere or not
        inside_sphere = torch.ones_like(inside_sphere)

        # # Render with background
        # if background_alpha is not None:
        #     # If a point has norm >= 1, it is classified as outside_sphere, and therefore we use background alpha and color (instead of inside alpha and color)
        #     alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
        #     sampled_color = sampled_color * inside_sphere[:, :, None] +\
        #                     background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
        #     # !!!! My changes here => This might not be neccessary !!!!
        #     z_vals = z_vals * inside_sphere + all_z_vals[:,:n_samples] * (1.0 - inside_sphere) # n_rays n_points
        #     # Combine inside and outside predictions
        #     alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
        #     sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)
        #     # !!!! My changes here => This might not be neccessary !!!!
        #     z_vals = torch.cat([z_vals, all_z_vals[:, n_samples:]], dim=1)

        # This is the weights T_i alpha_i in the paper (equation 11)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # weights = alpha * torch.cumprod(torch.cat([torch.ones((batch_size, 1), device=alpha.device), 1.-alpha + epsilon ], -1), -1)[:, :-1]
        # Calculate this for logging only
        with torch.no_grad():   
            weight_inside = torch.sum(weights[:,:n_samples], dim=-1).detach()
            weight_outside = torch.sum(weights[:,n_samples:], dim=-1).detach()
        weights_sum = weights.sum(dim=-1, keepdim=True)

        # Compute final color
        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        # Compute predicted 'depth'
        depth_pred = (z_vals * weights).sum(dim=1).unsqueeze(-1)
        with torch.no_grad(): weighted_z_vals = torch.clone(depth_pred)
        if eval:
            depth_pred = depth_pred / rays_d_norm # change distance to depth, consistent with gt depth for evaluation

        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)


        # Eikonal loss
        # relax_inside_sphere = (pts_norm < 1.2).float().detach()  # World center is the center of the sphere?
        # # TODO: Not sure whether we need to have inside_sphere or not
        # relax_inside_sphere = torch.ones_like(relax_inside_sphere)
        # gradient_error = (torch.linalg.norm(normals.reshape(batch_size, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2
        # gradient_error = # (relax_inside_sphere * gradient_error).sum() / ((relax_inside_sphere).sum() + 1e-5)

        return {
            'color': color,
            'depth_pred': depth_pred,
            'weighted_z_vals': weighted_z_vals,
            'sdf': sdf,
            'dists': dists,
            'normals': normals.reshape(batch_size, n_samples, 3),
            'sdf_flows': sdf_flows.reshape(batch_size, n_samples, 1),
            'sampled_points': pts.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            # 'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'weight_inside': weight_inside,
            'weight_outside': weight_outside
        }

        
    def forward(self, rays_o, rays_d, ray_d_norm, time_step, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, it=-1, eval=False):
        batch_size = len(rays_o)

        if it >= self.importance_sampling_start:
            n_samples = self.n_samples
            n_importance = self.n_importance
        else:    
            n_samples = self.n_samples + self.n_importance
            n_importance = 0
        # n_samples = self.n_samples

        # n_samples = 64
        # sample_dist is distance between 2 samples
        sample_dist = (far[0,0]-near[0,0]) / n_samples   # Assuming the region of interest is a unit sphere. 
        # Corse sampling: Uniformly sample points z_vals with distance sample_dist
        z_vals = torch.linspace(0.0, 1.0, n_samples).cuda() 
        z_vals = near * (1. - z_vals[None, :]) + far*z_vals[None, :]

        # z_vals_outside are points that are not in the unit sphere.
        z_vals_outside = None
        if self.n_outside > 0: # By default, n_outside = 0 for conf with mask; and n_outside = 32 for w/o mask
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside).cuda()

        # Use nope-nerf-noise
        add_noise = not eval
        if add_noise:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand([batch_size, z_vals.shape[-1]]).cuda()
            z_vals = lower + (upper - lower) * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Fine sampling - Imporatance sampling
        if n_importance > 0:
            with torch.no_grad():
                # Compute the SDF of each points sampled uniformly previously
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # batch len(z_vals) 3 
                pts = pts.reshape(-1, 3)
                time_step_ = time_step.unsqueeze(0).repeat(pts.shape[0],1)
                pts_time = torch.cat([pts, time_step_], dim=-1)
                sdf = self.sdf_network.sdf(pts_time).reshape(batch_size, n_samples) # batch len(z_vals) 
                for i in range(self.up_sample_steps):
                    # Sample more points based on the currently predicted SDF
                    # new_z_vals of shape torch.Size([512, 16])
                    if self.naive_render:
                        new_z_vals = self.up_sample_naive(rays_o,
                                                    rays_d,
                                                    z_vals,
                                                    sdf,
                                                    n_importance // self.up_sample_steps,
                                                    64 * 2**i)       
                    else:
                        new_z_vals = self.up_sample(rays_o,
                                                    rays_d,
                                                    z_vals,
                                                    sdf,
                                                    n_importance // self.up_sample_steps,
                                                    64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  time_step,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = n_samples + n_importance

        # Background model: check this one later ("The outside scene is represented using NeRF++", NERF++ use density instead of SDF)
        # This model compute rgb and alpha for both (1) points inside the sphere and (2) points ourside the sphere. 
        # Later on, if a points pts_norm > 1, the inside points are classified as outside points, and thus use the predicted background rgb and alpha. 
        # (see this if statement "if background_alpha is not None" in render_core function)
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)
            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']
        else:
            z_vals_feed = z_vals
        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    ray_d_norm,
                                    time_step,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    all_z_vals=z_vals_feed,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    it=it, eval=eval)

        color_fine = ret_fine['color']
        depth_pred = ret_fine['depth_pred']
        weighted_z_vals = ret_fine['weighted_z_vals']

        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        normals = ret_fine['normals']
        sdf_flows = ret_fine['sdf_flows']
        sampled_points = ret_fine['sampled_points']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)
        sdf = ret_fine['sdf']
        return {
            'sdf': sdf,
            'color_fine': color_fine,
            'depth_pred': depth_pred, 
            'weighted_z_vals': weighted_z_vals,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'normals': normals,
            'sdf_flows': sdf_flows,
            "sampled_points": sampled_points,
            'weights': weights,
            # 'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'weight_inside': ret_fine['weight_inside'],
            'weight_outside': ret_fine['weight_outside'],
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
