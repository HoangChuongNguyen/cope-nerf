import torch
import torch.nn as nn
from model.common import make_c2w, convert3x4_4x4


class PoseRetriever(nn.Module):
    def __init__(self, num_cams, learn_R=True, learn_t=True, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(PoseRetriever, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        else:
            self.init_c2w = nn.Parameter(torch.eye(4).float().unsqueeze(0).repeat(num_cams,1,1), requires_grad=False)
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        cam_id = int(cam_id)
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]
        return c2w
    
   
    
    

