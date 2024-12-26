import torch
import numpy as np

def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def rodrigues_torch(rvec):
    theta = torch.norm(rvec)
    if theta < torch.finfo(torch.float32).eps:
        rotation_mat = torch.eye(3, device=rvec.device, dtype=rvec.dtype)
    else:
        r = rvec / theta 
        I = torch.eye(3, device=rvec.device)
        
        r_rT = torch.outer(r, r)
        r_cross = torch.tensor([[0, -r[2], r[1]],
                                [r[2], 0, -r[0]],
                                [-r[1], r[0], 0]], device=rvec.device)
        rotation_mat = torch.cos(theta) * I + (1 - torch.cos(theta)) * r_rT + torch.sin(theta) * r_cross
    
    return rotation_mat


def gridy2x_pers2erp(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx):
    device = gridy.device
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx
    
    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    gridy[:, 0] *= np.tan(np.radians(hFOVy / 2.0))
    gridy[:, 1] *= np.tan(np.radians(wFOVy / 2.0))
    gridy = gridy.double().flip(-1)
    
    x0 = torch.ones(gridy.shape[0], 1, device=device)
    gridy = torch.cat((x0, gridy), dim=-1)
    gridy /= torch.norm(gridy, p=2, dim=-1, keepdim=True)
    
    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    gridy = torch.mm(R1, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R2, gridy.permute(1, 0)).permute(1, 0)

    ### sphere to gridx
    lat = torch.arcsin(gridy[:, 2]) / np.pi * 2
    lon = torch.atan2(gridy[:, 1] , gridy[:, 0]) / np.pi
    gridx = torch.stack((lat, lon), dim=-1)

    # masky
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]

    return gridx.to(torch.float32), mask.to(torch.float32)

def gridy2x_erp2pers(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx):
    device = gridy.device
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx

    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    lat = gridy[:, 0] * np.pi / 2
    lon = gridy[:, 1] * np.pi

    z0 = torch.sin(lat)
    y0 = torch.cos(lat) * torch.sin(lon)
    x0 = torch.cos(lat) * torch.cos(lon)
    gridy = torch.stack((x0, y0, z0), dim=-1).double()

    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    R1_inv = torch.inverse(R1)
    R2_inv = torch.inverse(R2)

    gridy = torch.mm(R2_inv, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R1_inv, gridy.permute(1, 0)).permute(1, 0)

    ### sphere to gridx
    z0 = gridy[:, 2] / gridy[:, 0]
    y0 = gridy[:, 1] / gridy[:, 0]
    gridx = torch.stack((z0, y0), dim=-1).float()

    # masky
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask *= torch.where(gridy[:, 0] < 0, 0, 1)

    return gridx.to(torch.float32), mask.to(torch.float32)