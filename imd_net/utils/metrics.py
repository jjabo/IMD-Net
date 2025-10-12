import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric

from .mesh import compute_face_normals


def facenormals_angles(
    v: torch.Tensor, v_gt: torch.Tensor, f: torch.Tensor, return_valid_mask: bool = False
) -> torch.Tensor:
    # compute face angles of both meshes
    fnormals = compute_face_normals(v, f)

    if return_valid_mask:
        fnormals_gt, valid_mask = compute_face_normals(v_gt, f, return_valid_mask=True)
        # get angles between ground truth and computed mesh and average
        cos_sim = F.cosine_similarity(fnormals[valid_mask], fnormals_gt[valid_mask], 1)
    else:
        fnormals_gt = compute_face_normals(v_gt, f, return_valid_mask=False)
        # get angles between ground truth and computed mesh and average
        cos_sim = F.cosine_similarity(fnormals, fnormals_gt, 2)

    # clamp to valid range
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    # convert to degree
    return torch.acos(cos_sim) * 180.0 / np.pi


def p2p_dist(v: torch.Tensor, v_gt: torch.Tensor):
    return torch.sqrt(torch.sum((v - v_gt) ** 2, dim=2))


class StatsMeterLazy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("vals", default=torch.FloatTensor(0))

    def update(self, vals: torch.Tensor):
        self.vals = torch.cat((self.vals, vals))

    def compute(self):
        return {
            "max": self.vals.max().detach().item(),
            "min": self.vals.min().detach().item(),
            "mean": self.vals.mean().detach().item(),
            "std": self.vals.std().detach().item(),
        }
