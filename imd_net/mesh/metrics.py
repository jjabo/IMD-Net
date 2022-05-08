import numpy as np
import torch
import torch.nn.functional as F
# from kaolin.nnsearch import nnsearch
# import kaolin.cuda.sided_distance as sd
# import kaolin.cuda.tri_distance as td

from .utils import compute_face_normals


# class SidedDistanceFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, S1, S2):
#
#         batchsize, n, _ = S1.size()
#         S1 = S1.contiguous()
#         S2 = S2.contiguous()
#         dist1 = torch.zeros(batchsize, n)
#         idx1 = torch.zeros(batchsize, n, dtype=torch.int)
#         dist1 = dist1.cuda()
#         idx1 = idx1.cuda()
#         try:
#             sd.forward(S1, S2, dist1, idx1)
#         except BaseException:
#             sd.forward_cuda(S1, S2, dist1, idx1)
#
#         return idx1.long()


# class SidedDistance(torch.nn.Module):
#     r"""For every point in set1 find the indecies of the closest point in set2
#
#     Args:
#             set1 (torch.cuda.Tensor) : set of pointclouds of shape B x N x 3
#             set2 (torch.cuda.Tensor) : set of pointclouds of shape B x M x 3
#
#     Returns:
#             torch.cuda.Tensor: indecies of the closest points in set2
#
#     Example:
#             >>> A = torch.rand(2,300,3)
#             >>> B = torch.rand(2,200,3)
#             >>> sided_minimum_dist = SidedDistance()
#             >>> indices = sided_minimum_dist(A,B)
#             >>> indices.shape
#             torch.Size([2, 300])
#
#
#     """
#
#     def forward(self, S1: torch.Tensor, S2: torch.Tensor):
#         assert len(S1.shape) == 3
#         assert len(S2.shape) == 3
#         return SidedDistanceFunction.apply(S1, S2).detach()




# class TriangleDistanceFunction(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, points, verts_1, verts_2, verts_3):
#         batchsize, n, _ = points.size()
#         points = points.contiguous()
#         verts_1 = verts_1.contiguous()
#         verts_2 = verts_2.contiguous()
#         verts_3 = verts_3.contiguous()
#
#         dist1 = points.new_zeros(batchsize, n)
#         idx1 = points.new_zeros(batchsize, n, dtype=torch.int)
#         type1 = points.new_zeros(batchsize, n, dtype=torch.int)
#
#         # dist1 = dist1.cuda()
#         # idx1 = idx1.cuda()
#         # type1 = type1.cuda()
#         td.forward_cuda(points, verts_1, verts_2, verts_3, dist1, idx1, type1)
#         ctx.save_for_backward(idx1)
#
#         return dist1[0].detach(), idx1[0].detach().long(), type1[0].long()


# class TriangleDistance(torch.nn.Module):
#     def forward(self, points, verts_1, verts_2, verts_3):
#         points = points.view(1, -1, 3)
#         verts_1 = verts_1.view(1, -1, 3)
#         verts_2 = verts_2.view(1, -1, 3)
#         verts_3 = verts_3.view(1, -1, 3)
#
#         return TriangleDistanceFunction.apply(points, verts_1, verts_2, verts_3)

# def closest_point_to_surface(points: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor):
#
#     # extract triangle defs from mesh
#     v1 = torch.index_select(vertices, 0, faces[:, 0])
#     v2 = torch.index_select(vertices, 0, faces[:, 1])
#     v3 = torch.index_select(vertices, 0, faces[:, 2])
#
#     if points.is_cuda:
#         tri_minimum_dist = TriangleDistance()
#         # pass to cuda
#         distance, indx, dist_type = tri_minimum_dist(points, v1, v2, v3)
#         indx = indx.data.cpu().numpy()
#         dist_type = torch.LongTensor(dist_type.data.cpu().numpy())
#         # reconpute distances to define gradient
#         grad_dist = _recompute_point_to_surface([v1, v2, v3], points, indx, dist_type)
#         #sanity check
#         print(distance.mean(), grad_dist)
#     else:
#         assert False, 'not implemented'
#
#     return 0 #grad_dist


import numpy as np
# def _recompute_point_to_surface(verts, p, indecies, dist_type):
#     # recompute surface based the calcualted correct assignments of points and triangles
#     # and the type of distacne, type 1 to 3 idicates which edge to calcualte to,
#     # type 4 indicates the distance is from a point on the triangle not an edge
#     v1, v2, v3 = verts
#     v1 = v1[indecies]
#     v2 = v2[indecies]
#     v3 = v3[indecies]
#
#     type_1 = (dist_type == 0)
#     type_2 = (dist_type == 1)
#     type_3 = (dist_type == 2)
#     type_4 = (dist_type == 3)
#
#     v21 = v2 - v1
#     v32 = v3 - v2
#     v13 = v1 - v3
#
#     p1 = p - v1
#     p2 = p - v2
#     p3 = p - v3
#
#     dists = []
#     dists.append(_compute_edge_dist(v21[type_1], p1[type_1]).view(-1))
#     dists.append(_compute_edge_dist(v32[type_2], p2[type_2]).view(-1))
#     dists.append(_compute_edge_dist(v13[type_3], p3[type_3]).view(-1))
#
#     if len(np.where(type_4)[0]) > 0:
#         nor = torch.cross(v21[type_4], v13[type_4])
#         dists.append(_compute_planar_dist(nor, p1[type_4]))
#
#     distances = torch.cat(dists)
#
#     return torch.mean((distances))
#
#
# def _compute_edge_dist(v, p):
#     # batched distance between an edge and a point
#     if v.shape[0] == 0:
#         return v
#     dots = _compute_dot(v, p)
#     dots_div = _compute_dot(v, v,)
#     dots = torch.clamp(dots / dots_div, 0.0, 1.0).view(-1, 1)
#     dots = (v * dots) - p
#     dots = _compute_dot(dots, dots)
#     return dots.view(-1, 1)
#
#
# def _compute_dot(p1, p2):
#     # batched dot product
#     return torch.bmm(p1.view(p1.shape[0], 1, 3),
#                      p2.view(p2.shape[0], 3, 1)).view(-1)
#
#
# def _compute_planar_dist(nor, p):
#     # batched distance between a point and a tiangle
#     if nor.shape[0] == 0:
#         return nor
#     dot = _compute_dot(nor, p)
#     dot_div = _compute_dot(nor, nor)
#     return dot * dot / dot_div
#
#
# def closest_point_to_surface2(points: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor):
#
#     # extract triangle defs from mesh
#     v1 = torch.index_select(vertices, 1, faces[:, 0])
#     v2 = torch.index_select(vertices, 1, faces[:, 1])
#     v3 = torch.index_select(vertices, 1, faces[:, 2])
#
#     if points.is_cuda:
#         l_dist = []
#         l_indx = []
#         l_u = []
#         l_v = []
#         l_w = []
#         for i in torch.arange(v1.shape[0]):
#             tri_minimum_dist = TriangleDistance()
#             # pass to cuda
#             distance, indx, dist_type = tri_minimum_dist(points[i], v1[i], v2[i], v3[i])
#
#             u, v, w = _recompute_point_to_surface2([v1[i], v2[i], v3[i]], points[i], indx, dist_type)
#             l_dist.append(distance)
#             l_indx.append(indx)
#             l_u.append(u)
#             l_v.append(v)
#             l_w.append(w)
#
#         distance = torch.stack(l_dist)
#         indx = torch.stack(l_indx)
#         u = torch.stack(l_u)
#         v = torch.stack(l_v)
#         w = torch.stack(l_w)
#     else:
#         assert False, 'not implemented'
#
#     return distance, indx, [u, v, w]
#
# def _recompute_point_to_surface2(verts, p, indices, dist_type):
#     # recompute surface based the calcualted correct assignments of points and triangles
#     # and the type of distacne, type 1 to 3 idicates which edge to calcualte to,
#     # type 4 indicates the distance is from a point on the triangle not an edge
#     v1, v2, v3 = verts
#     v1 = v1[indices]
#     v2 = v2[indices]
#     v3 = v3[indices]
#
#     type_1 = (dist_type == 0)
#     type_2 = (dist_type == 1)
#     type_3 = (dist_type == 2)
#     type_4 = (dist_type == 3)
#
#     u = v1.new_zeros(v1.shape[0], 1)
#     v = v1.new_zeros(v1.shape[0], 1)
#     w = v1.new_zeros(v1.shape[0], 1)
#
#     if torch.any(type_1):
#         u[type_1], v[type_1] = bary_coords_closest_point_on_edge(v1[type_1], v2[type_1], p[type_1])
#     if torch.any(type_2):
#         v[type_2], w[type_2] = bary_coords_closest_point_on_edge(v2[type_2], v3[type_2], p[type_2])
#     if torch.any(type_3):
#         w[type_3], u[type_3] = bary_coords_closest_point_on_edge(v3[type_3], v1[type_3], p[type_3])
#     if torch.any(type_4):
#         closest_points = projection_in_triangle(v1[type_4], v2[type_4], v3[type_4], p[type_4])
#         u[type_4], v[type_4], w[type_4] = bary_coords(
#             v1[type_4].unsqueeze(0),
#             v2[type_4].unsqueeze(0),
#             v3[type_4].unsqueeze(0),
#             closest_points.unsqueeze(0))
#
#     return u, v, w
#
# def bary_coords_closest_point_on_edge(v0: torch.Tensor, v1: torch.Tensor, points: torch.Tensor):
#     v = v1 - v0
#     p = points - v0
#     dots = _compute_dot(v, p)
#     dots_div = _compute_dot(v, v,)
#     u = torch.clamp(dots / dots_div, 0.0, 1.0).view(-1, 1)
#     return 1 - u, u
#
# def closest_point_on_edge(v0: torch.Tensor, v1: torch.Tensor, points: torch.Tensor):
#     v = v1 - v0
#     p = points - v0
#     dots = _compute_dot(v, p)
#     dots_div = _compute_dot(v, v,)
#     dots = torch.clamp(dots / dots_div, 0.0, 1.0).view(-1, 1)
#     closest_points = v0 + v * dots
#     return closest_points
#
# def projection_in_triangle(v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, points: torch.Tensor):
#     # batched distance between a point and a triangle
#     p = points - v0
#     nor = torch.cross(v1 - v0, v2 - v0)
#
#     dot = _compute_dot(nor, p)
#     dot_div = _compute_dot(nor, nor)
#
#     projection = points - (dot / dot_div).view(-1, 1) * nor
#     return projection
#
#
#
# # def get_nearestNeighbor(S1: torch.Tensor, S2: torch.Tensor):
# #     if S1.is_cuda and S2.is_cuda:
# #         sided_minimum_dist = SidedDistance()
# #         closest_index_in_S2 = sided_minimum_dist(S1, S2)
# #     else:
# #         closest_index_in_S2 = nnsearch(S1, S2)
# #
# #     return closest_index_in_S2
#
# def chamfer_distance(S1: torch.Tensor, S2: torch.Tensor,
#                      w1: float = 1., w2: float = 1.):
#     r"""Computes the chamfer distance between two point clouds
#
#     Args:
#             S1 (torch.Tensor): point cloud
#             S2 (torch.Tensor): point cloud
#             w1: (float): weighting of forward direction
#             w2: (float): weighting of backward direction
#
#     Returns:
#             torch.Tensor: chamfer distance between two point clouds S1 and S2
#
#     Example:
#             >>> A = torch.rand(300,3)
#             >>> B = torch.rand(200,3)
#             >>> >>> chamfer_distance(A,B)
#             tensor(0.1868)
#
#     """
#
#     assert (S1.dim() == S2.dim()), 'S1 and S2 must have the same dimesionality'
#     assert (S1.dim() == 3), 'the dimensions of the input must be 3 '
#
#     dist_to_S2 = directed_distance(S1, S2)
#     dist_to_S1 = directed_distance(S2, S1)
#
#     distance = w1 * dist_to_S2 + w2 * dist_to_S1
#
#     return distance
#
#
# def directed_distance(S1: torch.Tensor, S2: torch.Tensor, mean: bool = True):
#     r"""Computes the average distance from point cloud S1 to point cloud S2
#
#     Args:
#             S1 (torch.Tensor): point cloud
#             S2 (torch.Tensor): point cloud
#             mean (bool): if the distances should be reduced to the average
#
#     Returns:
#             torch.Tensor: ditance from point cloud S1 to point cloud S2
#
#     Args:
#
#     Example:
#             >>> A = torch.rand(300,3)
#             >>> B = torch.rand(200,3)
#             >>> >>> directed_distance(A,B)
#             tensor(0.1868)
#
#     """
#
#     if S1.is_cuda and S2.is_cuda:
#         sided_minimum_dist = SidedDistance()
#         closest_index_in_S2 = sided_minimum_dist(S1, S2)
#         closest_S2 = torch.gather(S2, 1, closest_index_in_S2.unsqueeze(-1).expand([-1, -1, 3]))
#
#     else:
#         from time import time
#         closest_index_in_S2 = nnsearch(S1, S2)
#         closest_S2 = S2[closest_index_in_S2]
#
#     dist_to_S2 = (((S1 - closest_S2)**2).sum(dim=-1))
#     if mean:
#         dist_to_S2 = dist_to_S2.mean()
#
#     return dist_to_S2
#
#
#
# def bary_coords(v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, points: torch.Tensor):
#     # void Barycentric(Point a, Point b, Point c, Point p, float & u, float & v, float & w)
#     # {
#     #     Vector v0 = b - a, v1 = c - a, v2 = p - a;
#     #     float d00 = Dot(v0, v0);
#     #     float d01 = Dot(v0, v1);
#     #     float d11 = Dot(v1, v1);
#     #     float d20 = Dot(v2, v0);
#     #     float d21 = Dot(v2, v1);
#     #     float denom = d00 * d11 - d01 * d01;
#     #     v = (d11 * d20 - d01 * d21) / denom;
#     #     w = (d00 * d21 - d01 * d20) / denom;
#     #     u = 1.0f - v - w;
#     # }
#
#     v01 = v1 - v0
#     v02 = v2 - v0
#     v0p = points - v0
#     d00 = torch.sum(v01 * v01, dim=2, keepdims=True)
#     d01 = torch.sum(v01 * v02, dim=2, keepdims=True)
#     d11 = torch.sum(v02 * v02, dim=2, keepdims=True)
#     d20 = torch.sum(v0p * v01, dim=2, keepdims=True)
#     d21 = torch.sum(v0p * v02, dim=2, keepdims=True)
#     denom = d00 * d11 - d01 * d01
#     v = (d11 * d20 - d01 * d21) / denom
#     w = (d00 * d21 - d01 * d20) / denom
#     u = 1.0 - v - w
#     return u, v, w
#
#
def facenormals_angles(v: torch.Tensor, v_gt: torch.Tensor, f: torch.Tensor, return_valid_mask=False):
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


from pytorch_lightning.metrics import Metric
#
#
# class P2PDistance(Metric):
#     def __init__(self, compute_on_step=False):
#         super().__init__(compute_on_step=compute_on_step)
#         self.add_state("mean", default=torch.FloatTensor([0]))
#         self.add_state("max", default=torch.FloatTensor([0]))
#         self.add_state("count", default=torch.tensor(0))
#
#     def update(self, v_pred: torch.Tensor, v_target: torch.Tensor):
#         assert v_pred.shape == v_target.shape
#
#         dist = p2p_dist(v_pred, v_target)
#         self.max = torch.max(self.max, torch.max(dist))
#         self.mean += torch.mean(dist) * v_pred.shape[0]
#         self.count += v_pred.shape[0]
#
#     def compute(self):
#         return self.max, (self.mean.float() / self.count)
#
#
class StatsMeterLazy(Metric):
    def __init__(self, compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step)
        self.add_state("vals", default=torch.FloatTensor(0))

    def update(self, vals: torch.Tensor):
        self.vals = torch.cat((self.vals, vals))

    def compute(self):
        return {'max': self.vals.max().detach().item(),
                'min': self.vals.min().detach().item(),
                'mean': self.vals.mean().detach().item(),
                'std': self.vals.std().detach().item()}
#
#
# class MeanAngularDifference(Metric):
#     def __init__(self, compute_on_step=False):
#         super().__init__(compute_on_step=compute_on_step)
#         self.add_state("mean", default=torch.FloatTensor([0]))
#         self.add_state("max", default=torch.FloatTensor([0]))
#         self.add_state("count", default=torch.tensor(0))
#
#     def update(self, v_pred: torch.Tensor, v_target: torch.Tensor, faces):
#         assert v_pred.shape == v_target.shape
#
#         angles = facenormals_angles(v_pred, v_target, faces, return_valid_mask=True)
#         self.max = torch.max(self.max, torch.max(angles))
#         self.mean += torch.mean(angles) * v_pred.shape[0]
#         self.count += v_pred.shape[0]
#
#     def compute(self):
#         return self.max, (self.mean.float() / self.count)
#
#
# class AverageMeter(Metric):
#     def __init__(self):
#         super().__init__()
#         self.add_state("sum", default=torch.FloatTensor([0]))
#         self.add_state("count", default=torch.tensor(0))
#
#     def update(self, value: torch.Tensor, count):
#         self.sum += value * count
#         self.count += count
#
#     def compute(self):
#         return self.sum / self.count

