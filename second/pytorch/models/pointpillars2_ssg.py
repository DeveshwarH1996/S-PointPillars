"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Code written by Deveshwar Hariharan and Abhishek Ranjan Sing, 2021.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.models.voxel_encoder import get_paddings_indicator, register_vfe
from second.pytorch.models.middle import register_middle
from torchplus.nn import Empty
from torchplus.tools import change_default_args
import numpy as np 
from second.pytorch.models.pointnet_util import PointNetSetAbstraction, PointNetFeaturePropagation


@register_vfe
class PillarFeatureNet2_SSG(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PointNetSetAbstraction 
        and PointNetFeaturePropagation layers. 
        This net performs a similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet2_SSG'
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNetOld layers
        # num_filters = [num_input_features] + list(num_filters)
        # pfn_layers = []
        # for i in range(len(num_filters) - 1):
        #     in_filters = num_filters[i]
        #     out_filters = num_filters[i + 1]
        #     if i < len(num_filters) - 2:
        #         last_layer = False
        #     else:
        #         last_layer = True
        #     pfn_layers.append(
        #         PFNLayer(
        #             in_filters, out_filters, use_norm, last_layer=last_layer))
        # self.pfn_layers = nn.ModuleList(pfn_layers)
    
        self.sa = PointNetSetAbstraction(npoint=64, radius=0.2, nsample=32, in_channel=6+6, mlp=[64,64], group_all=False)
        self.fp = PointNetFeaturePropagation(in_channel=64+9, mlp=[64,64])

        

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        
        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        #Reshape the input data, unroll it and send it to the the set abstraction layer
        features = features.reshape((-1, features.shape[2]))
        features = features.reshape((1, features.shape[0], features.shape[1]))
        features = features.permute(0, 2, 1)
        features_xyz = features[:, 1:4, :]
        
        new_xyz, new_features = self.sa(features_xyz, features)
        features = self.fp(features_xyz, new_xyz, features, new_features)

        features = features.permute(0, 2, 1).contiguous()
        features = features.reshape(-1, 100, 64)


        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        features = torch.max(features,dim=1,keepdim=True)[0]
        
        # # Forward pass through PFNLayers
        # for pfn in self.pfn_layers:
        #     features = pfn(features)m - Rules and Tactics	Apr 30

        return features.squeeze()

# @register_vfe
# class PillarFeatureNetRadius(nn.Module):
#     def __init__(self,
#                  num_input_features=4,
#                  use_norm=True,
#                  num_filters=(64, ),
#                  with_distance=False,
#                  voxel_size=(0.2, 0.2, 4),
#                  pc_range=(0, -40, -3, 70.4, 40, 1)):
#         """
#         Pillar Feature Net.
#         The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
#         similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
#         :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
#         :param use_norm: <bool>. Whether to include BatchNorm.
#         :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
#         :param with_distance: <bool>. Whether to include Euclidean distance to points.
#         :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
#         :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
#         """

#         super().__init__()
#         self.name = 'PillarFeatureNetRadius'
#         assert len(num_filters) > 0
#         num_input_features += 5
#         num_input_features -= 1 # radius xy->r, z->z
#         if with_distance:
#             num_input_features += 1
#         self._with_distance = with_distance

#         # Create PillarFeatureNetOld layers
#         num_filters = [num_input_features] + list(num_filters)
#         pfn_layers = []
#         for i in range(len(num_filters) - 1):
#             in_filters = num_filters[i]
#             out_filters = num_filters[i + 1]
#             if i < len(num_filters) - 2:
#                 last_layer = False
#             else:
#                 last_layer = True
#             pfn_layers.append(
#                 PFNLayer(
#                     in_filters, out_filters, use_norm, last_layer=last_layer))
#         self.pfn_layers = nn.ModuleList(pfn_layers)

#         # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
#         self.vx = voxel_size[0]
#         self.vy = voxel_size[1]
#         self.x_offset = self.vx / 2 + pc_range[0]
#         self.y_offset = self.vy / 2 + pc_range[1]

#     def forward(self, features, num_voxels, coors):
#         device = features.device

#         dtype = features.dtype
#         # Find distance of x, y, and z from cluster center
#         points_mean = features[:, :, :3].sum(
#             dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
#         f_cluster = features[:, :, :3] - points_mean

#         # Find distance of x, y, and z from pillar center
#         f_center = torch.zeros_like(features[:, :, :2])
#         f_center[:, :, 0] = features[:, :, 0] - (
#             coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
#         f_center[:, :, 1] = features[:, :, 1] - (
#             coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
#         features_radius = torch.norm(features[:, :, :2], p=2, dim=2, keepdim=True)
#         features_radius = torch.cat([features_radius, features[:, :, 2:]], dim=2)
#         # Combine together feature decorations
#         features_ls = [features_radius, f_cluster, f_center]
#         if self._with_distance:
#             points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
#             features_ls.append(points_dist)
#         features = torch.cat(features_ls, dim=-1)

#         # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
#         # empty pillars remain set to zeros.
#         voxel_count = features.shape[1]
#         mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
#         mask = torch.unsqueeze(mask, -1).type_as(features)
#         features *= mask

#         # Forward pass through PFNLayers
#         for pfn in self.pfn_layers:
#             features = pfn(features)

#         return features.squeeze()

# @register_vfe
# class PillarFeatureNetRadiusHeight(nn.Module):
#     def __init__(self,
#                  num_input_features=4,
#                  use_norm=True,
#                  num_filters=(64, ),
#                  with_distance=False,
#                  voxel_size=(0.2, 0.2, 4),
#                  pc_range=(0, -40, -3, 70.4, 40, 1)):
#         """
#         Pillar Feature Net.
#         The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
#         similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
#         :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
#         :param use_norm: <bool>. Whether to include BatchNorm.
#         :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
#         :param with_distance: <bool>. Whether to include Euclidean distance to points.
#         :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
#         :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
#         """

#         super().__init__()
#         self.name = 'PillarFeatureNetRadiusHeight'
#         assert len(num_filters) > 0
#         num_input_features += 6
#         num_input_features -= 1 # radius xy->r, z->z
#         if with_distance:
#             num_input_features += 1
#         self._with_distance = with_distance

#         # Create PillarFeatureNetOld layers
#         num_filters = [num_input_features] + list(num_filters)
#         pfn_layers = []
#         for i in range(len(num_filters) - 1):
#             in_filters = num_filters[i]
#             out_filters = num_filters[i + 1]
#             if i < len(num_filters) - 2:
#                 last_layer = False
#             else:
#                 last_layer = True
#             pfn_layers.append(
#                 PFNLayer(
#                     in_filters, out_filters, use_norm, last_layer=last_layer))
#         self.pfn_layers = nn.ModuleList(pfn_layers)

#         # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
#         self.vx = voxel_size[0]
#         self.vy = voxel_size[1]
#         self.x_offset = self.vx / 2 + pc_range[0]
#         self.y_offset = self.vy / 2 + pc_range[1]

#     def forward(self, features, num_voxels, coors):
#         device = features.device

#         dtype = features.dtype
#         # Find distance of x, y, and z from cluster center
#         points_mean = features[:, :, :3].sum(
#             dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
#         f_cluster = features[:, :, :3] - points_mean
#         pp_min = features[:, :, 2:3].min(dim=1, keepdim=True)[0]
#         pp_max = features[:, :, 2:3].max(dim=1, keepdim=True)[0]
#         pp_height = pp_max - pp_min
#         # Find distance of x, y, and z from pillar center
#         f_height = torch.zeros_like(features[:, :, :1])
#         f_height[:] = pp_height
#         f_center = torch.zeros_like(features[:, :, :2])
#         f_center[:, :, 0] = features[:, :, 0] - (
#             coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
#         f_center[:, :, 1] = features[:, :, 1] - (
#             coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
#         features_radius = torch.norm(features[:, :, :2], p=2, dim=2, keepdim=True)
#         features_radius = torch.cat([features_radius, features[:, :, 2:]], dim=2)
#         # Combine together feature decorations
#         features_ls = [features_radius, f_cluster, f_center, f_height]
#         if self._with_distance:
#             points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
#             features_ls.append(points_dist)
#         features = torch.cat(features_ls, dim=-1)

#         # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
#         # empty pillars remain set to zeros.
#         voxel_count = features.shape[1]
#         mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
#         mask = torch.unsqueeze(mask, -1).type_as(features)
#         features *= mask

#         # Forward pass through PFNLayers
#         for pfn in self.pfn_layers:
#             features = pfn(features)

#         return features.squeeze()


# @register_middle
# class PointPillarsScatter(nn.Module):
#     def __init__(self,
#                  output_shape,
#                  use_norm=True,
#                  num_input_features=64,
#                  num_filters_down1=[64],
#                  num_filters_down2=[64, 64],
#                  name='SpMiddle2K'):
#         """
#         Point Pillar's Scatter.
#         Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
#         second.pytorch.voxelnet.SparseMiddleExtractor.
#         :param output_shape: ([int]: 4). Required output shape of features.
#         :param num_input_features: <int>. Number of input features.
#         """

#         super().__init__()
#         self.name = 'PointPillarsScatter'
#         self.output_shape = output_shape
#         self.ny = output_shape[2]
#         self.nx = output_shape[3]
#         self.nchannels = num_input_features

#     def forward(self, voxel_features, coords, batch_size):

#         # batch_canvas will be the final output.
#         batch_canvas = []
#         for batch_itt in range(batch_size):
#             # Create the canvas for this sample
#             canvas = torch.zeros(
#                 self.nchannels,
#                 self.nx * self.ny,
#                 dtype=voxel_features.dtype,
#                 device=voxel_features.device)

#             # Only include non-empty pillars
#             batch_mask = coords[:, 0] == batch_itt
#             this_coords = coords[batch_mask, :]
#             indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
#             indices = indices.type(torch.long)
#             voxels = voxel_features[batch_mask, :]
#             voxels = voxels.t()

#             # Now scatter the blob back to the canvas.
#             canvas[:, indices] = voxels

#             # Append to a list for later stacking.
#             batch_canvas.append(canvas)

#         # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
#         batch_canvas = torch.stack(batch_canvas, 0)

#         # Undo the column stacking to final 4-dim tensor
#         batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny,
#                                          self.nx)
#         return batch_canvas
