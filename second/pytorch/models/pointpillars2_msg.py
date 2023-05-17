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
from second.pytorch.models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction
from second.pytorch.models.pointpillars import PFNLayer


@register_vfe
class PillarFeatureNet2_MSG(nn.Module):
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
        self.name = 'PillarFeatureNet2_MSG'
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        #Defining the setabstraction layer and feature propogation layer
        self.sa0 = PointNetSetAbstractionMsg(10, [0.32, 0.36, 0.4], [6, 8, 10], 6, [[16, 32, 48, 64], [32, 32, 48, 64], [32, 64, 96, 128]])   
        self.sa1 = PointNetSetAbstraction(None, None, None, 64+64+128 + 3, [512, 256, 128], True)

        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

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

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        #Create another mask and anti_mask, features_new = features*mask features_pfn = features*anti_mask

        Threshold = 50

        pfn_features = features[(num_voxels <= Threshold)]

        features = features.permute(0, 2, 1)
        ###
        '''
        pfn_features = features[num_voxels<=10]
        features_points = features_points[(num_voxels>10)]
        features_data = features_data[(num_voxels>10)]
        '''
        features_points = features[(num_voxels > Threshold), :3, :]
        features_data = features[(num_voxels > Threshold), 3:, :]

        # print("features_points shape before processing:", features_points.shape)
        
        features_points, features_data = self.split_and_run(features_points, features_data, self.sa0, 5)
        # print("features_points shape after first processing:", features_points.shape)
        # print("features_data shape after first processing:", features_data.shape)
        features_points, features_data = self.split_and_run(features_points, features_data, self.sa1, 5)
        # print("features_points shape after second processing:", features_points.shape)
        # print("features_data shape before squeezing:", features_data.shape)
        features_data = features_data.squeeze()
        # print("features_data shape after squeezing:", features_data.shape)
        
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            pfn_features = self.PFN_split_and_run(pfn_features, pfn, 5)
        
        pfn_features = pfn_features.squeeze()

        features = torch.zeros(features.shape[0], features_data.shape[1]).to(device=features_data.device, dtype=features_data.dtype)
        features[(num_voxels > Threshold)] = features_data
        features[(num_voxels <= Threshold)] = pfn_features
        # print("\n")

        return features

    def split_and_run(self, input_points, input_data, model, num_sections):
        """
        splits the input data into sections, runs the model on those batches, returns the concatenated outputs
        """
        if (input_data is not None):
            assert input_points.shape[0] == input_data.shape[0]
            assert input_points.shape[2] == input_data.shape[2]

        points_sections = torch.chunk(input_points, num_sections)
        if (input_data is not None):
            data_sections = torch.chunk(input=input_data, chunks=num_sections, dim=0)

        points_outs = []
        data_outs = []
        for i in range(len(points_sections)):
            p = points_sections[i]
            d = None
            if input_data is not None:
                d = data_sections[i]

            p_out, d_out = model(p, d)
            # print("batch", i, "final shape of points:", p_out.shape)
            # print("batch", i, "final shape of data:", d_out.shape)
            points_outs.append(p_out)
            data_outs.append(d_out)
        
        points_outs = torch.cat(points_outs, dim=0)
        data_outs = torch.cat(data_outs, dim=0)

        return points_outs, data_outs

    def PFN_split_and_run(self, input_points, model, num_sections):
        """
        splits the input data into sections, runs the model on those batches, returns the concatenated outputs
        """
        points_sections = torch.chunk(input_points, num_sections)

        points_outs = []
        for i in range(len(points_sections)):
            p = points_sections[i]
            points_outs.append(model(p))
        
        points_outs = torch.cat(points_outs, dim=0)

        return points_outs