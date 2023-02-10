#!/usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os
os.chdir("/home/ecoprt/PointPillars2/second")
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.append(os.path.abspath('../'))

from second.data.nuscenes_dataset import NuScenesDataset

import open3d as o3d

# import rospy
# from second.pytorch.builder import input_reader_builder
# from vision_msgs.msg import *
# from geometry_msgs.msg import *
# from sensor_msgs.msg import *
# from tf.transformations import quaternion_from_euler

import torch
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool

from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)

from second.pytorch.train import example_convert_to_torch
from second.data.preprocess import merge_second_batch
import torchplus

# import ros_numpy


def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qx, qy, qz, qw]

def box_center_to_corner(box_center):
    # To return
    corner_boxes = np.zeros((8, 3))

    translation = box_center[0:3]
    h, w, l = box_center[3], box_center[4], box_center[5]
    rotation = box_center[6]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    return corner_box.transpose()

class Detector:
    
   def __init__(self, scene):
     self.scene = scene

   def run(self, config_path, model_dir):
    # assert len(kwargs) == 0
    model_dir = str(Path(model_dir).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model_dir = Path(model_dir)
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time=True).to(device)
    if train_cfg.enable_mixed_precision:
        # net.half()
        # print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator


    assert model_dir is not None
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net], device)

    batch_size = 1
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    # if train_cfg.enable_mixed_precision:
    #     float_dtype = torch.float16
    # else:
    float_dtype = torch.float32

    net.eval()
    detection = None

    example = eval_dataset.__getitem__(self.scene)
    example.pop("lidar")
    example = merge_second_batch([example])
    example = example_convert_to_torch(example, float_dtype, device=device)



    with torch.no_grad():
        if train_cfg.enable_mixed_precision:
            with torch.cuda.amp.autocast():
                detection = net(example)
        else:
            detection = net(example)

    detection = detection[0]
    mask = detection["scores"] > 0.4

    # print(mask)

    boxes_lidar = detection["box3d_lidar"][mask].detach().cpu().numpy()
    labels = detection["label_preds"][mask].detach().cpu().numpy()
    # print(labels)
    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -30, -3, 50, 30, 1]
    obj = eval_dataset.__getitem__(self.scene)
    points = obj["lidar"]["points"][:, :3]
    gt_boxes_lidar = obj["lidar"]["annotations"]["boxes"]
    # print(boxes_lidar)
    print(type(points))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    new_pcd = o3d.t.geometry.PointCloud(points)
    for attr in new_pcd.point:
        new_pcd.point[attr] = new_pcd.point[attr].to(o3d.core.float64)
    o3d.t.io.write_point_cloud("./test.ply", new_pcd, write_ascii=False, compressed=False)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render = vis.get_render_option()
    render.background_color = np.asarray([0, 0, 0])
    render.point_size = 0.1
    render.line_width = 2.5
    vis.add_geometry(pcd)

    color_pallete = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]

    for i in range(len(boxes_lidar)):
        # print(boxes_lidar[i])
        corner_box = box_center_to_corner(boxes_lidar[i])

        # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                [4, 5], [5, 6], [6, 7], [4, 7],
                [0, 4], [1, 5], [2, 6], [3, 7]]

        # Use the same color for all line
        colors = [[1, 0, 0] for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Display the bounding boxes:
        vis.add_geometry(line_set)

    for i in range(len(gt_boxes_lidar)):
        #print(gt_boxes_lidar[i])
        corner_box = box_center_to_corner(gt_boxes_lidar[i])

        # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                [4, 5], [5, 6], [6, 7], [4, 7],
                [0, 4], [1, 5], [2, 6], [3, 7]]

        # Use the same color for all lines
        colors = [[1, 1, 1] for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Display the bounding boxes:
        vis.add_geometry(line_set)
    



    vis.run()
    # bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    # bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)
    #plt.imshow(bev_map)
    vis.destroy_window()
    # return detection
    


if __name__ == "__main__":
    
    scene = int(sys.argv[1])
    detector = Detector(scene)
    detector.run("/home/ecoprt/PointPillars2/second/configs/nuscenes/all.pp2.largea.supes.config", "/home/ecoprt/PointPillars2/model_dirs/pp2_model_dir_v0.5_p512")
