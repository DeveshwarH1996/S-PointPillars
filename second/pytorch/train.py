import copy
import json
import os
os.chdir("/home/ecoprt/PointPillars2/second")
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.append(os.path.abspath('../'))
from pathlib import Path
import pickle
import shutil
import time
import re 
import fire
import numpy as np
import torch
from google.protobuf import text_format

import subprocess

import second.data.kitti_common as kitti
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.log_tool import SimpleModelLog
from second.utils.progress_bar import ProgressBar
import psutil

from torch.cuda.amp.grad_scaler import GradScaler

from numba import cuda as c
c.detect()

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v, device=device)
        elif k == "lidar":
            continue
        else:
            example_torch[k] = v
    return example_torch


def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim
    net = second_builder.build(
        model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net

def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def freeze_params(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    remain_params = []
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                continue 
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                continue 
        remain_params.append(p)
    return remain_params

def freeze_params_v2(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                p.requires_grad = False
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                p.requires_grad = False

def filter_param_dict(state_dict: dict, include: str=None, exclude: str=None):
    assert isinstance(state_dict, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    res_dict = {}
    for k, p in state_dict.items():
        if include_re is not None:
            if include_re.match(k) is None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is not None:
                continue 
        res_dict[k] = p
    return res_dict

def run_voxel_stats(config_path, train_file_name=None, id=0, max_points=None):
    import csv

    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    if max_points is not None:
        assert type(max_points) is int
        model_cfg.voxel_generator.max_number_of_points_per_voxel = max_points
    
    print(model_cfg.voxel_generator)

    device = torch.device(f"cuda:{id}" if torch.cuda.is_available() else "cpu")
    net = build_network(model_cfg, False).to(device)

    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn,
        drop_last=False)
    
    print("training dataset len:", len(dataset), "\n\n\n\n\n\n")

    # eval_dataset = input_reader_builder.build(
    #     input_cfg,
    #     model_cfg,
    #     training=False,
    #     voxel_generator=voxel_generator,
    #     target_assigner=target_assigner)
    # eval_dataloader = torch.utils.data.DataLoader(
    #     eval_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=input_cfg.preprocess.num_workers,
    #     pin_memory=False,
    #     collate_fn=merge_second_batch)
    
    # print("eval dataset len:", len(dataset), "\n\n\n\n\n\n")

    print ("len of training dataloader:", len(dataloader))
    # print ("len of eval dataloader:", len(eval_dataloader))

    def process(ex_raw, csv_handle):
        example = example_convert_to_torch(ex_raw, torch.float32, device=device)
        voxels = example["voxels"]
        num_points = example["num_points"]
        assert voxels.shape[0] == num_points.shape[0]
        assert len(num_points.shape) == 1 # multi-gpu not supported

        # num_points = num_points[num_points > 1]

        max_pts = voxels.shape[1]
        tv = voxels.shape[0]
        l2 = round(((num_points < 2).sum().type(torch.float16) / tv).item(), 2)
        l5 = round(((num_points < 5).sum().type(torch.float16) / tv).item(), 2)
        l10 = round(((num_points < 10).sum().type(torch.float16) / tv).item(), 2)
        m10 = round(((num_points > int(max_pts * 0.1)).sum().type(torch.float16) / tv).item(), 2)
        m25 = round(((num_points > int(max_pts * 0.25)).sum().type(torch.float16) / tv).item(), 2)
        m50 = round(((num_points > int(max_pts * 0.5)).sum().type(torch.float16) / tv).item(), 2)
        m75 = round(((num_points > int(max_pts * 0.75)).sum().type(torch.float16) / tv).item(), 2)
        avgf = round((num_points.type(torch.float16) / max_pts).mean().item(), 2)
        avgm = round((num_points.type(torch.float16) / max_pts).median().item(), 2)
        max = num_points.max().item()
        min = num_points.min().item()
        median = num_points.median().item()
        mean = num_points.type(torch.float16).mean().item()

        stat_list = [l2, l5, l10, m10, m25, m50, m75, tv, avgf, avgm, max, min, median, mean]
        # print(max_pts)
        if writer is not None:
            writer.writerow(stat_list)
        else:
            print("<2, <5, <10, >10%, >25%, >50%, >75%, total voxels, avg % filled, median % filled, max, min, median, mean:", stat_list)
        
        return stat_list

    header = ["<2", "<5", "<10", ">10%", ">25%", ">50%", ">75%", "total_voxels", "avg_filled", "med_filled", "max", "min", "median", "mean"]

    writer = None
    f = None
    if train_file_name is not None:
        f = open(train_file_name, 'w', encoding='UTF8', newline='')
        writer = csv.writer(f)
        writer.writerow(header)
    
    try:
        counter = 0
        bar = ProgressBar()
        bar.start(len(dataloader))
        total_stats = []
        for ex in dataloader:
            process(ex, writer)
            # total_stats.append(stat_list)
            counter+=1
            bar.print_bar()
        # print(total_stats)
        # total_stats = torch.Tensor(total_stats)
        # print(total_stats)
        # total_stats = np.average(total_stats, axis=0)
    except KeyboardInterrupt as e:
        print(counter, "\n")
        if f is not None:
            f.flush()
            f.close()
        raise e

    # print("\nmean:", total_stats.mean(dim=0).detach().numpy().tolist())
    # print("median:", total_stats.median(dim=0)[0].detach().numpy().tolist())
    # print("min:", total_stats.min(dim=0)[0].detach().numpy().tolist())
    # print("max:", total_stats.max(dim=0)[0].detach().numpy().tolist())
    # print(header, "\n", total_stats)
    # writer.writerow(total_stats)

    print("training count:", counter)
    if f is not None:
        f.flush()
        f.close()
    
    # writer = None
    # f = None
    # if eval_file_name is not None:
    #     f = open(eval_file_name, 'w', encoding='UTF8', newline='')
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    
    # try:
    #     counter = 0
    #     bar = ProgressBar()
    #     bar.start(len(dataloader))
    #     for ex in eval_dataloader:
    #         process(ex, writer)
    #         counter+=1
    #         bar.print_bar()
    # except KeyboardInterrupt as e:
    #     print(counter)
    #     if f is not None:
    #         f.flush()
    #         f.close()
    #     raise e

    # print("eval count:", counter)
    # if f is not None:
    #     f.flush()
    #     f.close()
    
def voxel_stats(config_path, file_name, max_points_list):
    for points in max_points_list:
        print(points)
        print(int(points))
        run_voxel_stats(config_path, file_name + "_" + str(points) + ".csv", 0, points)

def send_mail(to, message):
    try :
        import smtplib
 
        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)
        
        # start TLS for security
        s.starttls()

        email = os.environ["em"]
        pw = os.environ["emp"]
        
        # Authentication
        s.login(email, pw)
        
        # sending the mail
        s.sendmail(email, to, message)

        print(message)
        
        # terminating the session
        s.quit()
    except:
        print("unable to send message")
    
    

def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pretrained_path=None,
          pretrained_include=None,
          pretrained_exclude=None,
          freeze_include=None,
          freeze_exclude=None,
          multi_gpu=False,
          measure_time=False,
          resume=False,
          gpu_ids=None,
          gpu_id=None,
          eval_name=None):
    """train a VoxelNet model specified by a config file.
    """
    if multi_gpu and gpu_ids:
        device = torch.device(f"cuda:{str(gpu_ids[0])}")
        print(device)
    else:
        if gpu_id is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
            print(device)
    
    model_dir = str(Path(model_dir).resolve())
    if create_folder:
        if Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)
    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume.")
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time).to(device)

    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    print("num parameters:", len(list(net.parameters())))
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    if pretrained_path is not None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = filter_param_dict(pretrained_dict, pretrained_include, pretrained_exclude)
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_pretrained_dict[k] = v        
        print("Load pretrained parameters:")
        for k, v in new_pretrained_dict.items():
            print(k, v.shape)
        model_dict.update(new_pretrained_dict) 
        net.load_state_dict(model_dict)
        freeze_params_v2(dict(net.named_parameters()), freeze_include, freeze_exclude)
        net.clear_global_step()
        net.clear_metrics()
    if multi_gpu:
        if gpu_ids:
            net_parallel = torch.nn.DataParallel(net, device_ids=gpu_ids)
            device = torch.device(f"cuda:{str(gpu_ids[0])}")
            print(device)
        else:
            net_parallel = torch.nn.DataParallel(net)
    else:
        net_parallel = net
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor
    fastai_optimizer = optimizer_builder.build(
        optimizer_cfg,
        net,
        mixed=False,
        loss_scale=loss_scale)
    if loss_scale < 0:
        loss_scale = "dynamic"
    if train_cfg.enable_mixed_precision:
        # max_num_voxels = input_cfg.preprocess.max_number_of_voxels * input_cfg.batch_size
        # assert max_num_voxels < 65535, "spconv fp16 training only support this"
        # from apex import amp


        amp_optimizer = fastai_optimizer
        scaler = GradScaler()
        net.metrics_to_float()
        torchplus.train.try_restore_latest_checkpoints(model_dir, { "amp_optimizer": scaler })
    else:
        amp_optimizer = fastai_optimizer
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [fastai_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)
    # if train_cfg.enable_mixed_precision:
    #     float_dtype = torch.float16
    # else:
    float_dtype = torch.float32

    if multi_gpu:
        if gpu_ids:
            num_gpu = len(gpu_ids)
        else:
            num_gpu = torch.cuda.device_count()
        
        assert num_gpu <= torch.cuda.device_count()

        print(f"MULTI-GPU: use {num_gpu} gpu")
        collate_fn = merge_second_batch_multigpu
    else:
        collate_fn = merge_second_batch
        num_gpu = 1

    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=multi_gpu)
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size * num_gpu,
        shuffle=True,
        num_workers=input_cfg.preprocess.num_workers * num_gpu,
        pin_memory=False,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=not multi_gpu)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size, # only support multi-gpu train
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    ######################
    # TRAINING
    ######################
    model_logging = SimpleModelLog(model_dir)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag="config")
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval
    steps_per_save = train_cfg.save_summary_steps
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch


    amp_optimizer.zero_grad()
    step_times = []
    step = start_step
    try:
        while True:
            if clear_metrics_every_epoch:
                net.clear_metrics()
                print("##############################################")
                print("clearing metrics: Starting a new epoch")
                print("##############################################")
            else:
                print("##############################################")
                print("NOT clearing metrics: Continuing from previous epoch")
                print("##############################################")
            t1 = time.time()
            for example in dataloader:
                # print("getting next datapoints:", time.time() - t1)
                lr_scheduler.step(net.get_global_step())
                time_metrics = example["metrics"]
                example.pop("metrics")
                example_torch = example_convert_to_torch(example, float_dtype, device=device)


                batch_size = example["anchors"].shape[0]

                if train_cfg.enable_mixed_precision:
                    with torch.cuda.amp.autocast():
                        # print("using mixed precision")
                        ret_dict = net_parallel(example_torch)
                        cls_preds = ret_dict["cls_preds"]
                        loss = ret_dict["loss"].mean()
                        cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                        loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                        cls_pos_loss = ret_dict["cls_pos_loss"].mean()
                        cls_neg_loss = ret_dict["cls_neg_loss"].mean()
                        loc_loss = ret_dict["loc_loss"]
                        cls_loss = ret_dict["cls_loss"]
                        
                        cared = ret_dict["cared"]
                        labels = example_torch["labels"]
                else:
                    ret_dict = net_parallel(example_torch)
                    cls_preds = ret_dict["cls_preds"]
                    loss = ret_dict["loss"].mean()
                    cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                    loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                    cls_pos_loss = ret_dict["cls_pos_loss"].mean()
                    cls_neg_loss = ret_dict["cls_neg_loss"].mean()
                    loc_loss = ret_dict["loc_loss"]
                    cls_loss = ret_dict["cls_loss"]
                    
                    cared = ret_dict["cared"]
                    labels = example_torch["labels"]

                if train_cfg.enable_mixed_precision:
                    # with amp.scale_loss(loss, amp_optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                    scaler.scale(loss).backward()
                    scaler.unscale_(amp_optimizer)
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 20.0)
                if train_cfg.enable_mixed_precision:
                    scaler.step(amp_optimizer)
                    scaler.update()
                else:
                    amp_optimizer.step()

                amp_optimizer.zero_grad()
                net.update_global_step()
                net_metrics = net.update_metrics(cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)

                step_time = (time.time() - t)
                step_times.append(step_time)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                if 'anchors_mask' not in example_torch:
                    num_anchors = example_torch['anchors'].shape[1]
                else:
                    num_anchors = int(example_torch['anchors_mask'][0].sum())
                global_step = net.get_global_step()

                if global_step % display_step == 0:
                    if measure_time:
                        for name, val in net.get_avg_time_dict().items():
                            print(f"avg {name} time = {val * 1000:.3f} ms")

                    loc_loss_elem = [
                        float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                              batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["runtime"] = {
                        "step": global_step,
                        "steptime": np.mean(step_times),
                    }
                    metrics["runtime"].update(time_metrics[0])
                    step_times = []
                    metrics.update(net_metrics)
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        dir_loss_reduced = ret_dict["dir_loss_reduced"].mean()
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())

                    metrics["misc"] = {
                        "num_vox": int(example_torch["voxels"].shape[0]),
                        "num_pos": int(num_pos),
                        "num_neg": int(num_neg),
                        "num_anchors": int(num_anchors),
                        "lr": float(amp_optimizer.lr),
                        "mem_usage": psutil.virtual_memory().percent,
                    }
                    model_logging.log_metrics(metrics, global_step)

                if global_step % steps_per_save == 0:
                    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                net.get_global_step())
                    if train_cfg.enable_mixed_precision:
                        torchplus.train.save_models(model_dir, { "amp_optimizer": scaler },
                                                net.get_global_step())

                if global_step % steps_per_eval == 0:
                    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                net.get_global_step())
                    if train_cfg.enable_mixed_precision:
                        torchplus.train.save_models(model_dir, { "amp_optimizer": scaler },
                                                net.get_global_step())
                    
                    send_mail("8589146150@tmomail.net", "step " + str(global_step) + "\nstarting eval")
        
                    if eval_name:
                        print(eval_name)
                        model_logging.log_text("#################################",
                                            global_step)
                        model_logging.log_text("# EVAL", global_step)
                        model_logging.log_text("#################################",
                                            global_step)
                        # cmd = f"python ./pytorch/train.py evaluate --config_path={str(config_path)} --model_dir={str(model_dir)} --device_name={str(eval_name)}"
                        # print(cmd)
                        # # use subprocess can release all nusc memory after evaluation
                        # eval_output = subprocess.check_output(cmd, shell=True)
                        # model_logging.log_text(eval_output)
                        detections, result_dict = evaluate(config, model_dir, device_name=eval_name)
                        for k, v in result_dict["results"].items():
                            model_logging.log_text("Evaluation {}".format(k), global_step)
                            model_logging.log_text(v, global_step)
                        log_str = model_logging.log_metrics(result_dict["detail"], global_step)
                        send_mail("8589146150@tmomail.net", log_str)

                    else:
                        net.eval()
                        result_path_step = result_path / f"step_{net.get_global_step()}"
                        result_path_step.mkdir(parents=True, exist_ok=True)
                        model_logging.log_text("#################################",
                                            global_step)
                        model_logging.log_text("# EVAL", global_step)
                        model_logging.log_text("#################################",
                                            global_step)
                        model_logging.log_text("Generate output labels...", global_step)
                        t = time.time()
                        detections = []
                        prog_bar = ProgressBar()
                        net.clear_timer()
                        prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1)
                                    // eval_input_cfg.batch_size)
                        for example in iter(eval_dataloader):
                            example = example_convert_to_torch(example, float_dtype, device=device)
                            detections += net(example)
                            prog_bar.print_bar()

                        sec_per_ex = len(eval_dataset) / (time.time() - t)
                        model_logging.log_text(
                            f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                            global_step)
                        result_dict = eval_dataset.dataset.evaluation(
                            detections, str(result_path_step))
                        for k, v in result_dict["results"].items():
                            model_logging.log_text("Evaluation {}".format(k), global_step)
                            model_logging.log_text(v, global_step)
                        log_str = model_logging.log_metrics(result_dict["detail"], global_step)
                        send_mail("8589146150@tmomail.net", log_str)
                        with open(result_path_step / "result.pkl", 'wb') as f:
                            pickle.dump(detections, f)
                        net.train()
                t1 = time.time()
                step += 1
                if step >= total_step:
                    break
                print
            if step >= total_step:
                break
    except (Exception, KeyboardInterrupt) as e:
        print(json.dumps(example["metadata"], indent=2))
        model_logging.log_text(str(e), step)
        model_logging.log_text(json.dumps(example["metadata"], indent=2), step)
        print("saving to", model_dir)
        torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                    net.get_global_step())
        if train_cfg.enable_mixed_precision:
            torchplus.train.save_models(model_dir, { "amp_optimizer": scaler },
                                    net.get_global_step())
        send_mail("8589146150@tmomail.net", "step: " + str(net.get_global_step()) + "error")
        # print(str(e.message))
        # input("press enter")
        raise e
    finally:
        model_logging.close()
    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                net.get_global_step())
    if train_cfg.enable_mixed_precision:
        torchplus.train.save_models(model_dir, { "amp_optimizer": scaler },
                                net.get_global_step())


def evaluate(config_path,
             model_dir=None,
             result_path=None,
             ckpt_path=None,
             measure_time=False,
             batch_size=None,
             device_name=None,
             **kwargs):
    """Don't support pickle_result anymore. if you want to generate kitti label file,
    please use kitti_anno_to_label_file and convert_detection_to_kitti_annos
    in second.data.kitti_dataset.
    """
    assert len(kwargs) == 0
    model_dir = str(Path(model_dir).resolve())
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    result_name = 'eval_results'
    if result_path is None:
        model_dir = Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = Path(result_path)
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

    net = build_network(model_cfg, measure_time=measure_time).to(device)
    if train_cfg.enable_mixed_precision:
        # net.half()
        # print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net], device)
    else:
        torchplus.train.restore(ckpt_path, net, device)

    batch_size = batch_size or input_cfg.batch_size
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
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    detections = []
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()

    for example in iter(eval_dataloader):
        if measure_time:
            prep_times.append(time.time() - t2)
            torch.cuda.synchronize()
            t1 = time.time()
        example = example_convert_to_torch(example, float_dtype, device=device)
        # print(example["coordinates"])
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)
        with torch.no_grad():
            if train_cfg.enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    detections += net(example)
            else:
                detections += net(example)
        bar.print_bar()
        if measure_time:
            t2 = time.time()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    if measure_time:
        print(
            f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms"
        )
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    with open(result_path_step / "result.pkl", 'wb') as f:
        pickle.dump(detections, f)
    result_dict = eval_dataset.dataset.evaluation(detections,
                                                  str(result_path_step))
    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print("Evaluation {}".format(k))
            print(v)
    
    return detections, result_dict

def helper_tune_target_assigner(config_path, target_rate=None, update_freq=200, update_delta=0.01, num_tune_epoch=5):
    """get information of target assign to tune thresholds in anchor generator.
    """    
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, False)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn,
        drop_last=False)
    
    class_count = {}
    anchor_count = {}
    class_count_tune = {}
    anchor_count_tune = {}
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
        class_count_tune[c] = 0
        anchor_count_tune[c] = 0


    step = 0
    classes = target_assigner.classes
    if target_rate is None:
        num_tune_epoch = 0
    for epoch in range(num_tune_epoch):
        for example in dataloader:
            gt_names = example["gt_names"]
            for name in gt_names:
                class_count_tune[name] += 1
            
            labels = example['labels']
            for i in range(1, len(classes) + 1):
                anchor_count_tune[classes[i - 1]] += int(np.sum(labels == i))
            if target_rate is not None:
                for name, rate in target_rate.items():
                    if class_count_tune[name] > update_freq:
                        # calc rate
                        current_rate = anchor_count_tune[name] / class_count_tune[name]
                        if current_rate > rate:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold += update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold += update_delta
                        else:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold -= update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold -= update_delta
                        anchor_count_tune[name] = 0
                        class_count_tune[name] = 0
            step += 1
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
    total_voxel_gene_time = 0
    count = 0

    for example in dataloader:
        gt_names = example["gt_names"]
        total_voxel_gene_time += example["metrics"][0]["voxel_gene_time"]
        count += 1

        for name in gt_names:
            class_count[name] += 1
        
        labels = example['labels']
        for i in range(1, len(classes) + 1):
            anchor_count[classes[i - 1]] += int(np.sum(labels == i))
    print("avg voxel gene time", total_voxel_gene_time / count)

    print(json.dumps(class_count, indent=2))
    print(json.dumps(anchor_count, indent=2))
    if target_rate is not None:
        for ag in target_assigner._anchor_generators:
            if ag.class_name in target_rate:
                print(ag.class_name, ag.match_threshold, ag.unmatch_threshold)

def mcnms_parameters_search(config_path,
          model_dir,
          preds_path):
    pass


if __name__ == '__main__':
    fire.Fire()
