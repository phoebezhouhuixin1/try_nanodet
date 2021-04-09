# from tools.train import main as nanodet_train
from nanodet.util import cfg, load_config # cfg is a basic CfgNode object that we will edit later
from voc2coco import voc2coco # if we are not running nanodet_train.py as a standalone file, need a dot
from types import SimpleNamespace
import yaml
import os
import datetime


import os
import torch
import logging
import argparse
import numpy as np
import torch.distributed as dist

from nanodet.util import mkdir, Logger, cfg, load_config, DataParallel
from nanodet.trainer.trainer import Trainer
from nanodet.trainer.dist_trainer import DistTrainer 
from nanodet.data.collate import collate_function
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.evaluator import build_evaluator




def train(args):
    base_dir = args.project_dir
    train_path = os.path.join(base_dir, "train")
    valid_path = os.path.join(base_dir, "validate")
    labels_path = os.path.join(base_dir, ".videoio", "classes.classes")
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(base_dir, 'models', f'{args.model_name}_training_output', timestamp_str)
    os.makedirs(output_path, exist_ok=True)
    weights_path = os.path.join(os.getcwd(), r"model\nanodet_m.pth") # TODO: assuming running from videoio directory; change to scripts\nanodet later
    cfg_path = os.path.join(os.getcwd(), r"config\nanodet-m.yml") # TODO: assuming running from videoio directory; change to scripts\nanodet later
    # # Slice the images if applicable
    # auto_slice = hasattr(args, 'auto_slice') and args.auto_slice
    # if auto_slice:
    #     slice_images(base_dir, "train", class_names, True)
    #     slice_images(base_dir, "validate", class_names, True)
    #     train_path = os.path.join(base_dir, ".videoio", "train")
    #     valid_path = os.path.join(base_dir, ".videoio", "validate")

    train_annots_path = os.path.join(train_path, "train_annots.json")
    valid_annots_path = os.path.join(valid_path, "valid_annots.json")
    voc2coco(train_path, labels_path, train_annots_path)
    voc2coco(valid_path, labels_path, valid_annots_path)

    # Get the names and number of classes, so that we know the shape of the model Head for cfg
    class_names = []
    with open(labels_path, "r") as labelfile:
        for line in labelfile:
            class_names.append(line.split("\n")[0])
    print("number of classes = ", len(class_names), class_names)

    load_config(cfg, cfg_path)
    cfg.defrost() 
    cfg.save_dir = output_path
    cfg.model.arch.head.num_classes = len(class_names)
    cfg.data.train.img_path = train_path
    cfg.data.train.ann_path = train_annots_path
    cfg.data.val.img_path = valid_path
    cfg.data.val.ann_path = valid_annots_path
    if hasattr(args, "use_gpu") and args.use_gpu:
        import GPUtil
        gpuInfo = GPUtil.getGPUs()
        if len(gpuInfo) > 0:
            cfg.device.gpu_ids = [gpuInfo[0].id] # TODO: if there is more than one gpu, what will be gpuInfo[0].id? This cfg needs to be a list e.g. [0]
            cfg.device.batchsize_per_gpu = 2 # TODO can it go more than this since nanodet is a smaller model than yolo?
            batch_size = 2 # TODO: assuming only one gpu?
            # TODO: not sure how to set cfg.device.workers_per_gpu
            cfg.device.workers_per_gpu = 1
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
    else:
        cfg.device.gpu_ids = [] # TODO: THERE IS NO SUPPORT FOR CPU TRAINING IN NANODET. 
        # NEED TO OVERWRITE the build_trainer() function in nanodet/trainer/__init__.py AND tools/train.py
        # Will do that below :(
        batch_size = args.batch_size
    
    cfg.schedule.total_epochs = args.epochs
    stage1_epochs = min(25, args.epochs//2)
    cfg.schedule.lr_schedule.milestones = [stage1_epochs] # epoch number, not iteration number like detectron2
    # TODO: Stage 1 and 2 learning rate configurations necssary like YOLO?
    args.config = os.path.join(base_dir, 'models',f"{args.model_name}_train_config.yaml")
    stream = open(args.config, 'w')
    yaml.dump(cfg, stream)
    print("DUMPED THE YAML\n")
    args.seed = 0
    args.local_rank = -1 # TODO: not sure how to set this for distributed training. What is a rank? What is distributed training
    # nanodet_train(args)

    # mkdir(args.local_rank, cfg.save_dir)
    logger = Logger(args.local_rank, cfg.save_dir)
    if args.seed is not None:
        logger.log('Set random seed to {}'.format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if len(cfg.device.gpu_ids) != 0:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            if args.seed == 0: # i dont know why this is necessary but it is defined in tools/train.py
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    logger.log('Creating model...')
    model = build_model(cfg.model)

    logger.log('Setting up data...')
    train_dataset = build_dataset(cfg.data.train, 'train')
    val_dataset = build_dataset(cfg.data.val, 'test')

    if len(cfg.device.gpu_ids) > 1:
        print('rank = ', args.local_rank)
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank % num_gpus)
        dist.init_process_group(backend='nccl')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                       num_workers=cfg.device.workers_per_gpu, pin_memory=True,
                                                       collate_fn=collate_function, sampler=train_sampler,
                                                       drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                 pin_memory=True, collate_fn=collate_function, drop_last=True)
        print("CUDA DATALOADER")
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size,
                                                       shuffle=True, num_workers=0, collate_fn=collate_function, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_function, drop_last=True)
        print("CPU DATALOADER")

    if len(cfg.device.gpu_ids) > 1: # distributed trainer for multi gpu training
        print("GPU IDS (cuda)", cfg.device.gpu_ids)
        trainer = DistTrainer(args.local_rank, cfg, model, logger)
        trainer.set_device(cfg.device.batchsize_per_gpu, rank, device=torch.device('cuda'))  
    else:
        print("GPU IDS (cpu)", cfg.device.gpu_ids)
        trainer = Trainer(args.local_rank, cfg, model, logger)
    if hasattr(args, "use_gpu") and args.use_gpu:
        trainer.set_device(cfg.device.batchsize_per_gpu, rank, device=torch.device('cuda')) 
    else
        # trainer.set_device(batch_size, cfg.device.gpu_ids, device=torch.device('cpu'))
        trainer.model = DataParallel(model, cfg.device.gpu_ids).to("cpu")
        # TODO: CPU TRAINING IS NOT SUPPORTED. 
        # RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
        # See Trainer class for details.

    if 'load_model' in cfg.schedule:
        trainer.load_model(cfg)
    if 'resume' in cfg.schedule:
        trainer.resume(cfg)

    # print(trainer.__dict__)
    evaluator = build_evaluator(cfg, val_dataset)

    logger.log('Starting training...')
    print("torch.cuda.is_available()? ", torch.cuda.is_available())
    trainer.run(train_dataloader, val_dataloader, evaluator)


if __name__ == "__main__":
    args = SimpleNamespace()
    args.project_dir = r"C:\Users\Phoebe\Desktop\kangaroo_only"
    args.base_model = "base:nanodet"
    args.model_name = "nanodetmodel"
    args.use_gpu = True # specify args.batch_size if False.
    # args.batch_size = 2
    args.autoslice = False
    args.epochs = 4
    train(args)
    

