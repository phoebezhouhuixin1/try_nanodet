import cv2
import os
import time
import torch
import argparse
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']

if __name__ == "__main__":
    args = SimpleNamespace()
    args.project_dir = r"C:\Users\Phoebe\Desktop\kangaroo_only"
    args.base_model = "custom:nanodet.pth"
    args.model_name = "nanodetmodel"
    args.use_gpu = True

'''
let config = {
                command: 'predict',
                project_type: project.type, // required=True, type=str, metavar='Project Type'
                project_dir: project.dir, // required=True, metavar='Project Directory', type=str
                model_name: project.predictModel, // type=str, metavar="Model Name", default="my_model.h5", help="Name of saved model with .h5 if custom model, default='model.h5'"
                input_dir: project.predictInputDir, // metavar='Input Directory'
                output_dir: project.predictOutputDir, // metavar='Output Directory'
                threshold: null, // type=float, metavar="Report Threshold", help="Set custom threshold, the max prob < threshold, reject class"
                use_gpu: project.predictUseGpu,
                auto_slice: project.predictAutoSlice,
                skip_frame: project.skip_frame 
'''