import cv2
import numpy as np
import os
import time
import torch
import argparse
import shutil
import json
from types import SimpleNamespace
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
# from nanodet.evaluator.coco_detection import CocoDetectionEvaluator
from pycocotools.coco import COCO # for per-class AP; we will not use CocoDetectionEvaluator that nanodet provides
from pycocotools.cocoeval import COCOeval
from voc2coco import voc2coco # if we are not running nanodet_predict.py as a standalone file, need a dot

def main(args):
    SCORE_THRESH_TEST = 0.3
    NMS_THRESH_TEST = 0.3
    base_dir = args.project_dir
    args.model_name = args.model_name.split(".pth")[0] # strip .pth if any
    config_fp = os.path.join(base_dir, "models", f"{args.model_name.split(':')[1]}_train_config.yaml")
    if args.model_name.startswith("custom:"):
        weights_fp = os.path.join(base_dir, "models", f"{args.model_name.split(':')[1]}.pth")
    else:
        weights_fp = os.path.join(os.getcwd(), r"nanodet\model\nanodet_m.pth") # assuming running from videoio directory 
    logger = Logger(-1, use_tensorboard=False)
    load_config(cfg, config_fp)
    cfg.defrost()
    cfg.model.arch.head.nms_thresh_test = NMS_THRESH_TEST # i added this to cfg; NMS threshold was originally unconfigurable and set to default 0.6
    class_names = cfg.class_names
    print("class_names", class_names)

    output_path = os.path.join(args.output_dir, "output", f"{args.model_name.split(':')[1]}")
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    eval_output_path = os.path.join(output_path, "nanodet_eval.json")
    # cfg.save_dir = output_path
   
    if hasattr(args, "use_gpu") and args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    predictor = Predictor(cfg, weights_fp, logger, device)

    got_truth = False
    for fp in os.listdir(args.input_dir):
        if fp.endswith(".xml"):
            got_truth = True
            break
  
    image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
    video_ext = ['mp4', 'mov', 'avi', 'mkv']
    img_list = []
    video_list = []
    for fp in os.listdir(args.input_dir):
        ext = os.path.splitext(fp)[1]
        if ext in image_ext:
            img_list.append(fp)
        elif ext in video_ext:
            video_list.append(fp)
    results = {}
    for img_fp in img_list:
        meta, res = predictor.inference(os.path.join(args.input_dir, img_fp))
        img_id = os.path.splitext(img_fp)[0]
        results[img_id] = res
        result_image = predictor.visualize(res, meta, cfg.class_names, SCORE_THRESH_TEST)
        save_file_name = os.path.join(output_path, f"{img_id}.png")
        cv2.imwrite(save_file_name, result_image)
    
    if got_truth:
        labels_ids = list(range(1, len(class_names)+1))
        label2id = dict(zip(class_names, labels_ids))
        test_annots_path = os.path.join(args.input_dir, "test_annots.json")
        voc2coco(args.input_dir, label2id, test_annots_path)
        # cfg.defrost()
        cfg.data.val.img_path = args.input_dir
        cfg.data.val.ann_path = test_annots_path
        test_dataset = build_dataset(cfg.data.val, 'test')
        # print("RESULTS IS", results)
        evaluator = build_evaluator(cfg, test_dataset, SCORE_THRESH_TEST)
        results_json = evaluator.results2json(results) # list of dict for every instance (can be many instances per image)
        # print("RESULTS_JSON IS", results_json)
        json_path = os.path.join(output_path, "coco_instances_results.json")
        json.dump(results_json, open(json_path, 'w'))
        evaluate_coco(test_annots_path, results_json, class_names, args.input_dir, eval_output_path)


    # TODO: video inference
    
        
class Predictor:
    def __init__(self, cfg, weights_fp, logger, device):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model) # build the model architecture according to the config used during training
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        ckpt = torch.load(weights_fp, map_location=lambda storage, loc: storage) # load checkpoint weights onto CPU
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval() # set the model in evaluation mode
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio) # image preprocessing in the validation stage: normalization and resizing.
    
    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=True)
        print('viz time: {:.3f}s'.format(time.time()-time1))
        return result_img
            
def evaluate_coco(gt_json, result_json, class_names, input_dir, eval_output_path):
    '''
    Compare ground truth with predictions, using pycocotools.cocoeval.COCOEval API.
    Prints the mAP, and AP per class. Writes these metrics to a json file. 
   
    Args:
        gt_json (str): Path to ground truth annotations json (in the official COCO annotation format; can use voc2coco() to generate this)
        result_json (str): Path to predictions json (in the official COCO output format https://cocodataset.org/#format-results; can use make_all_preds_json() to generate this)
        class_names (list(str)): List of class names 
        input_dir (str): Path of folder in which the evaluation images and xmls are located
        eval_output_path (str): Path to the json file where evaluation metrics will be saved
    Returns:
        COCOeval.stats (list)
    '''
    # Begin evaluation
    try:
        cocoGt=COCO(gt_json) # Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        cocoDt=cocoGt.loadRes(result_json)
    except: # no detections with confidence score>0.05; might be the case during initial few epochs of training
        print("NO DETECTIONS")
        with open(eval_output_path, "w") as f:
            pass
        results = {}
        for class_name in class_names:
            results[f"AP50-{class_name}"] = 0.0
             # write prediction metrics to model config file
            try:
                config_file = open(eval_output_path, "r+")
                config = json.load(config_file, object_hook=lambda d: SimpleNamespace(**d))
                config_file.seek(0)
            except:
                config_file = open(eval_output_path, 'w')
                config = SimpleNamespace()

            metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
            metrics = {k: 0.0 for k in metrics}
            metrics.update(results)
            config.test = {"input_dir":input_dir, "metrics":metrics}
            json.dump(config, config_file, default=lambda o: o.__dict__, sort_keys=True, indent=2)
            config_file.truncate()
        return [0.0, 0.0], results
    
    imgIds=sorted(cocoDt.getImgIds()) # will look at the image entries in the gt json file, and then map each of the image_id to its corresponding annotation entry/entries
    print("Image IDs to use in evaluation calculation: ", imgIds)
    
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("cocoEval.eval now contains: ", type(cocoEval.eval))
    for k in cocoEval.eval.keys():
        print(k, type(cocoEval.eval[k]))
    print()
    print("Shape of precision, recall and scores: ", cocoEval.eval["precision"].shape, cocoEval.eval["recall"].shape, cocoEval.eval["scores"].shape)
    print()
    print("cocoEval.stats now contains: ", cocoEval.stats)
    precisions = cocoEval.eval["precision"]

    # precision has dims (10 different iou thresholds, recall thresholds??, num_classes, area range, max detections??)
    assert len(class_names) == precisions.shape[2]
    
    
    # Print AP per class
    results_per_category = []
    for idx, name in enumerate(class_names):
        # for area range, choose index 0: all area ranges
        # for max detections, choose index -1: typically 100 per image
        # precision = precisions[:, :, idx, 0, -1]
        precision = precisions[0, :, idx, 0, -1] # only AP50 (not AP) per class!!
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category.append(("{}".format(name), float(ap)))
    results = {"AP50-" + name: ap for name, ap in results_per_category}
    print("\n", results)


    # Print the results
    mAP = cocoEval.stats[1] # AP50
    print('{"progress":{"percent":100, "metrics":{"mAP":'+ str(mAP) +'}}}\n')

    # write prediction metrics to model config file
    try:
        config_file = open(eval_output_path, "r+")
        config = json.load(config_file, object_hook=lambda d: SimpleNamespace(**d))
        config_file.seek(0)
    except:
        config_file = open(eval_output_path, 'w')
        config = SimpleNamespace()

    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    metrics = {k: cocoEval.stats[i] for i,k in enumerate(metrics)}
    metrics.update(results)
    config.test = {"input_dir":input_dir, "metrics":metrics}
    json.dump(config, config_file, default=lambda o: o.__dict__, sort_keys=True, indent=2)
    config_file.truncate()

    return cocoEval.stats, results
    
if __name__ == "__main__":
    args = SimpleNamespace()
    args.project_dir = r"C:\Users\Phoebe\Desktop\kangaroo_only"
    args.model_name = "custom:nanodetmodel.pth"
    args.input_dir = rf"{args.project_dir}\validate"
    args.output_dir = rf"{args.project_dir}\validate"
    args.use_gpu = True
    args.auto_slice = False
    main(args)
