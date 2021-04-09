import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
import glob
from PIL import Image

def get_label2id(labels_path: str) -> Dict[str, int]:
    # ids must start from 1
    with open(labels_path, 'r') as f:
        labels_str = f.read().split("\n")
        # print("labels_str: ", labels_str)
    labels_ids = list(range(1, len(labels_str)+1))
    # print(labels_ids)
    return dict(zip(labels_str, labels_ids))

def get_image_info(dirname, annotation_root):#, extract_num_from_imgid=True):
   
    filename = annotation_root.findtext('filename')
    im= Image.open(os.path.join(dirname,filename))
    width, height = im.size
    # print(width, height)
    
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
   

    # size = annotation_root.find('size')
    # width = int(size.findtext('width'))
    # height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info

def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))

    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height], # TODO:  ymin is the top left corner of the bbox? i thinkk
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str):#,extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    print("Directory is now: ", os.path.dirname(annotation_paths[0]))
    dirname = os.path.dirname(annotation_paths[0])
    for a_path in tqdm(annotation_paths):
       
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(dirname =dirname, annotation_root=ann_root)# TODO: width and height are not in the VOC xml files that are output by the videoio annotator
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)
        
def voc2coco(ann_dir, labels, output) -> None:
    """
    Convert all xml files in a folder into a single json annotation file in COCO format.

    Args:
        ann_dir (str): The path to a subdirectory (e.g. myProject/train) containing xml files
        labels (str OR dict): Path to a txt file containing class labels (one class label on each line) (e.g. myProject/labels.txt), 
            OR if dict, will be taken to be {label:id}
        output (str): Desired output path (e.g. myProject/train_annots.json)

    """
    if type(labels) == str:
        label2id = get_label2id(labels_path=labels)
    else:
        label2id = labels
    ann_paths = glob.glob(ann_dir+"/*.xml")
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=output
    )
