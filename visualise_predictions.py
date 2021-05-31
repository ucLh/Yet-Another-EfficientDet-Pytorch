"""
This script draws predictions bounding boxes using json predicitions from coco_eval.
The color of predicted boxes is white, the color of ground truth boxes is red.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm


def get_id2box_dict(annot):
    id2box = dict()
    for box in annot:
        image_id = box['image_id']
        if image_id in id2box.keys():
            id2box[image_id].append(box)
        else:
            id2box[image_id] = []
            id2box[image_id].append(box)
    return id2box, id2box.keys()


def get_id2path_dict(imgs_json):
    id2path = dict()
    for img in imgs_json:
        id = img['id']
        path = img['file_name']
        id2path[id] = path
    return id2path


def visualise(ids, id2box, id2path, dataset_dir, out_dir, color, legend_coords, legend_name):
    for x in tqdm(ids):
        # Read image
        path = os.path.join(dataset_dir, id2path[x])
        img = Image.open(path)
        img = np.array(img, dtype=np.uint8)
        img = draw_line_legend(img, color, legend_coords, legend_name)

        # Prepare dir for saving
        save_path = os.path.join(out_dir, id2path[x])
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if x not in id2box.keys():
            # Write image unchanged if there are no boxes
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img)
        else:
            boxes = id2box[x]
            for box in boxes:
                # # In case of pretrained model. 3 is a category_id for 'car' class
                # if box['category_id'] != 3:
                #     continue
                # Draw a box on the image
                x0, y0, w, h = list(map(int, box['bbox']))
                x1 = x0 + w
                y1 = y0 + h
                img = cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img)


def draw_line_legend(img, color, coords, name):
    img = cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), color, thickness=1)
    return cv2.putText(img, name, (coords[2] + 5, coords[3]), cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.5, color=color, thickness=1)


def main(args):
    with open(args.gt_annotations) as f:
        gt = json.load(f)
    with open(args.effdet_preds) as f:
        pred = json.load(f)
    with open(args.yolo_preds) as f:
        pred_yolo = json.load(f)

    id2box_pred, _ = get_id2box_dict(pred)
    id2box_pred_yolo, _ = get_id2box_dict(pred_yolo)
    id2box_gt, gt_ids = get_id2box_dict(gt['annotations'])
    id2path = get_id2path_dict(gt['images'])
    colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0)]
    coords = [(20, 20, 40, 20), (20, 40, 40, 40), (20, 60, 40, 60)]
    # First draw predictions of effdet using white color
    visualise(gt_ids, id2box_pred, id2path, args.dataset_dir, args.output_dir, colors[0], coords[0], 'Effdet')
    # Then draw Yolo predictions
    visualise(gt_ids, id2box_pred_yolo, id2path, args.output_dir, args.output_dir, colors[1], coords[1], 'Yolo')
    # Then draw gt on top of predictions using red color
    visualise(gt_ids, id2box_gt, id2path, args.output_dir, args.output_dir, colors[2], coords[2], 'GT')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--effdet_preds', type=str,
                        help='Path to json predictions from coco_eval.py by EfficientDet model',
                        default='preds/test_bbox_results.json')
    parser.add_argument('--yolo_preds', type=str,
                        help='Path to json predictions from coco_eval.py by Yolo model',
                        default='preds/test_bbox_results_yolo_001_nms03.json')
    parser.add_argument('--dataset_dir', type=str,
                        help='Path to a directory with dataset images',
                        default='datasets/trash/test')
    parser.add_argument('--gt_annotations', type=str,
                        help='Path to json ground truth annotations.'
                             'They are needed to get the list of images of the dataset',
                        default='datasets/trash/annotations/instances_test.json')
    parser.add_argument('--output_dir', type=str,
                        help='Path to a directory to save the results',
                        default='preds/trash_effdet_yolo001_nms03')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
