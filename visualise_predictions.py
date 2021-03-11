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


def visualise(ids, id2box, id2path, dataset_dir, out_dir, color=(255, 255, 255)):
    for x in tqdm(ids):
        # Read image
        path = os.path.join(dataset_dir, id2path[x])
        img = cv2.imread(path)

        # Prepare dir for saving
        save_path = os.path.join(out_dir, id2path[x])
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if x not in id2box.keys():
            # Write image unchanged if there are no boxes
            cv2.imwrite(save_path, img)
        else:
            boxes = id2box[x]
            for box in boxes:
                # In case of pretrained model. 3 is a category_id for 'car' class
                if box['category_id'] != 3:
                    continue
                # Draw a box on the image
                x0, y0, w, h = list(map(int, box['bbox']))
                x1 = x0 + w
                y1 = y0 + h
                img = cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

            cv2.imwrite(save_path, img)


def main(args):
    with open(args.gt_annotations) as f:
        gt = json.load(f)
    with open(args.model_preds) as f:
        pred = json.load(f)

    id2box_pred, _ = get_id2box_dict(pred)
    id2box_gt, gt_ids = get_id2box_dict(gt['annotations'])
    id2path = get_id2path_dict(gt['images'])
    # First draw predictions using white color
    visualise(gt_ids, id2box_pred, id2path, args.dataset_dir, args.output_dir, color=(255, 255, 255))
    # Then draw gt on top of predictions using red color
    visualise(gt_ids, id2box_gt, id2path, args.output_dir, args.output_dir, color=(0, 0, 255))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_preds', type=str,
                        help='Path to json predictions from coco_eval.py',
                        default='test_bbox_results.json')
    parser.add_argument('--dataset_dir', type=str,
                        help='Path to a directory with dataset images',
                        default='datasets/detection_dataset/test')
    parser.add_argument('--gt_annotations', type=str,
                        help='Path to json ground truth annotations.'
                             'They are needed to get the list of images of the dataset',
                        default='datasets/detection_dataset/annotations/instances_test.json')
    parser.add_argument('--output_dir', type=str,
                        help='Path to a directory to save the results',
                        default='preds/test')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
