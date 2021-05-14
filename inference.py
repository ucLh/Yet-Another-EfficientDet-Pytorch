import os
from pathlib import Path
import json

import argparse
import cv2
import torch
import yaml
from tqdm import tqdm

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default='weights/efficientdet-d0.pth', help='/path/to/weights')
ap.add_argument('--save_json', type=str, default='bbox_results.json',
                help='path to save the resulting json annotations')
ap.add_argument('-i', '--input_dir', type=str, default='datasets/detection_dataset/test',
                help='dir with images for inference')
ap.add_argument('--nms_threshold', type=float, default=0.5,
                help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--conf_threshold', type=float, default=0.1,
                help='confidence threshold')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--visualise', action="store_true", help='If specified, saves prediction images')
ap.add_argument('-s', '--save_dir', type=str, default='preds',
                help='dir where inference results will be saved if --visualise is set')
ap.add_argument('--use_coco_classes', action="store_true", help='If specified, uses coco classes')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
conf_threshold = args.conf_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights
save_dir = args.save_dir
json_save_path = args.save_json
input_dir = args.input_dir
visualise = args.visualise

print(f'running infernce on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
if args.use_coco_classes:
    obj_list = params['obj_list_coco']
else:
    obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def make_csv_entry(image_name, box, score):
    return f'{image_name},{box[0]},{box[1]},{box[2]},{box[3]},{score}\n'


def inference(input_path, save_dir, image_names, obj_list, model, threshold=0.05):
    box_id_static = 1
    json_dict = {
        "info": {
            "version": 1.0,
            "url": "no url specified",
            "year": 2021,
            "date_created": "today",
            "contributor": "no contributor specified",
            "description": ""
        },
        "categories": [
            {
                "name": "car",
                "id": 3,
                "supercategory": "unknown"
            }
        ],
        "licenses": [],
        "images": [],
        "annotations": []
    }

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    json_save_dir = os.path.dirname(json_save_path)
    Path(json_save_dir).mkdir(parents=True, exist_ok=True)
    csv_save_path = os.path.basename(json_save_path).split('.')[0] + '.csv'

    with open(csv_save_path, 'w') as csv_file:
        csv_file.write('image,xmin,ymin,w,h,score\n')
        for i, image_name in tqdm(enumerate(image_names)):
            image_base_entry = {
                "extra_info": {},
                "subdirs": ".",
                "id": i,
                "width": 0,
                "height": 0,
                "file_name": image_name,
            }

            image_path = os.path.join(input_path, image_name)

            ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef],
                                                             mean=params['mean'], std=params['std'])
            x = torch.from_numpy(framed_imgs[0])
            img = ori_imgs[0]
            image_base_entry["width"] = img.shape[1]
            image_base_entry["height"] = img.shape[0]
            json_dict["images"].append(image_base_entry)

            if use_cuda:
                x = x.cuda(gpu)
            x = x.float()

            x = x.unsqueeze(0).permute(0, 3, 1, 2)
            features, regression, classification, anchors = model(x)

            preds = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, nms_threshold)

            if not preds:
                continue

            preds = invert_affine(framed_metas, preds)[0]

            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']

            if rois.shape[0] > 0:
                bbox_score = scores

                for roi_id in range(rois.shape[0]):
                    score = float(bbox_score[roi_id])
                    if score < conf_threshold:
                        continue
                    label = int(class_ids[roi_id])
                    x0, y0, x1, y1 = list(map(int, rois[roi_id, :]))
                    box = [x0, y0, x1 - x0, y1 - y0]
                    area = int(box[2] * box[3])
                    if visualise:
                        img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
                        cv2.putText(img, obj_list[label], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                    bbox_base_entry = {
                        "image_id": i,
                        "extra_info": {
                            "human_annotated": True
                        },
                        "category_id": label + 1,
                        "iscrowd": 0,
                        "id": box_id_static,
                        "score": float(score),
                        "bbox": box,
                        "area": area
                    }
                    box_id_static += 1

                    json_dict["annotations"].append(bbox_base_entry)
                    csv_entry = make_csv_entry(image_name, box, score)
                    csv_file.write(csv_entry)
            if visualise:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                save_path = os.path.join(save_dir, image_name)
                cv2.imwrite(save_path, img)

            if i % 1000 == 0:
                json.dump(json_dict, open(json_save_path, 'w'), indent=4)

        json.dump(json_dict, open(json_save_path, 'w'), indent=4)


if __name__ == '__main__':
    image_names = os.listdir(input_dir)
    image_names = sorted(image_names)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(gpu)

    inference(input_dir, save_dir, image_names, obj_list, model)
