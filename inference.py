import os
from pathlib import Path

import argparse
import cv2
import torch
import yaml
from tqdm import tqdm

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='detection_dataset', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default='weights/efficientdet-d0.pth', help='/path/to/weights')
ap.add_argument('-s', '--save_dir', type=str, default='preds', help='dir where inference results will be saved')
ap.add_argument('-i', '--input_dir', type=str, default='datasets/detection_dataset/test', help='dir with images for inference')
ap.add_argument('--nms_threshold', type=float, default=0.5,
                help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--use_coco_classes', action="store_true", help='If specified, uses coco classes')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights
save_dir = args.save_dir
input_dir = args.input_dir

print(f'running infernce on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
if args.use_coco_classes:
    obj_list = params['obj_list_coco']
else:
    obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

def inference(input_path, save_dir, image_names, obj_list, model, threshold=0.05):
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_name in tqdm(image_names):
        image_path = os.path.join(input_path, image_name)

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef],
                                                         mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])
        img = ori_imgs[0]

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
                if score < 0.4:
                    continue
                label = int(class_ids[roi_id])
                x0, y0, x1, y1 = list(map(int, rois[roi_id, :]))
                img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
                cv2.putText(img, obj_list[label], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    MAX_IMAGES = 10000
    image_names = os.listdir(input_dir)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(gpu)

    inference(input_dir, save_dir, image_names, obj_list, model)
