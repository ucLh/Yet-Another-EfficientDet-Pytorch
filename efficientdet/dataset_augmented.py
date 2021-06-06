import os
import torch
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
from PIL import Image
import random
from copy import deepcopy


class CocoDataset(Dataset):
    def __init__(self, root_dir, phase, transforms, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), set='train2017',
                 transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.get_image_ids_alt()
        # self.augmentation = get_transforms(phase, mean, std)
        self.mean = mean
        self.std = std
        self.transforms = transforms
        self.test = (phase == 'val')

        self.load_classes()

    def get_mean(self, image):
        image = image.astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image

    def get_image_ids_alt(self):
        annot_imgs = self.coco.imgs
        result_ids = []
        for i, img in (enumerate(annot_imgs.values())):
            id = img['id']
            result_ids.append(id)
        return result_ids

    def get_image_ids(self):
        result_ids = set()
        annot_imgs = self.coco.imgs
        # imgs_path = os.path.join(self.root_dir, self.set_name, 'thermal_8_bit')
        imgs_path = os.path.join(self.root_dir, self.set_name)
        real_names = os.listdir(imgs_path)
        suffix = '.' + real_names[0].split('.')[-1]
        names_dict = {}

        for i, img in (enumerate(annot_imgs.values())):
            id_ = img['id']
            name = img['file_name']
            name = os.path.basename(name)
            name = name.split('.')[0]
            name += suffix
            names_dict[id_] = name

        anns = self.coco.anns
        for ant in anns.values():
            id = ant['image_id']
            # name = os.path.join(images_dir_path, images[id]['file_name'])
            if not (names_dict[id] in real_names):
                continue
            result_ids.add(id)
        return list(result_ids)

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def check_bboxes(self, image: np.ndarray, boxes):
        h, w = image.shape[:2]
        # [x1, y1, x2, y2]
        for box in boxes:
            box[2] = min(w, box[2])
            box[3] = min(h, box[3])
            box[0] = max(box[0], 0)
            box[1] = max(box[1], 0)

    def crop_aug(self, img, boxes, visualise=False):
        index = random.randint(0, len(boxes) - 1)
        window = deepcopy(boxes[index])
        window[:2] -= 404
        window[2:4] += 404
        self.check_bboxes(img, [window])
        window = list(map(int, window))
        aug = A.Compose([
            A.Crop(x_min=window[0], x_max=window[2], y_min=window[1], y_max=window[3], p=1.0),
            A.Resize(808, 808, p=1.0),
            A.RandomCrop(640, 640, p=1.0)
        ], p=1.0, bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
            ))
        sample = aug(image=img, bboxes=boxes)
        res_img, res_boxes = sample['image'], sample['bboxes']

        if visualise:
            for box in res_boxes:
                box = list(map(int, box))
                cv2.rectangle(res_img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 1)
            cv2.imwrite('temp.jpg', res_img)
        return res_img, np.array(res_boxes)

    def __getitem__(self, index):

        image = self.load_image(index)
        boxes = self.load_annotations(index)
        if boxes.shape[0] == 0:
            return None, None
        if not self.test:
            image, boxes = self.crop_aug(image, boxes)
        # self.check_bboxes(image, boxes)

        # there is only one class
        labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        out_image, out_boxes = None, None

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    out_image = sample['image']
                    out_boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    break

        return out_image, out_boxes

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        #img = cv2.imread(path)
        img = Image.open(path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = np.array(img, dtype=np.float32)
        img /= 255.0

        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        img_ids = self.image_ids[image_index]
        annotations_ids = self.coco.getAnnIds(imgIds=img_ids)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2] - 0.1
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def get_train_transforms():
    return A.Compose(
        [
            # A.RandomSizedCrop(min_max_height=(380, 676), height=1024, width=1024, p=0.5),
            # A.RandomSizedBBoxSafeCrop(640, 640, erosion_rate=0.0, interpolation=1, p=1),
            # A.OneOf([
            #     A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
            #                          val_shift_limit=0.2, p=0.9),
            #     A.RandomBrightnessContrast(brightness_limit=0.2,
            #                                contrast_limit=0.2, p=0.9),
            # ], p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.Resize(height=512, width=512, p=1),
            # A.Cutout(num_holes=4, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            #A.Normalize(p=1),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=640, width=640, p=1.0),
            #A.Normalize(p=1),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )
