import os
import torch
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, ToTensor
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, phase, transforms, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), set='train2017',
                 transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.get_image_ids()
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

    def __getitem__(self, index):

        image = self.load_image(index)
        boxes = self.load_annotations(index)

        # there is only one class
        labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    # target['boxes'] = sample['bboxes']
                    break

        return image, target['boxes']

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0

        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
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
            A.RandomSizedBBoxSafeCrop(512, 512, erosion_rate=0.0, interpolation=1, p=0.5),
            # A.OneOf([
            #     A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
            #                          val_shift_limit=0.2, p=0.9),
            #     A.RandomBrightnessContrast(brightness_limit=0.2,
            #                                contrast_limit=0.2, p=0.9),
            # ], p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=4, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            A.Normalize(p=1),
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
            A.Resize(height=512, width=512, p=1.0),
            A.Normalize(p=1),
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
