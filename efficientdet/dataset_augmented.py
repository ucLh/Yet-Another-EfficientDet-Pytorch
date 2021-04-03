import os
import torch
import numpy as np

import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2, ToTensor
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), set='train2017',
                 transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.get_image_ids()
        self.augmentation = get_transforms(phase, mean, std)
        self.mean = mean
        self.std = std

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

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)

        params = {'image': img,
                  'bboxes': annot}
        # sample = {'img': img, 'annot': annot}
        # if self.transform:
        #     sample = self.transform(sample)

        sample = self.augmentation(**params)
        image, annot = sample['image'], sample['bboxes']

        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pt1, pt2 = annot[0][:2], annot[0][2:4]
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        cv2.rectangle(image_copy, pt1, pt2, (255, 255, 255), 1)
        cv2.imwrite('temp.jpg', image_copy)

        image = self.get_mean(image)


        # image = image.permute(2, 0, 1)

        return image, annot

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)

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
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2] - 1
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs, annots = [], []
    for entry in data:
        imgs.append(entry[0])
        annots.append(torch.from_numpy(np.asarray(entry[1])))
    # imgs = [s['image'] for s in data]
    # annots = [s['bboxes'] for s in data]
    # scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).float()
    # imgs = torch.stack(imgs)

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'image': imgs, 'bboxes': annot_padded}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'image': ((image.astype(np.float32) - self.mean) / self.std), 'bboxes': annots}


def get_transforms(phase, mean, std):
    list_transforms = list()
    if phase == "train":
        list_transforms.extend(
            [
                # albu.CoarseDropout(4, 32, 32),
                # albu.RandomSizedBBoxSafeCrop(512, 512, erosion_rate=0.0, interpolation=1, p=1.0),
                # albu.Resize(256, 512, interpolation=1, p=1),
                # albu.CoarseDropout(4, 16, 16),
                albu.HorizontalFlip(p=0.5),
                albu.RandomBrightness(p=1.0),
                # albu.Resize(512, 512, interpolation=1, p=1),

                # albu.OneOf([albu.RandomContrast(),
                #             # albu.RandomGamma(),
                #             albu.RandomBrightness()], p=1.0),
                # albu.CLAHE(p=1.0)
            ]
        )
    if phase == "val":
        list_transforms.extend(
            [
                albu.Resize(512, 512, interpolation=1, p=1),
            ]
        )
    # list_transforms.extend(
    #     [
    #         albu.Normalize(mean=mean, std=std, p=1),
    #         ToTensorV2(),
    #     ]
    # )
    list_trfms = albu.Compose(list_transforms,
                              bbox_params=albu.BboxParams(format='pascal_voc'))
    return list_trfms

# class AlbumentationsDataset(Dataset)