import csv
import os
import random

from os import path as osp


def remove_bboxes(source_csv='train_bbox.csv', out_csv='train_bbox_noisy_test.csv'):
    # Remove some objects and save their coordinates
    ids = random.sample(range(1, 550), 150)
    saved_bboxes = []
    with open(source_csv) as f, open(out_csv, 'w') as out:
        inp = csv.reader(f)
        header = next(inp, None)
        header = ','.join(header) + '\n'
        out.write(header)
        for i, row in enumerate(inp):
            if i in ids:
                saved_bboxes.append(row[1:])
                continue
            else:
                res_str = ','.join(row) + '\n'
                out.write(res_str)
    return saved_bboxes


def add_bboxes(saved_bboxes, souce_csv='train_bbox.csv', out_csv='train_bbox_noisy_test.csv'):
    # Add bboxes to the images with no objects
    image_dir = 'train'
    all_image_names = os.listdir(image_dir)
    all_image_names = sorted(all_image_names)
    bbox_names = set()
    empty_names = set()
    ids = random.sample(range(1, 550), 150)
    with open(souce_csv) as f, open(out_csv, 'a') as out:
        inp = csv.reader(f)
        next(inp, None)
        for i, row in enumerate(inp):
            bbox_names.add(row[0])
        for name in all_image_names:
            if name not in bbox_names:
                empty_names.add(name)
        empty_names = list(empty_names)
        j = 0
        for i, name in enumerate(empty_names):
            if i in ids:
                res_str = name + ',' + ','.join(saved_bboxes[j]) + '\n'
                j += 1
                out.write(res_str)


saved_boxes = remove_bboxes()
add_bboxes(saved_boxes)
