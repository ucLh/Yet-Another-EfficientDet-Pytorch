import argparse
import json
import os
import pandas as pd
from pathlib import Path
import sys


def main(args):
    # Create coco dictionary template
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

    # Get image names and paths
    image_names = os.listdir(args.image_dir)
    image_names = sorted(image_names)

    # Get a annotations from csv
    df = pd.read_csv(args.csv_annotations)

    # Start generating coco annotations for each image
    bbox_id = 0
    for i, name in enumerate(image_names):
        # Add an image entry to dict
        image_base_entry = {
            "extra_info": {},
            "subdirs": ".",
            "id": i,
            "width": 676,
            "height": 380,
            "file_name": name,
        }
        json_dict["images"].append(image_base_entry)

        # Get boxes for a corresponding image
        bboxes_list = df.loc[df['image'] == name].values.tolist()

        if not bboxes_list:
            continue

        for bbox in bboxes_list:
            bbox_id += 1
            bbox = bbox[1:]  # Skip image name, we already know it

            # Convert bbox format
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            new_bbox = [x_min, y_min, width, height]

            # Add a bbox entry to dict
            bbox_base_entry = {
              "image_id": i,
              "extra_info": {
                "human_annotated": True
              },
              "category_id": 3,
              "iscrowd": 0,
              "id": bbox_id,
              "bbox": new_bbox,
              "area": width * height
            }
            json_dict["annotations"].append(bbox_base_entry)

    # Save the coco annotations dict into json
    save_dir = os.path.dirname(args.output_file)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(json_dict, f, indent=4)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str,
                        help='Path to directory with images',
                        default='datasets/detection_dataset/train')
    parser.add_argument('--csv_annotations', type=str,
                        help='Path to a csv file with annotations',
                        default='datasets/detection_dataset/train_bbox.csv')
    parser.add_argument('--output_file', type=str,
                        help='Path to save the resulting json annotations',
                        default='datasets/detection_dataset/annotations/instances_train.json')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
