import json
import os
import pandas as pd

from os import path as osp

# Define paths
IMAGE_DIR = 'train'
CSV_ANNOTATIONS = 'train_bbox.csv'
OUTPUT_FILE = 'train_annotations.json'

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
image_names = os.listdir(IMAGE_DIR)
image_names = sorted(image_names)
image_paths = list(map(lambda x: osp.join(IMAGE_DIR, x), image_names))

# Get a annotations from csv
df = pd.read_csv(CSV_ANNOTATIONS)

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
with open(OUTPUT_FILE, 'w') as f:
    json.dump(json_dict, f, indent=4)
