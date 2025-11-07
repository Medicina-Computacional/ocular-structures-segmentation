import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm

# Paths
annotations_path = r''
output_masks_folder = r''
os.makedirs(output_masks_folder, exist_ok=True) # Create output folder if it doesn't exist

# Load COCO annotations
coco = COCO(annotations_path)
image_ids = coco.getImgIds()

# Get category IDs
cat_ids = coco.getCatIds(catNms=['Arcada vascular', 'Nervio optico'])
category_mapping = {coco.loadCats(cat_ids[i])[0]['name']: cat_ids[i] for i in range(len(cat_ids))}
arcada_vascular_cat_id = category_mapping.get('Arcada vascular', None)
nervio_optico_cat_id = category_mapping.get('Nervio optico', None)

# Process images in order
for img_id in tqdm(sorted(image_ids)):
    image_info = coco.loadImgs(img_id)[0]
    image_name = os.path.splitext(image_info['file_name'])[0]

    # Get all annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # Initialize empty mask
    combined_mask = None

    for ann in annotations:
        rle = coco.annToRLE(ann)
        mask = maskUtils.decode(rle)

        if combined_mask is None:
            combined_mask = np.zeros_like(mask, dtype=np.uint8)

        # Arcada Vascular -> label 1
        if ann['category_id'] == arcada_vascular_cat_id:
            combined_mask[mask == 1] = 1
        # Optic Nerve -> label 2
        if ann['category_id'] == nervio_optico_cat_id:
            combined_mask[mask == 1] = 2

    # Save the combined mask
    if combined_mask is not None:
        mask_filename = f"{image_name}.png"
        mask_path = os.path.join(output_masks_folder, mask_filename)
        cv2.imwrite(mask_path, combined_mask)
