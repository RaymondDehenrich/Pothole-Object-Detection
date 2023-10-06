from torchvision.datasets import CocoDetection
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
import torch
from config import (
    RESIZE_TARGET,
)

class CustomDataset(CocoDetection):
    def __init__(self, root,custom_annotations,transforms=None):
        self.transforms = transforms
        self.root = root
        self.custom_annotations = parse_annotation(custom_annotations)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.custom_annotations)
    
    def __getitem__(self, index):
        #Get image from dataset and change to RGB form
        img = cv2.imread(os.path.join(self.root, self.custom_annotations[index]['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        #Get bounding box from annotations
        custom_annotations = self.custom_annotations[index].copy()

        #Note: 1 = width, 0 = height
        aspect_ratio = img.shape[1] / img.shape[0]

        #Resizing while maintaining aspect ratio
        # Calculate aspect ratio of the original image
        resize_scale=1
        # Determine which dimension to fix (width or height)
        if aspect_ratio >= 1:  # Image is wider
            resize_scale = img.shape[1]/RESIZE_TARGET
        elif aspect_ratio < 1:  # Image is taller or square
            resize_scale = img.shape[0]/RESIZE_TARGET
        img = resize_and_pad_image(img,target_size=RESIZE_TARGET)
        img /=255.0

        #adjusting bounding box according to resize scale
        custom_annotations['annotations'] = adjust_bounding_boxes(custom_annotations['annotations'],resize_scale)

        #Format the annotations to supported format
        coco_target = annotation_format_to_coco(index,custom_annotations)
        if self.transforms:
            sample = self.transforms(image = img,
                                     bboxes = coco_target['boxes'],
                                     labels = coco_target['labels'])
            image_resized = sample['image']
            coco_target['boxes'] = torch.Tensor(sample['bboxes'])
        return image_resized, coco_target
    


def annotation_format_to_coco(index,annotations):
    labels =[] 
    boxes=[]
    for bbox in annotations['annotations']:
        x1, y1, x2, y2 = bbox['bbox']
        x_2 = float(x2)
        y_2 = float(y2)
        x = float(x1)
        y = float(y1)
        if x_2 >= RESIZE_TARGET:
            x_2 = RESIZE_TARGET-1
        if y_2 >= RESIZE_TARGET:
            y_2 = RESIZE_TARGET-1
        if x_2 <= x or y_2 <= y:
            # Handle the invalid bounding box by skipping it.
            continue
        else:
            boxes.append([x, y, x_2, y_2])
            labels.append(1)

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
        else torch.as_tensor(boxes, dtype=torch.float32)
    iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    # Convert custom annotations to COCO format
    coco_target = {
        "boxes": boxes,
        'labels':labels,
        'area':area,
        'iscrowd':iscrowd,
        "image_id": torch.as_tensor(index,dtype=torch.int64),
    }
    if np.isnan((coco_target['boxes']).numpy()).any() or coco_target['boxes'].shape == torch.Size([0]):
        coco_target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
    return coco_target



def parse_annotation(annotation_folder):
    final_annotation =[]
    for annotation_file in os.listdir(annotation_folder):
        tree = ET.parse(os.path.join(annotation_folder,annotation_file))
        root = tree.getroot()
        parse_annotate_file = []
        for obj in root.findall('object'):
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)
            
            bounding_boxes=[
                xmin,
                ymin,
                xmax,
                ymax
            ]
            parse_annotate_file.append({'bbox':bounding_boxes,'category_id': 1})
        final_annotation.append({'file_name':annotation_file.replace('.xml', '.png'),'annotations':parse_annotate_file})
    return final_annotation


def resize_and_pad_image(image, target_size=RESIZE_TARGET):
    # Calculate the aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]

    # Determine which dimension (width or height) to fix and resize
    if aspect_ratio > 1:  
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    # Resize the image while preserving aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    # Pad the image to the desired dimensions
    pad_x = max(0, target_size - new_width)
    pad_y = max(0, target_size - new_height)
    padded_image = cv2.copyMakeBorder(resized_image, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)

    return padded_image


def adjust_bounding_boxes(bounding_boxes, resize_scale):
    resize_scale = float(resize_scale)
    temp_box = []
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box['bbox']
        xmin *= 1/resize_scale
        ymin *= 1/resize_scale
        xmax *= 1/resize_scale
        ymax *= 1/resize_scale
        adjusted_boxes=[
            xmin,
            ymin,
            xmax,
            ymax,
        ]
        temp_box.append({'bbox':adjusted_boxes})
    return temp_box