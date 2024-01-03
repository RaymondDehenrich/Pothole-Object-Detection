import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from modules.models import save_ckp

from config import OUTPUT_DIR
class SaveBestModel:
    def __init__(self, best_valid_map=float(0)):
        self.best_valid_map = best_valid_map
        
    def __call__(self, model, current_valid_map,optimizer, epoch,model_used ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch}\n")
            save_ckp(model,optimizer,epoch,model_used,Best=True)


def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_image_with_boxes(count,image, bounding_boxes,confidence):
    fig, ax = plt.subplots(1)
    
    image_data = np.transpose(image, (1, 2, 0))
    ax.imshow(image_data)  

    # Add bounding boxes to the image
    for box,conf in zip(bounding_boxes,confidence):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (xmin, ymin), 
            xmax-xmin, 
            ymax-ymin, 
            linewidth=1, 
            edgecolor='r', 
            facecolor='none', 
            label=0
        )
        ax.add_patch(rect)
        ax.annotate(f"Conf:{conf:.5f}",(xmin,ymin),color='red',weight='bold', fontsize=8)

    # Show the image with bounding boxes
    plt.savefig(os.path.join(OUTPUT_DIR,f'{count}.png'))
    plt.close()