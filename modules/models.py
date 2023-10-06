import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import (
    NUM_CLASSES,
    CKP_PATH
)

#fasterrcnn_resnet50_fpn_v2 Model
#Heavyweight but more accurate model
class rcnn_model(nn.Module):
    def __init__(self):
        super(rcnn_model, self).__init__()
        self.base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    def forward(self, image,target=None):
        if target is not None:
            return self.base_model(image, target)
        else:
            return self.base_model(image)

#SSDLITE Model
#Lightweight but less accurate model
class ssdlite_model(nn.Module):
    def __init__(self):
        super(ssdlite_model, self).__init__()
        self.base_model = ssdlite320_mobilenet_v3_large(num_classes=NUM_CLASSES)
   
    def forward(self, image,target=None):
        if target is not None:
            return self.base_model(image, target)
        else:
            return self.base_model(image)

def save_ckp(model,optimizer,epoch,checkpoint_dir='./checkpoint/',latest=False,Best=False):
    os.makedirs(os.path.join(CKP_PATH,checkpoint_dir), exist_ok = True)
    state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }
    if Best == False:
        if latest == False:
            f_path = os.path.join(CKP_PATH,checkpoint_dir,f'checkpoint_epoch_{epoch}.pth')
        else: f_path = os.path.join(CKP_PATH,checkpoint_dir,f'latest_checkpoint.pth')
    else:
        f_path = os.path.join(CKP_PATH,checkpoint_dir,'best_model.pth')
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model, optimizer,Best=False):
    if Best==True:
        checkpoint = torch.load(os.path.join(CKP_PATH,checkpoint_fpath,'best_model.pth'))
    else:
        checkpoint = torch.load(os.path.join(CKP_PATH,checkpoint_fpath,'latest_checkpoint.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']