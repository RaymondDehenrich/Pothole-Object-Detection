import os
import cv2
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')

from config import (DEVICE,INPUT_DIR,OUTPUT_DIR,RESIZE_TARGET,BOX_THRESHOLD)
from modules.dataset import resize_and_pad_image
from modules.models import rcnn_model,ssdlite_model,load_ckp
from modules.utils import visualize_image_with_boxes





if __name__ == "__main__":
    os.makedirs(INPUT_DIR,exist_ok=True)
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    print("Select model to use:")
    print("1. FasterRCNN")
    print("2. SSDLite")
    answer = input(">>")
    if int(answer) == 1:
        model = rcnn_model()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.001)
        model_used = 'fasterrcnn'
    else:
        model = ssdlite_model()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
        model_used ='ssdlite'

    try:
        model, optimizer, start_epoch = load_ckp(model_used,model, optimizer,Best=False)
        print(f"[Notice] Best performing {model_used} checkpoint loaded [{start_epoch} epoch trained]")
    except:
        print(f"[Notice] There is no {model_used} checkpoint, please train a new one!")
        exit()
    model.to(DEVICE)
    model.eval()
    if len(os.listdir(INPUT_DIR)) < 1:
         print(f'Please add some image to {INPUT_DIR}!')
         exit()
    for image in os.listdir(INPUT_DIR):
        img = cv2.imread(os.path.join(INPUT_DIR,image))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
        img = resize_and_pad_image(img,target_size=RESIZE_TARGET)
        img /=255.0
        img = np.transpose(img, (2, 0, 1))
        image_input = torch.tensor(img, dtype=torch.float).to(DEVICE)
        image_input = torch.unsqueeze(image_input, 0)
        with torch.no_grad():
                outputs = model(image_input)
        image = image.replace('.png', '')
        outputs = [{k: v.detach().cpu() for k, v in t.items()} for t in outputs]
        if len(outputs[0]['boxes']) !=0:
            Kept_indices = torchvision.ops.nms(outputs[0]['boxes'],outputs[0]['scores'],iou_threshold=BOX_THRESHOLD)
            outputs[0]['boxes'] = outputs[0]['boxes'][Kept_indices]
            outputs[0]['scores'] = outputs[0]['scores'][Kept_indices]
        visualize_image_with_boxes(image,img,outputs[0]['boxes'])
    print("[Notice]Predicting complete!")