import os
import cv2
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')

from config import (DEVICE,INPUT_DIR,OUTPUT_DIR,RESIZE_TARGET,BOX_THRESHOLD,INFERENCE_BEST_EPOCH)
from modules.dataset import resize_and_pad_image
from modules.models import rcnn_model,ssdlite_model,mobilenet,load_ckp
from modules.utils import visualize_image_with_boxes





if __name__ == "__main__":
    os.makedirs(INPUT_DIR,exist_ok=True)
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    print("Select model to use:")
    print("1. FasterRCNN (ResNet50)")
    print("2. FasterRCNN (Mobilenet)")
    print("3. SSDLite")
    answer = input(">>")
    if int(answer) == 1:
        model = rcnn_model()
        model_used = 'rcnn_model'
    elif int(answer)==2:
        model = mobilenet()
        model_used = 'mobilenet'
    else:
        model = ssdlite_model()
        model_used ='ssdlite'
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    try:
        model, optimizer, start_epoch = load_ckp(model_used,model, optimizer,Best=INFERENCE_BEST_EPOCH)
        print(f"[Notice] {model_used} checkpoint loaded [{start_epoch} epoch trained]")
    except:
        print(f"[Notice] There is no {model_used} checkpoint, please train a new one!")
        exit()
    model.to(DEVICE)
    model.eval()
    if len(os.listdir(INPUT_DIR)) < 1:
         print(f'Please add some image to {INPUT_DIR}!')
         exit()
    print("[Notice]Starting Inference!")
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
        filtered_boxes = []
        filtered_scores = []
        if len(outputs[0]['boxes']) != 0:
                for box, score in zip(outputs[0]['boxes'], outputs[0]['scores']):
                    if (score >= 0.1) or model_used=='fasterrcnn':
                        filtered_boxes.append(box)
                        filtered_scores.append(score)
                if len(filtered_boxes) != 0:
                    outputs[0]['boxes'] = torch.stack(filtered_boxes)
                    outputs[0]['scores'] = torch.tensor(filtered_scores)
                    Kept_indices = torchvision.ops.nms(outputs[0]['boxes'],outputs[0]['scores'],iou_threshold=BOX_THRESHOLD)
                    outputs[0]['boxes'] = outputs[0]['boxes'][Kept_indices]
                    outputs[0]['scores'] = outputs[0]['scores'][Kept_indices]
        visualize_image_with_boxes(image,img,outputs[0]['boxes'],outputs[0]['scores']) 
    print("[Notice]Inference complete!")