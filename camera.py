import cv2
import torch
import numpy as np
import torchvision
from modules.dataset import resize_and_pad_image
from config import (DEVICE,RESIZE_TARGET,INFERENCE_BEST_EPOCH)
from modules.models import rcnn_model,ssdlite_model,mobilenet,load_ckp



if __name__ == "__main__":
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
        print(f"[Notice] Best performing {model_used} checkpoint loaded [{start_epoch} epoch trained]")
    except:
        print(f"[Notice] There is no {model_used} checkpoint, please train a new one!")
        exit()
    model.to(DEVICE)
    model.eval()
    cap = cv2.VideoCapture(1)
    print("Begining Camera Capture")
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = resize_and_pad_image(frame,target_size=RESIZE_TARGET)
        frame /=255.0
        image_input = np.transpose(frame, (2, 0, 1))
        image_input = torch.tensor(image_input, dtype=torch.float).to(DEVICE)
        image_input = torch.unsqueeze(image_input, 0)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        with torch.no_grad():
                outputs = model(image_input)
        
        outputs = [{k: v.detach().cpu() for k, v in t.items()} for t in outputs]
        filtered_boxes = []
        filtered_scores = []
        frame = cv2.resize(frame,(700,700),interpolation=cv2.INTER_CUBIC)
        scale = 700/RESIZE_TARGET
        if len(outputs[0]['boxes']) !=0:
            for box, score in zip(outputs[0]['boxes'],outputs[0]['scores']):
                if (score >= 0.25)or model_used=='fasterrcnn':
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
                if(len(filtered_boxes)!=0):
                    outputs[0]['boxes'] = torch.stack(filtered_boxes)
                    outputs[0]['scores'] = torch.tensor(filtered_scores)  
                    Kept_indices = torchvision.ops.nms(outputs[0]['boxes'],outputs[0]['scores'],iou_threshold=0.5)
                    outputs[0]['boxes'] = outputs[0]['boxes'][Kept_indices]
                    outputs[0]['scores'] = outputs[0]['scores'][Kept_indices]
                    for detection,conf in zip(outputs[0]['boxes'],outputs[0]['scores']):
                        x, y, x2, y2 = detection
                        x, y, width, height = int(x*scale), int(y*scale), int(x2*scale), int(y2*scale)

                
                        cv2.rectangle(frame, (x, y), (width,height), (0, 255, 0), 1)  
                        cv2.putText(frame, f"Conf:{conf:.5f}", (x, y),cv2.FONT_HERSHEY_SIMPLEX ,0.4, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Pothole Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

