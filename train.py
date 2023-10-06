import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm

from modules.dataset import CustomDataset
from modules.utils import get_train_transform,collate_fn,SaveBestModel,visualize_image_with_boxes
from modules.models import rcnn_model,ssdlite_model,load_ckp,save_ckp
from config import (
   DEVICE,
   TRAIN_DIR,
   VALIDATE_DIR,
   BATCH_SIZE,
   NUM_WORKERS,
   BOX_THRESHOLD,
   NUM_EPOCHS,
   VISUALIZE
)
seed = 40
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
   
def train(trainloader,model):
    model.train()

    progress_bar = tqdm(trainloader,total=len(trainloader))
    count=0
    for i,data in enumerate(progress_bar):
        optimizer.zero_grad()
        images, labels = data
        count+=1
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in labels]
        outputs= model(images,targets)
        losses = sum(loss for loss in outputs.values())

        losses.backward()
        optimizer.step()
        loss_value = losses.item()
        progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value

def validate(valloader,model,model_used,visualize=False):
    model.eval()
    progress_bar = tqdm(valloader,total=len(valloader))
    target_val = []
    preds = []
    count=0
    for i, data in enumerate(progress_bar, 0):
        images,targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
                outputs = model(images, targets)
        outputs = [{k: v.detach().cpu() for k, v in t.items()} for t in outputs]
        targets = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
        images = list(image.detach().cpu() for image in images)
        img_count=0
        for output,target in zip(outputs,targets):
            if len(output['boxes']) != 0:
                filtered_boxes = []
                filtered_scores = []
                filtered_labels = []
                target_box=[]
                target_label=[]
                for box, score,label,tar_box,tar_label in zip(output['boxes'], output['scores'],output['labels'],target['boxes'],target['labels']):
                    if (score >= 0.1) or model_used=='fasterrcnn':
                        filtered_boxes.append(box)
                        filtered_scores.append(score)
                        filtered_labels.append(label)
                        target_box.append(tar_box)
                        target_label.append(tar_label)
                if len(filtered_boxes) != 0:
                    output['boxes'] = torch.stack(filtered_boxes)
                    output['scores'] = torch.tensor(filtered_scores)
                    output['labels'] = torch.tensor(filtered_labels)
                    target['boxes'] = torch.stack(target_box)
                    target['labels']=torch.tensor(target_label)
                    Kept_indices = torchvision.ops.nms(output['boxes'],output['scores'],iou_threshold=BOX_THRESHOLD)
                    output['boxes'] = output['boxes'][Kept_indices]
                    output['scores'] = output['scores'][Kept_indices]
                    output['labels'] = output['labels'][Kept_indices]
                    target['boxes'] = target['boxes'][Kept_indices]
                    target['labels']=target['labels'][Kept_indices]
                    if visualize:
                        visualize_image_with_boxes(count,images[img_count],output['boxes']) 
                        img_count+=1
                        count+=1
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target_val.append(true_dict)
    metric = MeanAveragePrecision()
    metric.update(preds,target_val)
    return metric.compute()




if __name__ == "__main__":
    train_dataset=CustomDataset(os.path.join(TRAIN_DIR,'images'),os.path.join(TRAIN_DIR,'annotations'),transforms=get_train_transform())
    trainloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,collate_fn=collate_fn,drop_last=False)
    val_dataset=CustomDataset(os.path.join(VALIDATE_DIR,'images'),os.path.join(VALIDATE_DIR,'annotations'),transforms=get_train_transform())
    valloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,collate_fn=collate_fn,drop_last=False)
    print(f"Training Samples  : {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    print("Select model to use:")
    print("1. FasterRCNN")
    print("2. SSDLite")
    answer = input(">>")
    model_used = ""
    if int(answer) == 1:
        model = rcnn_model()
        params = [p for p in model.parameters() if p.requires_grad]
        model_used = 'fasterrcnn'
        optimizer = torch.optim.Adam(params, lr=0.001)
    else:
        model = ssdlite_model()
        model_used = 'ssdlite'
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    
    model.to(DEVICE)
    scheduler = MultiStepLR(
            optimizer=optimizer, milestones=[100], gamma=0.1, verbose=True
        )
    start_epoch = 0
    try:
        model, optimizer, start_epoch = load_ckp(model_used, model, optimizer)
        print(f"[Notice] Latest {model_used} checkpoint loaded [{start_epoch} epoch trained]")
    except:
        print(f"[Notice] Creating new checkpoint for {model_used}")

    best_model = SaveBestModel()
    model_save_rate = 3
    for epoch in range(start_epoch,NUM_EPOCHS):
        print(f"EPOCH [{epoch+1}] of [{NUM_EPOCHS}]")
        train_loss = train(trainloader,model)

        
        if(epoch%3==0):
            #visualize=True will print the validate image in output folder
            metric_summary = validate(valloader, model,model_used,visualize=VISUALIZE)   
            print(f"Epoch #{epoch+1} mAP@0.50:0.95: {metric_summary['map']}")
            print(f"Epoch #{epoch+1} mAP@0.50: {metric_summary['map_50']}")
            best_model(model, float(metric_summary['map']),optimizer, epoch+1,model_used)
        if(epoch%model_save_rate==0):
            if epoch % 10 ==0:
                save_ckp(model,optimizer,epoch,checkpoint_dir=model_used,latest=False)
            save_ckp(model,optimizer,epoch,checkpoint_dir=model_used,latest=True)
        scheduler.step()
    print('Finished Training')