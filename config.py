import torch

BATCH_SIZE = 2
RESIZE_TARGET = 300
NUM_EPOCHS = 100
NUM_WORKERS = 6

BOX_THRESHOLD = 0.25

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = './Dataset/train'

VALIDATE_DIR = './Dataset/validate'

CLASSES = [
    '__background__', 'potholes'
]

NUM_CLASSES = len(CLASSES)

INPUT_DIR ='./input'

OUTPUT_DIR = './output'


CKP_PATH = "./checkpoint"

#Change this to True if you want to print validation image in output
VISUALIZE = False