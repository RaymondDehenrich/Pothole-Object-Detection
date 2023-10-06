# Pothole-Object-Detection
Pothole object detection using SSDLite in PyTorch
Dataset: https://www.kaggle.com/datasets/andrewmvd/pothole-detection

# How to setup
1. Create a new environment (optional)
2. Download a stable torch build (such as 2.0.1) from Pytorch website.
3. Run 'pip install -r requirements.txt'
4. You should be good to go.

# How to train
0. Look at how to setup first.
1. Download/Get dataset and put in a folder structure of './Dataset/train/annotations' and './Dataset/train/images' for train dataset, and './Dataset/validate/annotations' and './Dataset/validate/images' for validation.<br>
Note: Annotations files need to have the same filename as the image with only a different file type, e.g. "potholes001.png" and "potholes001.xml". annotation only needs one or multiple (xmin,ymin,xmax,ymax). aside from this, it will be ignored.
2. Run train.py with environment(optional).

# How to predict
0. Look at how to setup
1. Input image at './input'
2. Run main.py with environment(optional).
