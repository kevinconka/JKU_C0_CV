#!/bin/bash

# data directory
DATA_DIR=~/GitHub/JKU_C0_CV/tmp/yolov5/dataset.yaml
# yolov5 github directory
YOLOV5_DIR=~/GitHub/yolov5

# activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jku-ai 

# Train YOLOv5 on custom dataset
python3 $YOLOV5_DIR/train.py \
    --img 640 \
    --batch 16 \
    --epochs 50 \
    --data $DATA_DIR \
    --weights yolov5s.pt \
    --name first-try