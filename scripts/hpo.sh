#!/bin/bash


python scripts/hpo.py \
    --init_model_path weights/yolo11m.pt \
    --data_yaml configs/fisheye8k.yaml