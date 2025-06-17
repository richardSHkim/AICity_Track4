#!/bin/bash


python scripts/eval.py \
    --model_path checkpoints/yolo11m.pt \
    --data_yaml configs/fisheye8k.yaml \
    --save_json checkpoints/yolo11m.json