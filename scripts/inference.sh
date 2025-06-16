#!/bin/bash


python scripts/inference.py \
    --image_folder datasets/FishEyeChallenge/FishEye8K/val/images \
    --model_path checkpoints/yolo11n.pt \
    --save_pred_json results/predictions/yolo11n.json
