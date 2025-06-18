#!/bin/bash


python scripts/inference.py \
    --image_folder datasets/FishEyeChallenge/FishEye8K/test/images \
    --model_path checkpoints/yolo11m.pt \
    --save_pred_json results/predictions/yolo11m.json
