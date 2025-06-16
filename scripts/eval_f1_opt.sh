#!/bin/bash


python scripts/eval_f1_opt.py \
    --gt_json datasets/FishEyeChallenge/FishEye8K/val/val.json \
    --pred_json results/predictions/yolo11n.json \
    --save_json results/f1/yolo11n.json
