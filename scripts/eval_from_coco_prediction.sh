#!/bin/bash


python scripts/eval_from_coco_prediction.py \
    --gt_json datasets/FishEyeChallenge/FishEye8K/val/val.json \
    --pred_json results/predictions.json \
    --save_json results/f1/score.json
