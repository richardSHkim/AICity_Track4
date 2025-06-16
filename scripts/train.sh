#!/bin/bash


python -m torch.distributed.run --nproc_per_node=4 scripts/train.py \
    --project-name fisheye_od \
    --run-name yolo11n-test \
    --yaml-file configs/fisheye8k.yaml \
    --init-weight weights/yolo11n.pt \
    --device 0,1,2,3