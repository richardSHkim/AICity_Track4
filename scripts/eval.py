from typing import Optional
import argparse
import json
import yaml
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

from ultralytics_custom.pycocotools_custom.coco import COCO
from ultralytics_custom.pycocotools_custom.cocoeval_modified import COCOeval


def main(model_path: str, data_yaml: str, save_json: Optional[str] = None):
    # load ground-truth from json
    with open(data_yaml) as stream:
        gt_json = yaml.safe_load(stream)["val_json"]
    coco_gt = COCO(gt_json)

    # update image id to filename
    for img_dict in coco_gt.loadImgs(coco_gt.getImgIds()):
        stem = Path(img_dict["file_name"]).stem
        image_id = int(stem) if stem.isnumeric() else stem
        for ann_dict in coco_gt.loadAnns(coco_gt.getAnnIds([img_dict["id"]])):
            ann_dict["image_id"] = image_id
        img_dict["id"] = image_id
    coco_gt.createIndex()

    # load model
    model = YOLO(model_path)

    # load validator
    custom = {"rect": True}
    kwargs = {"data": data_yaml, "save_json": True}
    args = {**model.overrides, **custom, **kwargs, "mode": "val"}
    validator = DetectionValidator(args=args, _callbacks=model.callbacks)

    # run validation
    validator(model=model.model)
    metrics = validator.metrics

    # get optimal confidence threshold
    conf_ticks = metrics.box.curves_results[1][0]
    f1 = metrics.box.curves_results[1][1]
    conf_indices = np.argmax(f1, axis=1)
    opt_conf_thresh_dict = {}
    for i in range(len(conf_indices)):
        opt_conf_thresh_dict[coco_gt.dataset["categories"][int(i)]["id"]] = float(
            conf_ticks[conf_indices[i]]
        )

    # filter predictions
    predictions_coco = []
    for pred in validator.jdict:
        # fix category id
        pred["category_id"] = coco_gt.dataset["categories"][pred["category_id"] - 1][
            "id"
        ]

        # thresholding with score
        if pred["score"] < opt_conf_thresh_dict[pred["category_id"]]:
            continue

        predictions_coco.append(pred)

    # get f1 score
    coco_dt = coco_gt.loadRes(predictions_coco)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    f1 = float(coco_eval.stats[20])
    f1_50 = float(coco_eval.stats[21])
    print(f"F1: {f1}")
    print(f"F1@0.5: {f1_50}")

    # save
    if save_json is not None:
        with open(save_json, "w") as f:
            data = {
                **opt_conf_thresh_dict,
                "f1": f1,
                "f1@0.5": f1_50,
            }
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_yaml", type=str, required=True)
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    main(
        args.model_path,
        args.data_yaml,
        args.save_json,
    )
