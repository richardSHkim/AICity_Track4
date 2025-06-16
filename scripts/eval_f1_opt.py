from typing import Optional
from pathlib import Path
import argparse
import json
import numpy as np

from ultralytics_custom.pycocotools_custom.coco import COCO
from ultralytics_custom.pycocotools_custom.cocoeval_modified import COCOeval
from ultralytics_custom.models.yolo.detect.val import CustomDetectionValidator


def find_optimal_threshold(gt_json: str, pred_json: str):
    validator = CustomDetectionValidator(args={"rect": True, "mode": "val"})
    validator.eval_from_coco_predictions(
        pred_json,
        gt_json,
    )

    conf_tick = validator.metrics.curves_results[1][0]
    f1_class_wise = validator.metrics.curves_results[1][1]
    conf_indices = np.argmax(f1_class_wise, axis=1)
    opt_threshold = {}
    for i, cat_id in enumerate(validator.names.keys()):
        opt_threshold[str(cat_id)] = float(conf_tick[conf_indices[i]])

    return opt_threshold


def main(
    gt_json: str,
    pred_json: str,
    conf_thresh_json: Optional[str] = None,
    conf_thresh: Optional[float] = None,
    save_json: Optional[str] = None,
):
    coco_gt = COCO(gt_json)
    pred_data = json.load(open(pred_json, "r"))

    # find or load optimal threshold
    if conf_thresh is not None:
        optimal_threshold_dict = {
            str(x["id"]): conf_thresh for x in coco_gt.dataset["categories"]
        }
    elif conf_thresh_json is not None:
        optimal_threshold_dict = json.load(open(conf_thresh_json, "r"))[
            Path(pred_json).stem
        ]
    else:
        optimal_threshold_dict = find_optimal_threshold(gt_json, pred_json)

    # filter prediction results by score
    pred_data = [
        x
        for x in pred_data
        if x["score"] >= optimal_threshold_dict[str(x["category_id"])]
    ]
    coco_dt = coco_gt.loadRes(pred_data)

    # Initialize evaluation object
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if save_json:
        results = {}
        with open(save_json, "w") as f:
            results[Path(pred_json).stem] = {
                **optimal_threshold_dict,
                "f1": float(coco_eval.stats[20]),
                "f1@0.5": float(coco_eval.stats[21]),
            }
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json", type=str, required=True)
    parser.add_argument("--pred_json", type=str, required=True)
    parser.add_argument("--conf_thresh_json", type=str, default=None)
    parser.add_argument("--conf_thresh", type=float, default=None)
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    main(
        args.gt_json,
        args.pred_json,
        args.conf_thresh_json,
        args.conf_thresh,
        args.save_json,
    )
