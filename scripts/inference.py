from typing import Optional
import os
import time
import argparse
import cv2
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np

from ultralytics import YOLO


def get_image_Id(img_name):
    img_name = img_name.split(".png")[0]
    sceneList = ["M", "A", "E", "N"]
    cameraIndx = int(img_name.split("_")[0].split("camera")[1])
    sceneIndx = sceneList.index(img_name.split("_")[1])
    frameIndx = int(img_name.split("_")[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId


def main(
    image_folder: str,
    model_path: str,
    score_threshold_json: str,
    max_fps: float = 25.0,
    save_pred_json: Optional[str] = None,
):
    # load model
    model = YOLO(model_path)
    if score_threshold_json is not None:
        score_threshold = json.load(open(score_threshold_json, "r"))
    else:
        score_threshold = None

    # gather image files
    image_files = sorted(
        [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    print(f"Found {len(image_files)} images.")

    # warmup
    img = cv2.imread(os.path.join(image_folder, image_files[0]))
    for _ in tqdm(range(100), desc="warm up ..."):
        _ = model.predict([img], verbose=False, device="cuda")

    # prediction
    print("Prediction started")
    results = []
    total_time = 0
    for image_path in tqdm(image_files, desc=f"running on {Path(model_path).stem}"):
        img = cv2.imread(os.path.join(image_folder, image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        t0 = time.time()
        output = model.predict([img], verbose=False, device="cuda", conf=0.0)

        boxes = output[0].boxes.xyxy.cpu().numpy()
        scores = output[0].boxes.conf.cpu().numpy()
        labels = output[0].boxes.cls.cpu().numpy()
        for lab, box, s in zip(labels, boxes, scores):
            if score_threshold is not None and float(s) < score_threshold[str(lab)]:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            results.append(
                {
                    "image_id": get_image_Id(image_path),
                    "category_id": int(lab),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                }
            )
        t3 = time.time()
        total_time += t3 - t0

    # fps
    fps = len(image_files) / total_time
    normfps = min(fps, max_fps) / max_fps

    print(f"\n--- Evaluation Complete ---")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {normfps:.4f}")

    # Save predictions to JSON
    if save_pred_json is not None:
        os.makedirs(os.path.dirname(save_pred_json), exist_ok=True)
        with open(save_pred_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved prediction results on {save_pred_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder", type=str, required=True, help="Path to image folder"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--score_threshold_json", type=str, default=None, help="Json file for class-wise score thresholding"
    )
    parser.add_argument(
        "--max_fps", type=float, default=25.0, help="Maximum FPS for evaluation"
    )
    parser.add_argument(
        "--save_pred_json",
        type=str,
        default=None,
        help="Output JSON file for predictions",
    )
    args = parser.parse_args()

    main(args.image_folder, args.model_path, args.score_threshold_json, args.max_fps, args.save_pred_json)
