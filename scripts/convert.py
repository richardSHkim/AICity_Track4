import argparse
from ultralytics import YOLO


def main(model_path: str, precision: str, data_yaml: str = None):
    model = YOLO(model_path)

    # set default kwargs
    kwargs = {
        "format": "engine",
        "batch": 1,
        "imgsz": 640,
        "dynamic": False,
        # "nms": True,  # TODO: check nms option (default: False)
    }
    # precision
    if precision == "fp16":
        kwargs["half"] = True
    elif precision == "int8":
        assert data_yaml is not None, f"data_yaml for calibration is missing"
        kwargs["int8"] = True
        kwargs["data"] = data_yaml

    # Export the model to TensorRT format
    model.export(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--precision", type=str, default="fp16", choices=["fp16", "int8"]
    )
    parser.add_argument(
        "--data_yaml", type=str, default=None, help="data yaml file for calibration."
    )
    args = parser.parse_args()

    main(args.model_path, args.precision, args.data_yaml)
