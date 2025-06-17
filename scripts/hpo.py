import argparse
from ultralytics import YOLO


def main(init_model_path, data_yaml):
    # Initialize the YOLO model
    model = YOLO(init_model_path)

    # Tune hyperparameters
    model.tune(
        use_ray=True,
        data=data_yaml,
        gpu_per_trial=4,
        iterations=50,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--data_yaml", type=str, required=True)
    args = parser.parse_args()

    main(args.init_model_path, args.data_yaml)
