import argparse

from ultralytics import YOLO, SETTINGS
from ultralytics_custom.models.yolo.detect import CustomDetectionTrainer


def main(args):
    if args.report_to == "wandb":
        SETTINGS["wandb"] = True
    else:
        SETTINGS["wandb"] = False

    model = YOLO(args.init_weight)
    model.train(
        trainer=CustomDetectionTrainer,
        project=args.project_name,
        name=args.run_name,
        data=args.yaml_file,
        # epochs=args.epochs,
        optimizer=args.optimizer,
        lr0=args.lr,
        imgsz=640,
        device=args.device,
        seed=args.seed,
        save_json=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # logging
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="wandb",
    )
    # parser.add_argument(
    #     "--save-dir",
    #     type=str,
    #     required=True,
    # )
    # parser.add_argument(
    #     "--val-period",
    #     type=int,
    #     default=1,
    # )

    # data
    parser.add_argument(
        "--yaml-file",
        type=str,
        required=True,
    )

    # hyper-parameters
    parser.add_argument(
        "--init-weight",
        type=str,
        required=True,
    )
    # parser.add_argument(
    #     "--epochs",
    #     type=int,
    #     default=200,
    # )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "adamw"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001111,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    # parser.add_argument(
    #     "--lrf",
    #     type=float,
    #     default=1.0,
    # )
    # parser.add_argument(
    #     "--weight-decay",
    #     type=float,
    #     default=0.01,
    # )
    # parser.add_argument(
    #     "--warmup",
    #     type=int,
    #     default=0,
    # )
    # parser.add_argument(
    #     "--use_cos_lr",
    #     action="store_true",
    # )

    # others
    parser.add_argument(
        "--device",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2035,
    )
    args = parser.parse_args()

    import os
    import requests
    import traceback

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    try:
        main(args)

        if os.environ.get("SLACK_ALARM_URL", None) and local_rank in [0, -1]:
            requests.post(
                os.environ["SLACK_ALARM_URL"], json={"text": "Train finished."}
            )

    except Exception as e:
        print(e)
        print(traceback.format_exc())

        if os.environ.get("SLACK_ALARM_URL", None) and local_rank in [0, -1]:
            requests.post(
                os.environ["SLACK_ALARM_URL"],
                json={"text": f"Training has failed: {e}"},
            )
