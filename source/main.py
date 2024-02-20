"""This file is to provide main functionalities for model and data analysis

author: danish ansari
copyright: na
"""

from visualize import Visualize
from inference import inference
from evaluation import evaluate
import argparse

parser = argparse.ArgumentParser("BDD-Data Yolov5 Detections")
parser.add_argument(
    "-t",
    "--task",
    type=str,
    default="inference",
    choices=["data-viz", "plot-viz", "pred-eval", "infer"],
)
parser.add_argument(
    "-d", "--data-path", type=str, default="/dataset", help="root path to dataset"
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="config/bdd_data.yaml",
    help="path to model config",
)
parser.add_argument(
    "-w",
    "--weights",
    type=str,
    default="weights/yolov5s_bdd.pt",
    help="path to trained weights",
)


def main(args):
    """Main driver function to models and data analysis

    Args: (command-line)
        `-t/--task`: one of `data-viz`, `plot-viz`, `pred-eval`, `infer`; defaults to `infer`.
        `-d/--data-path`: path to root directory of downloaded bdd-dataset.
        `-c/--config`: path to yolo config-file; defaults to `config/bdd_data.yaml`.
        `-w/--weights`: path to trained weights; defaults to `weights/yolov5s_bdd.pt`.

    Note: if the model is not found it uses `weights/download.sh` script to download model.
    """
    if args.task == "data-viz":
        v = Visualize(args.data_path)
        v.show_yolo_annotations()
    elif args.task == "plot-viz":
        v = Visualize(args.data_path)
        v.all()
    elif args.task == "infer":
        inference(args.data_path, args.weights)
    elif args.task == "pred-eval":
        evaluate(args.data_path, args.weights, args.config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
