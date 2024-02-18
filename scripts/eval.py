"""This file is to provide evaluation functionalities for Yolo models

author: danish ansari
copyright: na
"""

from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser("Yolv5 Training")
parser.add_argument(
    "-w", "--weights", default="weights/yolov5s.pt", help="path to weights"
)
parser.add_argument(
    "-c", "--config", default="config/bdd_data.yaml", help="path to model config"
)
parser.add_argument("-b", "--batch", default=1, help="batch size to train model")


def main(args):
    model = YOLO("yolov5s.yaml").load(args.weights)
    model.val(data=args.config, batch=args.batch)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
