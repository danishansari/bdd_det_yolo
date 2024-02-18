"""This file is to provide inference functionalities for Yolo models

author: danish ansari
copyright: na
"""

from ultralytics import YOLO
from data_loader import BDDLoader
import argparse

parser = argparse.ArgumentParser("Yolv5 Training")
parser.add_argument(
    "-w", "--weights", default="weights/yolov5s.pt", help="path to weights"
)
parser.add_argument("-d", "--data-path", default="/dataset", help="path to dataset")
parser.add_argument(
    "-c", "--config", default="config/bdd_data.yaml", help="path to model config"
)
parser.add_argument("-b", "--batch", default=1, help="batch size to train model")


def main(args):
    data = BDDLoader(args.data_path)
    model = YOLO(args.weights)
    for image, _, _ in data:
        result = model(image)
        print(result)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
