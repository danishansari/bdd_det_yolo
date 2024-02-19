"""This file is to provide main functionalities for Yolo models

author: danish ansari
copyright: na
"""

from visualize import Visualize
import argparse

parser = argparse.ArgumentParser("BDD-Data Yolov5 Detections")
parser.add_argument(
    "-t", "--task", type=str, choices=["data-visualize", "train", "eval"]
)
parser.add_argument(
    "-d", "--data-path", type=str, default="/dataset", help="root path to dataset"
)


def main(args):
    if args.task == "data-visualize":
        v = Visualize(args.data_path)
        v.all()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
