"""This file is to provide inference functionalities for Yolo models

author: danish ansari
copyright: na
"""

from ultralytics import YOLO
from data_loader import BDDLoader
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser("Yolv5 Training")
parser.add_argument(
    "-w",
    "--weights",
    type=str,
    default="weights/yolov5s_bdd.pt",
    help="path to weights",
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


def plot_predictions(image, bboxes):
    pass


def main(args, show=True):
    data = BDDLoader(args.data_path, "val")
    model = YOLO(args.weights)

    for image, _, _ in tqdm(data):
        result = model(image, verbose=False)
        bboxes = []
        for r in result:
            name = r.names
            pred = list(map(int, r.boxes.cls.numpy().tolist()))
            conf = r.boxes.conf.numpy().tolist()
            bbox = r.boxes.xyxy.numpy().tolist()
            assert len(pred) == len(conf) == len(bbox)
            for b, p, c in zip(bbox, pred, conf):
                bboxes.append([list(map(int, b)), name[p], c])
        if show:
            plot_predictions(image, bboxes)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
