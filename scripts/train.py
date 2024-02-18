"""This file is to provide training functionalities for Yolo models

author: danish ansari
copyright: na
"""

import torch
from ultralytics import YOLO
from typing import Any
import argparse

parser = argparse.ArgumentParser("Yolv5 Training")
parser.add_argument(
    "-w", "--weights", default="weights/yolov5s.pt", help="path to weights"
)
parser.add_argument(
    "-s", "--image-size", default=640, help="image input size to the model"
)
parser.add_argument("-e", "--epochs", default=1, help="number of epochs for training")
parser.add_argument(
    "-c", "--config", default="config/bdd_data.yaml", help="path to model config"
)
parser.add_argument("-b", "--batch", default=1, help="batch size to train model")


def train(model: YOLO, data: str, epochs: int, imgsz: int, batch: int) -> None:
    """Function to train yolo models

    Args:
        model(YOLO): Yolo model to train
        data(str): path to data-config file(yaml)
        epochs(int): number of epochs to run training
        imgsz(int): input image size to model
        batch(int): batch size for training
    """
    # Train the model
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch)


def main(args: Any) -> None:
    """Main function to train a yolo model

    Args:
        args(Any): command line arguments
    """
    model = YOLO("yolov5s.yaml").load(args.weights)
    train(model, args.config, args.epochs, args.image_size, args.batch)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
