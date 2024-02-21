"""This file is to provide inference functionalities for Yolo models

author: danish ansari
copyright: na
"""

from ultralytics import YOLO
from data_loader import BDDLoader
from visualize import Visualize
from PIL import Image, ImageDraw, ImageColor, ImageFont
import os


def plot_predictions(
    image: Image, preds: list, attr: list, fname: str, viz: Visualize
) -> None:
    """Function to over bbox and class name on images

    Args:
        image (Image): PIL-Image data
        preds (list): list of predictions from model
        attr (list): list of attributes associated with the image
        fname (str): name of the image file
        viz (Visualize): visualizer for plotting predictions
    """
    print("Plots will be saved in `plots/images/tmp.jpg`.")
    print(
        "Press `enter` for next, press `s` and then `enter` to save, `ctrl`+`c` to quit:"
    )
    os.makedirs("plots/images", exist_ok=True)
    image = viz.plot_labels(image, preds, attr)
    image.save("plots/images/tmp.jpg")
    if input(f"{fname} > ") == "s":
        image.save(f"plots/images/{os.path.basename(fname)}")


def inference(data_path: str, weights: str, classes: list = []) -> None:
    """Funtion to make prediction on val dataset and overlay on images
    click `enter` for next image
    click `s` + `enter` to save the plot

    Args:
        data_path (str): dataset directory path
        weights (str): path to trained weights
        classes (list): list of classes to be considered only
    """
    data = BDDLoader(data_path, "val")
    if not os.path.exists(weights):
        os.system("sh weights/download.sh")
    model = YOLO(weights)
    viz = Visualize(data_path)
    for image, lab, attr in data:
        result = model(image, verbose=False)
        lab = data.scale_box(lab)
        assert len(lab) == len(attr)
        preds = []
        for r in result:
            name = r.names
            pred = list(map(int, r.boxes.cls.numpy().tolist()))
            conf = r.boxes.conf.numpy().tolist()
            bbox = r.boxes.xyxy.numpy().tolist()
            assert len(pred) == len(conf) == len(bbox)
            for b, p, c in zip(bbox, pred, conf):
                if classes and name[p] not in classes:
                    continue
                preds.append([p, list(map(int, b)), c])

        plot_predictions(image, preds, attr, data.curr_fname, viz)
