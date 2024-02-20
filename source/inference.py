"""This file is to provide inference functionalities for Yolo models

author: danish ansari
copyright: na
"""

from ultralytics import YOLO
from data_loader import BDDLoader
from PIL import Image, ImageDraw, ImageColor, ImageFont
import os


def plot_predictions(image: Image, preds: list, attr: list, fname: str):
    """Function to over bbox and class name on images

    Args:
        image (Image): PIL-Image data
        preds (list): list of predictions from model
        attr (list): list of attributes associated with the image
        fname (str): name of the image file
    """
    os.makedirs("plots/images", exist_ok=True)
    colors = list(ImageColor.colormap.keys())[:10]
    font = ImageFont.truetype("plots/font/Arial.ttf", 24)
    for bbox in preds:
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox[1], fill=None, outline=colors[bbox[-1]], width=3)
        draw.text((bbox[1][0] + 5, bbox[1][1]), bbox[0][:3], font=font, fill="red")
    draw.text((5, 5), f"* {attr[0][3]}", align="left", font=font, fill="red")
    draw.text((5, 25), f"* {attr[0][4]}", align="left", font=font, fill="red")
    draw.text((5, 45), f"* {attr[0][5]}", align="left", font=font, fill="red")
    image.save("plots/images/tmp.jpg")
    if input("> ") == "s":
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
    model = YOLO(weights)

    for image, lab, attr in data:
        result = model(image, verbose=False)
        lab = data.scale_box(lab)
        assert len(lab) == len(attr)
        for i in range(len(lab)):
            assert lab[i][0] == attr[i][0]
            lab[i] += [1.0] + attr[i][1:]
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
                preds.append([name[p], list(map(int, b)), c, p])

        plot_predictions(image, preds, attr, data.curr_fname)
