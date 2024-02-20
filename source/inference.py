"""This file is to provide inference functionalities for Yolo models

author: danish ansari
copyright: na
"""

from ultralytics import YOLO
from data_loader import BDDLoader
from tqdm import tqdm
import os


def plot_predictions(image, bboxes):
    pass


def inference(data_path, weights):
    data = BDDLoader(data_path, "val")
    model = YOLO(weights)

    for image, lab, attr in tqdm(data):
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
                preds.append([name[p], [list(map(int, b))], c])

        plot_predictions(image, preds)
