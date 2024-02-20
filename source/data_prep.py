"""This file provides class implementation to prepare dataset for Yolo
Work on bdd-dataset loads annoatation json, converts to yolo and save.

author: danish ansari
copyright: na
"""

import os
import json
from PIL import Image
import yaml
from tqdm import tqdm
from loguru import logger


class DataPrep:
    """
    Class to prepare dataset to be consumed by Yolo (ultralytics)
    - Takes bdd dataset root direcory as input
    - Loads train/eval json files in memory
    - Iterate over each image-name in loaded data
    - converts and saves detection annotations to yolo format
    """

    def __init__(self, path: str, img_size: tuple = (1280, 720)) -> None:
        self.root_path = path
        # assignment_data_bdd/bdd100k_images_100k/bdd100k
        self.image_path = os.path.join(
            self.root_path, "bdd100k_images_100k/bdd100k/images/100k"
        )
        self.img_size = img_size

        # load data-config
        with open("config/bdd_data.yaml") as f:
            self.config = yaml.safe_load(f)

        self.class_map = {v: k for k, v in self.config["names"].items()}

        # generated yolo annotations path parallel to images
        self.label_path = os.path.join(
            self.root_path, "bdd100k_images_100k/bdd100k/labels/100k"
        )

        # generate yolo annotations if does not exists
        if not self.is_data_ready():
            os.makedirs(self.label_path)
            # assignment_data_bdd/bdd100k_labels_release/bdd100k
            self.prep_labels(
                os.path.join(self.root_path, "bdd100k_labels_release/bdd100k/labels")
            )
        else:
            logger.info(f"Dataset already prepared: {self.label_path}")

    def is_data_ready(self) -> bool:
        """Function to check is dataset already exists
        Returns:
            bool: whether or not labels-directory exists
        """
        return os.path.exists(self.label_path)

    def get_json_data(self, path: str) -> list:
        """Function to load json and return data

        Args:
            path(str): path to json file
        """
        data = []
        if path.endswith(".json"):
            with open(path, "r") as fp:
                data = json.load(fp)
        return data

    def prep_labels(self, path: str) -> None:
        """Function to load, convert and save anotataions to yolo format.
        Args:
            path(str): path to json annotations files train/eval.json

        """
        img_size = self.img_size
        for items in os.listdir(path):
            dataset = items.split("_")[-1][:-5]
            os.makedirs(os.path.join(self.label_path, dataset), exist_ok=True)
            logger.info(f"Preparing dataset for: {dataset}")
            with open(os.path.join(os.path.dirname(path), dataset + ".txt"), "w") as f1:
                for d in tqdm(self.get_json_data(os.path.join(path, items))):
                    img_path = os.path.join(
                        self.image_path, items.split("_")[-1][:-5], d["name"]
                    )
                    f1.write(f"{img_path}\n")
                    with open(
                        os.path.join(
                            self.label_path,
                            dataset,
                            os.path.basename(img_path).replace(".jpg", ".txt"),
                        ),
                        "w",
                    ) as f2:
                        with open(
                            os.path.join(
                                self.label_path,
                                dataset,
                                os.path.basename(img_path).replace(".jpg", "_meta.csv"),
                            ),
                            "w",
                        ) as f3:
                            if not self.img_size:
                                img_size = Image.open(
                                    os.path.join(
                                        self.image_path,
                                        items.split("_")[-1][:-5],
                                        d["name"],
                                    )
                                )
                            W, H = img_size
                            for label in d["labels"]:
                                if "box2d" in label:
                                    b = label["box2d"]
                                    x1, y1, x2, y2 = (
                                        b["x1"],
                                        b["y1"],
                                        b["x2"],
                                        b["y2"],
                                    )
                                    w, h = abs(x2 - x1), abs(y2 - y1)
                                    cx, cy = x1 + (w / 2), y1 + (h / 2)
                                    c = self.class_map[label["category"]]
                                    f2.write(
                                        "%d %f %f %f %f\n"
                                        % (
                                            c,
                                            cx / W,
                                            cy / H,
                                            w / W,
                                            h / H,
                                        )
                                    )

                                    f3.write(
                                        "%d, %d, %d, %s, %s, %s\n"
                                        % (
                                            c,
                                            label["attributes"]["occluded"],
                                            label["attributes"]["truncated"],
                                            d["attributes"]["weather"],
                                            d["attributes"]["scene"],
                                            d["attributes"]["timeofday"],
                                        )
                                    )
        logger.info("Dataset is ready for YoloV5")
