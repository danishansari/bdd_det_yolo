"""This file provides class implementation to prepare dataset for Yolo
Work on bdd-dataset loads annoatation json, converts to yolo and save.

author: danish ansari
copyright: na
"""

from torch.utils.data import Dataset
import random
import os
from PIL import Image
from data_prep import DataPrep
from typing import Any, List


class BDDLoader(Dataset):
    """BDD Dataset loader class"""

    def __init__(
        self, path: str, dataset: str, load_img: bool = True, transform: Any = None
    ) -> None:
        super().__init__()
        self.root_path = path
        self.load_image = load_img
        self.data = DataPrep(path=path)
        self.transform = transform
        self.curr_fname = ""
        self.files = self.load_img_files(dataset)

    def load_img_files(self, dataset: str) -> List:
        """Function to load image files name into memory

        Args:
            dataset (str): dataset type - train/val
        """
        image_path = os.path.join(self.data.image_path, dataset)
        fnames = []
        for files in os.listdir(image_path):
            fnames.append(os.path.join(image_path, files))
        return fnames

    def __len__(self) -> int:
        """class atribute to return no of images present"""
        return len(self.files)

    def on_epoch_start(self) -> None:
        """Function to shuffle image list"""
        random.shuffle(self.files)

    def get_labels_n_attribs(self, index: int) -> List:
        """Function to get labels and attributes associated to images at a particular index

        Args:
            index (int): index location of image
        """
        txt_files = (
            self.files[index].replace("/images", "/labels").replace(".jpg", ".txt")
        )
        labels = []
        if os.path.exists(txt_files):
            with open(txt_files) as f:
                for line in f.readlines():
                    v = line.strip().split(" ")
                    c = int(v[0])  # class
                    labels.append([c, list(map(float, v[1:]))])
        txt_files = (
            self.files[index].replace("/images", "/labels").replace(".jpg", "_meta.csv")
        )
        attributes = []
        if os.path.exists(txt_files):
            with open(txt_files) as f:
                for line in f.readlines():
                    v = line.strip().split(",")
                    for i in range(len(v)):
                        v[i] = v[i].strip()
                    v[:3] = list(map(int, v[:3]))
                    attributes.append(v)
        return [labels, attributes]

    def scale_box(self, bboxes: list, xyxy: bool = True) -> list:
        """Function to scale prediction to original image size

        Args:
            bboxes: list of bbox predicted
            xyxy (bool): expected output format; defaults to True
        """
        sz = self.data.img_size
        for c, bbox in bboxes:
            bbox[2] = bbox[2] * sz[0]  # width
            bbox[3] = bbox[3] * sz[1]  # height
            bbox[0] = (bbox[0] * sz[0]) - (bbox[2] / 2)  # x
            bbox[1] = (bbox[1] * sz[1]) - (bbox[3] / 2)  # y
            if xyxy:
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
            bbox[:] = list(map(int, bbox[-4:]))
        return bboxes

    def __getitem__(self, index: int) -> Any:
        """Class attribute to get items at an index

        Args:
            index (int): index location to access
        """
        image = self.files[index]
        self.curr_fname = image
        if self.load_image:
            image = Image.open(image)  # RGB image
        if self.transform:
            pass
        labels, attribs = self.get_labels_n_attribs(index)
        return image, labels, attribs
