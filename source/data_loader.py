"""This file provides class implementation to prepare dataset for Yolo
Work on bdd-dataset loads annoatation json, converts to yolo and save.

author: danish ansari
copyright: na
"""

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random


class BDDLoader(Dataset):
    """ """

    def __init__(self) -> None:
        super().__init__()
        self.files_labels = []

    def __len__(self) -> int:
        return len(self.files_labels)

    def on_epoch_start(self) -> None:
        random.shuffle(self.files_labels)

    def __getitem__(self, index: int) -> Any:
        img_fname = self.files_labels[index][0]
        img_label = self.files_labels[index][1]
        image = Image.open(img_fname)  # RGB image
        if self.augment:
            pass
        if self.transform:
            pass
        return image, img_label
