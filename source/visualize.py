"""This file is to provide visualization functionalities for Yolo models

author: danish ansari
copyright: na
"""

from data_loader import BDDLoader
from matplotlib import pyplot as plt
import seaborn as sn
import os
import numpy as np
import pandas as pd


class Visualize:

    def __init__(self, path: str) -> None:
        self.trn_loader = BDDLoader(path, "train")
        self.val_loader = BDDLoader(path, "val")
        self.train_val_data = {
            "train": {"class": {}},
            "val": {"class": {}},
        }
        self.load_data()
        os.makedirs("plots", exist_ok=True)

    def load_data(self):
        for im, lab in self.trn_loader:
            b = self.trn_loader.scale_box(lab)
            for c, b in lab:
                cat = self.trn_loader.data.config["names"][c]
                if cat in self.train_val_data["train"]["class"]:
                    self.train_val_data["train"]["class"][cat][0] += 1
                    self.train_val_data["train"]["class"][cat][1].append(b[2])
                    self.train_val_data["train"]["class"][cat][2].append(b[3])
                else:
                    self.train_val_data["train"]["class"][cat] = [1, [b[2]], [b[3]]]
        for im, lab in self.val_loader:
            b = self.trn_loader.scale_box(lab)
            for c, b in lab:
                cat = self.trn_loader.data.config["names"][c]
                if cat in self.train_val_data["val"]["class"]:
                    self.train_val_data["val"]["class"][cat][0] += 1
                    self.train_val_data["val"]["class"][cat][1].append(b[2])
                    self.train_val_data["val"]["class"][cat][2].append(b[3])
                else:
                    self.train_val_data["val"]["class"][cat] = [1, [b[2]], [b[3]]]

    def train_val_distribution(self):
        data = ["train", "val"]
        samples = [len(self.trn_loader), len(self.val_loader)]

        fig = plt.figure(figsize=(8, 6))
        plt.bar(data, samples, color="blue", width=0.4)
        plt.ylabel("Number of samples")
        plt.title("BDD dataset training data distribution")
        plt.savefig("plots/train_val_dist.png")
        plt.show()

    def class_wise_data_distribution(self):
        data = self.train_val_data["train"]["class"].keys()
        train_count = [x[0] for x in self.train_val_data["train"]["class"].values()]
        val_count = [x[0] for x in self.train_val_data["train"]["class"].values()]
        X_axis = np.arange(len(data))
        fig = plt.figure(figsize=(8, 6))
        plt.xticks(X_axis, data)
        plt.xticks(rotation=45, ha="right")
        plt.bar(X_axis - 0.2, train_count, 0.4, label="train")
        plt.bar(X_axis + 0.2, val_count, 0.4, label="train")
        plt.legend()
        plt.xlabel("class")
        plt.ylabel("Number of samples")
        plt.title("BDD dataset training class distribution")
        plt.savefig("plots/training_class_dist.png")
        plt.show()

    def object_size_distribution(self):
        array = [[0 for i in range(13)] for j in range(8)]
        tot = 0
        for c in self.train_val_data["train"]["class"]:
            for i in range(len(self.train_val_data["train"]["class"][c][1])):
                w = self.train_val_data["train"]["class"][c][1][i] // 100
                h = self.train_val_data["train"]["class"][c][2][i] // 100
                tot += 1
                array[h][w] = array[h][w] + (1 / tot)

        df_cm = pd.DataFrame(
            array,
            index=[i * 100 for i in range(8)],
            columns=[i * 100 for i in range(13)],
        )
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, fmt=".1f")
        plt.savefig("plots/training_size_dist.png")
        plt.show()

    def class_size_distribution(self):
        class_area_avg = {}
        tot = 0
        for c in self.train_val_data["train"]["class"]:
            for i in range(len(self.train_val_data["train"]["class"][c][1])):
                w = self.train_val_data["train"]["class"][c][1][i] // 100
                h = self.train_val_data["train"]["class"][c][2][i] // 100
                tot += 1
                if c in class_area_avg:
                    class_area_avg[c] += (w * h) / tot
                else:
                    class_area_avg[c] = (w * h) / tot
        fig = plt.figure(figsize=(8, 6))
        plt.bar(class_area_avg.keys(), class_area_avg.values(), color="blue", width=0.4)
        plt.ylabel("Average area")
        plt.title("BDD dataset training area distribution")
        plt.savefig("plots/training_area_dist.png")
        plt.show()

    def data_attributes_distribution(self):
        pass

    def show_yolo_annotations(self):
        pass


def main():
    v = Visualize(path="/home/danish/danish/datasets/assignment_data_bdd")
    # v.train_val_dist()
    # v.class_wise_data_distribution()
    # v.object_size_distribution()
    v.class_size_distribution()


if __name__ == "__main__":
    main()
