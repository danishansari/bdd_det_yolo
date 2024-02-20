"""This file is to provide visualization functionalities for Yolo models

author: danish ansari
copyright: na
"""

from data_loader import BDDLoader
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sn
import os
from PIL import ImageDraw, ImageFont, ImageColor
import numpy as np
import pandas as pd


class Visualize:
    """
    Class to visulalize dataset distribution and attributes
    - plots train/val sample distribution
    - plots class wise sample distribution
    - plots size vs no of samples in training data
    - show yolo annotations plotted on images
    """

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
        """Function to load train and eval data classwise in memory from dataloader
        to perform analysis.
        """
        for im, lab, atr in self.trn_loader:
            b = self.trn_loader.scale_box(lab)
            for i, (c, b) in enumerate(lab):
                cat = self.trn_loader.data.config["names"][c]
                if cat in self.train_val_data["train"]["class"]:
                    self.train_val_data["train"]["class"][cat][0] += 1
                    self.train_val_data["train"]["class"][cat][1].append(b[2])
                    self.train_val_data["train"]["class"][cat][2].append(b[3])
                    self.train_val_data["train"]["class"][cat][3].append(atr[i])
                else:
                    self.train_val_data["train"]["class"][cat] = [
                        1,
                        [b[2]],
                        [b[3]],
                        [atr[i]],
                    ]

        for im, lab, atr in self.val_loader:
            b = self.trn_loader.scale_box(lab)
            for i, (c, b) in enumerate(lab):
                cat = self.trn_loader.data.config["names"][c]
                if cat in self.train_val_data["val"]["class"]:
                    self.train_val_data["val"]["class"][cat][0] += 1
                    self.train_val_data["val"]["class"][cat][1].append(b[2])
                    self.train_val_data["val"]["class"][cat][2].append(b[3])
                    self.train_val_data["val"]["class"][cat][2].append(atr[i])
                else:
                    self.train_val_data["val"]["class"][cat] = [
                        1,
                        [b[2]],
                        [b[3]],
                        [atr[i]],
                    ]

    def train_val_distribution(self):
        """Function to plot tain-val data distribution"""
        data = ["train", "val"]
        samples = [len(self.trn_loader), len(self.val_loader)]

        plt.figure(figsize=(8, 6))
        plt.bar(data, samples, color="blue", width=0.4)
        plt.ylabel("Number of samples")
        plt.title("BDD dataset training data distribution")
        plt.savefig("plots/train_val_dist.png")

    def class_wise_data_distribution(self):
        """Function to plot class wise data distribution"""
        data = self.train_val_data["train"]["class"].keys()
        train_count = [x[0] for x in self.train_val_data["train"]["class"].values()]
        X_axis = np.arange(len(data))
        plt.figure(figsize=(8, 6))
        plt.xticks(X_axis, data)
        plt.xticks(rotation=45, ha="right")
        plt.bar(data, train_count, 0.4, color="blue")
        plt.xlabel("class")
        plt.ylabel("Number of samples")
        plt.title("BDD dataset training class distribution")
        plt.savefig("plots/training_class_dist.png")
        data = self.train_val_data["val"]["class"].keys()
        val_count = [x[0] for x in self.train_val_data["val"]["class"].values()]
        X_axis = np.arange(len(data))
        plt.figure(figsize=(8, 6))
        plt.xticks(X_axis, data)
        plt.xticks(rotation=45, ha="right")
        plt.bar(data, val_count, 0.4, color="orange")
        plt.xlabel("class")
        plt.ylabel("Number of samples")
        plt.title("BDD dataset training class distribution")
        plt.savefig("plots/eval_class_dist.png")

    def object_size_distribution(self):
        """Function to plot width/heigh confusion metrix"""
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

    def class_size_distribution(self):
        """Function to plot object-size(area) vs no of sample distribution"""
        class_area_avg = {}
        tot_area = 0
        tot_sample = 0
        for c in self.train_val_data["train"]["class"]:
            for i in range(len(self.train_val_data["train"]["class"][c][1])):
                w = self.train_val_data["train"]["class"][c][1][i] // 100
                h = self.train_val_data["train"]["class"][c][2][i] // 100
                tot_area += w * h
                tot_sample += 1
                if c in class_area_avg:
                    class_area_avg[c][0] += w * h
                    class_area_avg[c][1] += 1
                else:
                    class_area_avg[c] = [(w * h), 1]
        color = cm.rainbow(np.linspace(0, 1, 10))
        fig, ax = plt.subplots()
        for i, c in enumerate(class_area_avg.keys()):
            circle = plt.Circle(
                ((i + 1) / 10, (class_area_avg[c][1] / tot_sample)),
                class_area_avg[c][0] / (tot_area * 2),
                color=color[i],
            )
            ax.annotate(
                c, xy=((i + 1) / 10, (class_area_avg[c][1] / tot_sample)), fontsize=12
            )
            ax.add_patch(circle)
        plt.ylabel("Number of samples")
        plt.title("Object-size vs #samples distribution")
        plt.savefig("plots/object_size_dist.png")

    def data_attributes_distribution(self):
        """Function to plot data occlusion and trucation distribution"""
        attrib_counts = {"occuluded": [0, 0], "truncated": [0, 0]}
        for c in self.train_val_data["train"]["class"]:
            for i in range(len(self.train_val_data["train"]["class"][c][3])):
                atr = self.train_val_data["train"]["class"][c][3][i]
                attrib_counts["occuluded"][1] += 1
                attrib_counts["truncated"][1] += 1
                if atr[1]:
                    attrib_counts["occuluded"][0] += 1
                if atr[2]:
                    attrib_counts["truncated"][0] += 1

        fig, ax = plt.subplots()
        bottom = np.zeros(2)
        for cls, count in attrib_counts.items():
            p = ax.bar(["occuluded", "truncated"], count, 0.5, label=cls, bottom=bottom)
            bottom += count
        plt.savefig("plots/object_attr_dist.png")

    def train_size_vs_map(self):
        """Function to plot train/val size vs map"""
        data = self.train_val_data["train"]["class"].keys()
        train_count = [x[0] for x in self.train_val_data["train"]["class"].values()]
        tot = sum(train_count)
        for i in range(len(train_count)):
            train_count[i] /= tot
        val_count = [0.57, 0.62, 0.77, 0.59, 0.43, 0.43, 0.62, 0.60, 0.42, 0.0]
        X_axis = np.arange(len(data))
        plt.figure(figsize=(8, 6))
        plt.xticks(X_axis, data)
        plt.xticks(rotation=45, ha="right")
        plt.bar(X_axis - 0.2, train_count, 0.4, label="samples")
        plt.bar(X_axis + 0.2, val_count, 0.4, label="mAP")
        plt.legend()
        plt.xlabel("class")
        plt.ylabel("Number of samples")
        plt.title("BDD dataset samples vs map distribution")
        plt.savefig("plots/sample_map_dist.png")

    def show_yolo_annotations(self) -> None:
        """Function to overlay annotations on images"""
        os.makedirs("plots/images", exist_ok=True)
        colors = list(ImageColor.colormap.keys())[:10]
        font = ImageFont.truetype("plots/font/Arial.ttf", 24)
        for img, lab, atr in self.val_loader:
            draw = ImageDraw.Draw(img)
            lab = self.val_loader.scale_box(lab)
            for ann in lab:
                c, b = ann
                cls = self.val_loader.data.config["names"][c]
                draw.rectangle(b, fill=None, outline=colors[c], width=3)
                draw.text((b[0] + 5, b[1]), cls[:3], font=font, fill="red")
            draw.text((5, 5), f"* {atr[0][3]}", align="left", font=font, fill="red")
            draw.text((5, 25), f"* {atr[0][4]}", align="left", font=font, fill="red")
            draw.text((5, 45), f"* {atr[0][5]}", align="left", font=font, fill="red")
            img.save("plots/images/tmp.jpg")
            if input("> ") == "s":
                img.save(f"plots/images/{os.path.basename(self.val_loader.curr_fname)}")

    def all(self):
        """Function to make all plots for data vizualization"""
        self.train_val_distribution()
        self.class_wise_data_distribution()
        self.object_size_distribution()
        self.class_size_distribution()
        self.data_attributes_distribution()
        self.train_size_vs_map()
