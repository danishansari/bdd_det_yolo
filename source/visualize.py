"""This file is to provide visualization functionalities for Yolo models

author: danish ansari
copyright: na
"""

from data_prep import DataPrep


class Visualize:

    def __init__(self, path: str) -> None:
        self.data = DataPrep(path=path)

    def train_data_dist(self):
        pass

    def class_wise_data_distribution(self):
        pass

    def object_size_distribution(self):
        pass

    def show_data_attributes(self):
        pass

    def show_yolo_annotations(self):
        pass
