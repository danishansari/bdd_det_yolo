"""This file is to provide evaluation functionalities for Yolo models

author: danish ansari
copyright: na
"""

from ultralytics import YOLO
from data_loader import BDDLoader
from tqdm import tqdm
import os
from loguru import logger


def create_test_path(data_path: str) -> None:
    """Function to create temporary directorty for doing evaluations

    Args:
        data_path (str): path to dataset
    """
    test_path = os.path.join(data_path, "bdd100k_images_100k/bdd100k/images/100k/temp")
    if os.path.exists(test_path):
        os.system(f"rm -rf {test_path}")
    os.makedirs(test_path)
    text_path = os.path.join(data_path, "bdd100k_images_100k/bdd100k/labels/100k/temp")
    if os.path.exists(text_path):
        os.system(f"rm -rf {text_path}")
    os.makedirs(text_path)


def evaluate_(data_path: str, filenames: str, weights: str, config: str) -> None:
    """Function to evaluate model on given filename

    Args:
        data_path (str): path to dataset
        filenames (str): image-filenames to evalute model on
        weights (str): path to trained weights
        config (str): yolo-config file
    """
    create_test_path(data_path)
    with open(
        os.path.join(data_path, "bdd100k_images_100k/bdd100k/test.txt"), "w"
    ) as fp:
        for samples in tqdm(filenames):
            os.system(f"ln -s {samples} {samples.replace('val', 'temp')}")
            txt_path = samples.replace("/images/", "/labels/")[:-4] + ".txt"
            os.system(f"ln -s {txt_path} {txt_path.replace('val', 'temp')}")
            fp.write(f"{samples}\n")
    if not os.path.exists(weights):
        os.system("sh weights/download.sh")
    model = YOLO(weights)
    model.val(data=config, split="test")


def evaluate_dn(data_path: str, weights: str, config: str) -> None:
    """Function to evaluate day and night samples

    Args:
        data_path (str): path to dataset
        weights (str): path to trained weights
        config (str): yolo-config file
    """
    eval_samples = {"d": [], "n": []}
    data = BDDLoader(data_path, "val", load_img=False)
    for fname, lab, attr in tqdm(data):
        if attr[0][-1] == "night":
            eval_samples["d"].append(fname)
        else:
            eval_samples["n"].append(fname)
    for items in eval_samples:
        logger.info(f"Evaluating {items}:")
        evaluate_(data_path, eval_samples[items], weights, config)


def evaluate_ot(data_path: str, weights: str, config: str) -> None:
    """Function to evalaute occuluded and trucated samples

    Args:
        data_path (str): path to dataset
        weights (str): path to trained weights
        config (str): yolo-config file
    """
    eval_samples = {
        "oc": [],
        "tr": [],
    }
    data = BDDLoader(data_path, "val", load_img=False)
    for fname, lab, attr in tqdm(data):
        if attr[0][0]:
            eval_samples["oc"].append(fname)
        if attr[0][1]:
            eval_samples["tr"].append(fname)

    for items in eval_samples:
        logger.info(f"Evaluating {items}:")
        evaluate_(data_path, eval_samples[items], weights, config)


def evaluate_ws(data_path: str, weights: str, config: str) -> None:
    """Function to evaluate weather and different street samples

    Args:
        data_path (str): path to dataset
        weights (str): path to trained weights
        config (str): yolo-config file
    """
    eval_samples = {"w": {}, "s": {}}
    data = BDDLoader(data_path, "val", load_img=False)
    for fname, lab, attr in tqdm(data):
        if attr[0][3] in eval_samples["w"]:
            eval_samples["w"][attr[0][3]].append(fname)
        else:
            eval_samples["w"][attr[0][3]] = [fname]
        if attr[0][4] in eval_samples["s"]:
            eval_samples["s"][attr[0][4]].append(fname)
        else:
            eval_samples["s"][attr[0][4]] = [fname]

    for items in eval_samples:
        for items2 in eval_samples[items]:
            logger.info(f"Evaluating {items}/{items2}:")
            evaluate_(data_path, eval_samples[items], weights, config)


def evaluate(data_path: str, weights: str, config: str) -> None:
    """driver function to perform model evaluation in difference categories
    - like:
    * day/night samples
    * occuluded/trucated samples
    * different weather samples
    * different street location samples

    Args:
        data_path (str): path to dataset
        weights (str): path to trained weights
        config (str): yolo-config file
    """
    evaluate_dn(data_path, weights, config)
    evaluate_ot(data_path, weights, config)
    evaluate_ws(data_path, weights, config)
