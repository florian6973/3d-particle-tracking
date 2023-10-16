import glob
import json
import os
import re

import cv2
import numpy as np
import pandas as pd
from detectron2.config import get_cfg

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from tensorboard import program

from Data.Models import Session, Experiment, Frame, Bead

dataset_name = "beads"
dataset_data = None

def init_tensorboard(cfg):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', cfg.OUTPUT_DIR, '--bind_all'])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


def get_custom_conf(nmax=114):
    #dlmodel = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    dlmodel = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(dlmodel))
    cfg.DATASETS.TRAIN = (dataset_name + "_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4 # 3
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(dlmodel)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1 #2 #10 # 2
    cfg.SOLVER.BASE_LR = 0.0002 #0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 120000 #40000 # 6000 #3000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #256 # 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.TEST.DETECTIONS_PER_IMAGE = nmax # https://github.com/facebookresearch/detectron2/issues/1045 https://github.com/facebookresearch/detectron2/issues/1164

    #cfg.OUTPUT_DIR = "output_DA"
    #cfg.OUTPUT_DIR = r'Z:\fasrc\users\fpollet\data\output_success10/'
    cfg.OUTPUT_DIR = r'Z:\fasrc\users\fpollet\data\output_success11/'
    #r"C:\Users\XPSLab1\Documents\Beads\shaking-beads\output/"#r'Z:\fasrc\users\fpollet\data\output_dl_data3/'
    return cfg


def get_custom_conf_old(nmax=114):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name + "_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4 # 3
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1 #2 #10 # 2
    cfg.SOLVER.BASE_LR = 0.0002 #0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 120000 #40000 # 6000 #3000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #256 # 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.TEST.DETECTIONS_PER_IMAGE = nmax # https://github.com/facebookresearch/detectron2/issues/1045 https://github.com/facebookresearch/detectron2/issues/1164

    #cfg.OUTPUT_DIR = "output_DA"
    cfg.OUTPUT_DIR = r'Z:\fasrc\users\fpollet\data\output_dl'
    return cfg

def split_test_train():
    files = glob.glob(os.path.join('counts', "*.png.txt"))
    all = []
    train = []
    test = {}
    for file in files:
        n, v, i, img = re.findall(r"counts\\(.*)_(.*)_(.*)_Dump_(.*).png.txt", file)[0]
        all.append((n, v, i, img))
    for n, v, i, img in all:
        if (n, v, i) not in test:
            test[(n, v, i)] = [(n, v, i, img)]
        else:
            train.append((n, v, i, img))
    test = list(map(lambda x: x[0], test.values()))

    return train, test

def split_test_train2():
    global dataset_data
    all = dataset_data.groupby(["n", "v", "i", "folder", "f", "path"]).groups
    #print(all)
    #all = []
    train = []
    test = {}
    # for file in files:
    #     n, v, i, img = re.findall(r"counts\\(.*)_(.*)_(.*)_Dump_(.*).png.txt", file)[0]
    #     all.append((n, v, i, img))
    for n, v, i, folder, f, img in all:
        img = os.path.join(folder, "countDump", img)
        if (n, v, i) not in test:
            test[(n, v, i)] = [(n, v, i, f, img)]
        else:
            train.append((n, v, i, f, img))
    test = list(map(lambda x: x[0], test.values()))

    return train, test

def get_dataset_dicts(type_dataset):
    dataset_dicts = []
    train, test = split_test_train()
    print(train)
    print(test)
    if type_dataset == "train":
        extract_metadata_img(dataset_dicts, train)
    elif type_dataset == "val":
        extract_metadata_img(dataset_dicts, test)
    return dataset_dicts


def get_dataset_dicts3(type_dataset):
    dataset_dicts = []
    train, test = split_test_train2()
    #print(train)
    #print(test)
    if type_dataset == "train":
        extract_metadata_img3(dataset_dicts, train)
    elif type_dataset == "val":
        extract_metadata_img3(dataset_dicts, test)
    return dataset_dicts


def get_data_all_dataset():
    global dataset_data
    session = Session()
    res = session.query(Experiment.exp_type, Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
                        Experiment.exp_folder,
                        Frame.frame_number, Frame.frame_nbtop, Frame.frame_img_path,
                        Bead.bead_x, Bead.bead_y, Bead.bead_r, Bead.bead_top) \
        .join(Experiment, Experiment.id == Frame.experiment_id) \
        .join(Bead, Bead.frame_id == Frame.id) \
        .where(Frame.frame_nbtop != None) \
        .order_by(Experiment.exp_type, Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
                  Frame.frame_number)
    print(res)
    res = res.all()
    #print(res)
    dataset_data = pd.DataFrame(res, columns=['type', 'n', 'v', 'i', 'folder', 'f', 'nb', 'path', 'x', 'y', 'r', 'top'])
    print(dataset_data)
    session.close()
    print('Query done')
    return dataset_data


def get_data_all_beads():
    global dataset_data
    session = Session()
    res = session.query(Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
                        Experiment.exp_folder,
                        Frame.frame_number, Frame.frame_nbtop, Frame.frame_img_path,
                        Bead.bead_x, Bead.bead_y, Bead.bead_r, Bead.bead_top) \
        .join(Experiment, Experiment.id == Frame.experiment_id) \
        .join(Bead, Bead.frame_id == Frame.id) \
        .where((Experiment.exp_type == 2)
               & (Frame.frame_nbtop != None)) \
        .order_by(Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
                  Frame.frame_number)
    print(res)
    res = res.all()
    #print(res)
    dataset_data = pd.DataFrame(res, columns=['n', 'v', 'i', 'folder', 'f', 'nb', 'path', 'x', 'y', 'r', 'top'])
    print(dataset_data)
    session.close()
    print('Query done')
    return dataset_data


def extract_metadata_img(dataset_dicts: list[dict], train: list[tuple]) -> None:
    """

    :rtype: None
    """
    for n, v, i, img in train:
        record = {}
        filename = rf"H:\Data_2\n_beads_{n:03}_Delrin_white_0.25_Vmin_210_Vmax_350_Step_10_Cooling_140_humidity_55_65\velocity_{v:03}\iteration_{i:03}\countDump\Dump_{img}.png"
        height, width = (702, 800)
        record["file_name"] = filename
        record["image_id"] = f"{n}-{v}-{i}-{img}"
        record["height"] = height
        record["width"] = width

        objs = []
        data_file = f"counts/{n:03}_{v:03}_{i:03}_Dump_{img}.png.txt"
        data_annot = np.loadtxt(data_file, dtype=int)
        for x, y, r, t in data_annot:
            poly = []
            n = 12
            for theta in np.linspace(0, 2 * np.pi, n):
                poly.append((x + r * np.cos(theta), y + r * np.sin(theta)))
            poly = [p for x in poly for p in x]
            #print(poly)
            obj = {
                "bbox": [x-r, y-r, x+r, y+r],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": t,
                'iscrowd': 0
            }
            objs.append(obj)

        record["annotations"] = objs
        #print(record)
        dataset_dicts.append(record)



def extract_metadata_img3(dataset_dicts, train):
    global dataset_data
    #print("calling metadata")
    for n, v, i, f, img in train:
        record = {}
        filename = img #rf"H:\Data_2\n_beads_{n:03}_Delrin_white_0.25_Vmin_210_Vmax_350_Step_10_Cooling_140_humidity_55_65\velocity_{v:03}\iteration_{i:03}\countDump\Dump_{img}.png"
        height, width = (702, 800)
        record["file_name"] = filename
        record["image_id"] = f"{n}-{v}-{i}-{f}"
        record["height"] = height
        record["width"] = width

        objs = []
        #data_file = f"counts/{n:03}_{v:03}_{i:03}_Dump_{f}.png.txt"
        #data_annot = np.loadtxt(data_file, dtype=int)
        data_annot = dataset_data[(dataset_data['n'] == n)
                                  & (dataset_data['v'] == v)
                                  & (dataset_data['i'] == i)
                                  & (dataset_data['f'] == f)][['x', 'y', 'r', 'top']].values
        for x, y, r, t in data_annot:
            poly = []
            n = 12
            for theta in np.linspace(0, 2 * np.pi, n):
                poly.append((x + r * np.cos(theta), y + r * np.sin(theta)))
            poly = [p for x in poly for p in x]
            #print(poly)
            obj = {
                "bbox": [x-r, y-r, x+r, y+r],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": t,
                'iscrowd': 0
            }
            objs.append(obj)

        record["annotations"] = objs
        #print(record)
        dataset_dicts.append(record)
        #break



def get_dataset_dicts2(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register():
    for d in ["train", "val"]:
        DatasetCatalog.register(dataset_name + '_' + d, lambda d=d: get_dataset_dicts(d))
        MetadataCatalog.get(dataset_name + '_' + d).set(thing_classes=["0", "1"]).set(thing_colors=[(100,100,0), (100,0,0)]).set(stuff_colors=[(100,100,0), (100,0,0)])

def get_metadata():
    register()
    return MetadataCatalog.get(dataset_name + "_train")

if __name__ == "__main__":
    get_data_all_beads()
    #print(get_dataset_dicts3("val"))
    #print(get_dataset_dicts3("train"))
