import shutil

import scipy
import skimage
import torch
import detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
DETECTRON_VERSION = detectron2.__version__
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION, "; detectron2: ", DETECTRON_VERSION)


import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper,  build_detection_test_loader, build_detection_train_loader

from detectron2.data import transforms as T

from Data.ML_Conf import get_dataset_dicts, get_custom_conf, get_metadata, init_tensorboard, dataset_name, \
    get_data_all_beads, get_dataset_dicts3

augmentations = [
            T.RandomBrightness(0.5, 2.),
            T.RandomContrast(0.5, 2.),
            T.RandomSaturation(0.5, 2.),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            #T.RandomSaturation(0.9, 1.1),
            T.RandomRotation([-180,180]),
            #T.RandomCrop("absolute", (702, 702))
        ]

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        custom_mapper = DatasetMapper(cfg, is_train=False, augmentations=augmentations)
        return build_detection_test_loader(cfg, dataset_name, mapper=custom_mapper)

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        #custom_mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        custom_mapper = DatasetMapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=custom_mapper)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    setup_logger()
    get_data_all_beads()
    # im = cv2.imread("Data/download.png")
    # cv2.imshow('def', im)
    # #cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    #
    #
    # cfg = get_cfg()
    # # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #
    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(im)
    #
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    #
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow('def', out.get_image()[:, :, ::-1])
    # #cv2.waitKey(0)
    # cv2.destroyAllWindows()

    from detectron2.structures import BoxMode

    dataset_metadata = get_metadata()
    dataset_dicts = get_dataset_dicts3("train")
    #print(dataset_dicts)

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=1, instance_mode=ColorMode.SEGMENTATION)
        out = visualizer.draw_dataset_dict(d)
        img = out.get_image()[:, :, ::-1]
        #img = cv2.resize(img, (int(702*1.5), int(800*1.5)))
        cv2.imshow("zi", img)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()


    #shutil.rmtree(cfg.OUTPUT_DIR)
    cfg = get_custom_conf()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    init_tensorboard(cfg)

    trainer = CustomTrainer(cfg)
    i = 0
    for t in trainer.build_train_loader(cfg):
        if i%1000 == 0:
            print(i)
        i += 1
    print("TOTAL", i)
    #trainer.resume_or_load(resume=False)
    #trainer.train()
