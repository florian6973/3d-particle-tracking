import glob
import os
import random

import cv2
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from Data import ML_Conf
import numpy as np

from Data.ML_Conf import get_data_all_beads


def get_center(box):
    return np.array(list(zip(((box[:, 0] + box[:, 2]) / 2).tolist(), ((box[:, 1] + box[:, 3]) / 2).tolist())))


def custom_predict(predictor, file=None, show=True, dataset_metadata=None, img_raw=None):
    if file == None:
        file = r"H:\Data_1\113\310_extract\velocity_310_iteration_011_archive\raw_images\img_06962.png"

        #file = r"H:\Data_1\113\310_extract\velocity_310_iteration_011_archive\raw_images\img_00058.png"

        #file = r"H:\Data_1\113\310_extract\velocity_310_iteration_011_archive\raw_images\img_05408.png"
        #file = \
        #    rf'H:\Data_2\n_beads_107_Delrin_white_0.25_Vmin_210_Vmax_350_Step_10_Cooling_140_humidity_55_65\velocity_350\iteration_004\countDump\Dump_63.png'

    im = cv2.imread(file)
    #if im.shape == (702,702,3): # be careful to the size of the image
    #    im = np.pad(im, [(0, 0), (49, 49), (0,0)], mode='constant')
    #print(im.shape)
    outputs = predictor(
        im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    data = outputs["instances"].to("cpu")
    boxes = data.pred_boxes.tensor.numpy()
    #print(boxes)
    classes = data.pred_classes.numpy()
    #print(classes)
    scores = data.scores.numpy()
    centers = get_center(boxes).astype(int)
    #print(f'{centers}')

    if show:
        for i in range(len(boxes)):
            x,y = centers[i]
            if classes[i] == 1:
                cv2.rectangle(im, (x - 1, y - 1), (x + 1, y + 1), (255, 128, 0), -1)
            else:
                cv2.rectangle(im, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)
        print(len(boxes))

        #print(outputs)

        cv2.imshow('test', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       scale=1.2,
                       instance_mode=ColorMode.SEGMENTATION
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('test', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return centers, classes

def get_predictor(n=114, cpu = False):
    setup_logger()
    cfg = ML_Conf.get_custom_conf(n)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # 0.2  # set a custom testing threshold
    #print(cfg.MODEL.DEVICE)
    if cpu:
        cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)

    return predictor


def test_performance(cfg):
    ML_Conf.register()
    evaluator = COCOEvaluator(ML_Conf.dataset_name + "_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, ML_Conf.dataset_name + "_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

def build_guesses_for_demo():
    files = glob.glob(r'C:\Users\XPSLab1\Documents\Beads\beads-counter-demo\imgs\*.png')
    predictor = get_predictor(113)
    for file in files:
        print("guessing", file)
        centers, classes = custom_predict(predictor,
                       file, False)
        circles = []
        for e in range(len(centers)):
            circles.append([centers[e][0], centers[e][1], 17, int(classes[e])])
        circles = np.array(circles)
        np.savetxt(rf'C:\Users\XPSLab1\Documents\Beads\beads-counter-demo\guesses\{os.path.basename(file) + ".txt"}', circles)

def build_guesses_for(v, it):
    files = glob.glob(rf'H:\Data_1\113\{v}_extract\velocity_{v}_iteration_{it:03}_archive\raw_images\*.png')
    predictor = get_predictor(113)
    folder = rf'H:\Data_1\113\{v}_extract\velocity_{v}_iteration_{it:03}_archive\counts'
    if not os.path.exists(folder):
        os.mkdir(folder)
    for i in range(19, len(files), 20): # todo multithread
        # can't use indices because some files are missing
        file = rf'H:\Data_1\113\{v}_extract\velocity_{v}_iteration_{it:03}_archive\raw_images\img_{i:05}.png'
        #file = files[i]
        print("guessing", file)
        try:
            centers, classes = custom_predict(predictor,
                           file, False)
            circles = []
            for e in range(len(centers)):
                circles.append([centers[e][0], centers[e][1], 17, int(classes[e])])
            circles = np.array(circles)
            np.savetxt(os.path.join(folder, os.path.basename(file) + ".txt"), circles)
        except:
            print('\t', file, "failed")



if __name__ == "__main__":
    #build_guesses_for(260,8)
    #build_guesses_for(270,11)
    #build_guesses_for(350, 2)
    #build_guesses_for(270, 12)
    #build_guesses_for(280, 2)
    #build_guesses_for(310, 2)

    build_guesses_for(280,3)
    build_guesses_for(290,2)
    #build_guesses_for(270,12)
    input()

    build_guesses_for_demo()
    input()

    get_data_all_beads()
    predictor = get_predictor(113) # to change according to the test

    custom_predict(predictor, r'H:\Data_1\113\310_extract\velocity_310_iteration_004_archive\raw_images\img_00000.png', True)
    input()

    dataset_metadata = MetadataCatalog.get(ML_Conf.dataset_name + "_train")
    dataset_dicts = ML_Conf.get_dataset_dicts3("val")

    test_performance(predictor.cfg)
    #custom_predict(predictor, dataset_metadata=dataset_metadata)

    # for d in random.sample(dataset_dicts, 3):
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(
    #         im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    #     print(outputs)
    #     v = Visualizer(im[:, :, ::-1],
    #                    metadata=dataset_metadata,
    #                    scale=1.2,
    #                    instance_mode=ColorMode.SEGMENTATION
    #                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #                    )
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imshow('test', out.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    # test_performance()