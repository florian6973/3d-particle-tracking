import gc
import os
import pickle

import numpy as np
import pickle5 as pickle


# import matplotlib.pyplot as plt
# import torch
# from detectron2.data import MetadataCatalog
# from detectron2.utils.logger import setup_logger
#
# from Data import ML_Conf
# import numpy as np
#
# from Data.DL_Predict import get_predictor, custom_predict
# from Data.ML_Conf import get_data_all_beads, dataset_name, get_dataset_dicts3
from matplotlib import pyplot as plt


def plot_d2_hough():
    import matlab.engine

    test_file = 'testset.pkl'
    with open(test_file, 'rb') as handle:
        dataout = pickle.load(handle)

    testpred_file = 'testpredH.pkl'
    preds = {}
    if not os.path.exists(testpred_file):
        engine = matlab.engine.start_matlab()
        datapred = {}
        dataorg = {}
        for i, img in enumerate(dataout):
            file = img[0]
            n = int(img[1].split('-')[1]) # right dataset
            nbtop = n - len(engine.count_circles_dark(file))
            print(f'infering {i}/{len(dataout)}', img[0])
            print(f'\t{nbtop} vs {img[2]}')
            datapred[img[0]] = nbtop
            dataorg[img[0]] = img[2]
            # gc.collect() # https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        with open(testpred_file, 'wb') as handle:
            pickle.dump([dataorg, datapred], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(testpred_file, 'rb') as handle:
            dataorg, datapred = pickle.load(handle)


    errs = []
    relerrs = []
    for k, v in datapred.items():
        if dataorg[k] > 10:
            errs.append(abs(v - dataorg[k]))
            if dataorg[k] != 0:
                relerrs.append(abs(v - dataorg[k]) / dataorg[k])

    print(datapred)

    print(errs)
    print(relerrs)
    print(f'average error {np.mean(errs)}')
    print(f'max error {np.max(errs)}')
    print(f'median error {np.median(errs)}')
    print(f'average relative error {np.mean(relerrs)}')
    print(f'max relative error {np.max(relerrs)}')
    print(f'median relative error {np.median(relerrs)}')

    fig, ax = plt.subplots()
    # ax.plot(list(dataorg.values()), list(dataorg.values()), '-+',
    #         label='Ideal predictions', color='black')
    ax.plot(list(dataorg.values()), list(datapred.values()), 'r+',
           label='Hough Transform predictions')
    ax.axvline(x=10, color='green')

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.xlabel('Groundtruth $N_{top}$')
    plt.ylabel('Prediction $N_{top}$')
    plt.title("Evaluation of the performance of Hough Transform for beads counting")
    # plt.title("Evaluation of the performance for beads counting")
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.show()

    # save the first phase because quite long

if __name__ == '__main__':
    print()
    plot_d2_hough()