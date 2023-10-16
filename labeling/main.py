import numpy as np

from Data.DL_Predict import custom_predict, get_predictor
from Data.Models import Session, Estimate
from window import Window
from params import Params
from optimizer import Optimizer
from navigator import Navigator
from image import Image
import logging
import traceback
import time
from multiprocessing import Pool
import multiprocessing as mp

# python -m cProfile -o out.profile main.py
# https://jiffyclub.github.io/snakeviz/

def high_speed_count(args):
    files, n, s, it = args
    try:
        if files.select(n, s, it, True):
            nb_tops = {}
            print("Begin prediction")
            for nb_i, img in enumerate(files.get_imgs()):
                print("\rPred", nb_i, end='')
                centers, classes = custom_predict(get_predictor(n), img, False)
                nb_tops[nb_i] = np.array(classes).sum()
            print("End of prediction")
            session = Session()

            print("Begin saving")
            for frame_n, nb_top in nb_tops.items():
                fid = files.ids[frame_n]
                try:
                    q = session.query(Estimate). \
                        filter((Estimate.frame_id == fid) & (Estimate.estimate_name == 'nb_top_ml1'))
                    q.delete(synchronize_session=False)
                except Exception as e:
                    print(e)
                    print("\t\tNo beads to delete")

                estim = Estimate(frame_id=fid, estimate_name='nb_top_ml1', estimate_value_int=nb_top)
                session.add(estim)
            print("End of Saving")

            session.commit()
            session.close()
            print("DB Closed")
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print(f"Error in compute_experiment {n:03}, {s:03}, {it:03}" + str(e))

def compute_experiment(n, s, it, files, params, all = False):
    #print(mp.current_process().pid)
    try:
        if files.select(n, s, it, all):  # wrong parameters, graphs look for everything
            print("Computing...")
            for nb_i, img in enumerate(files.get_imgs()):  # override
                img = Image(img, params)
                img.counter.try_count() # True for Histogram
                print(img.counter.loaded_or_saved)
                if not img.counter.loaded_or_saved:
                    print("no data found")
                img.counter.save_auto()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print(f"Error in compute_experiment {n:03}, {s:03}, {it:03}" + str(e))


if __name__ == "__main__":
    try:
        #folder_input = "imgs"
        #imgs = [os.path.join(folder_input, f) for f in os.listdir(folder_input) if os.path.isfile(os.path.join(folder_input, f))]
        params = Params()
        #files = Navigator(params)
        #files.select()
        #files.check_boiling()

        # optim = Optimizer(files.get_imgs(), params)
        # try:
        #     optim.load_params()
        #     params.optim = optim
        # except:
        #     print("Default params")

        print("Press d to display, o to optimize, a to auto-compute")
        #r = input()
        r = "d"
        print(r)

        if r in "d":
            exit_val = False
            #imgs = files.get_imgs()
            imgs = [
                r'H:\Data_1\113\290_extract\velocity_290_iteration_004_archive\raw_images/img_37537.png',
                r'H:\Data_2\n_beads_113_Delrin_white_0.25_Vmin_210_Vmax_350_Step_10_Cooling_140_humidity_55_65\velocity_330\iteration_001\countDump/Dump_119.png',
                r'H:\Data_2\n_beads_113_Delrin_white_0.25_Vmin_210_Vmax_350_Step_10_Cooling_140_humidity_55_65\velocity_260\iteration_000\countDump/Dump_46.png',
                'H:\\Cluster\\raw_data/1_113_260_000_39799.png',
                'H:\\Cluster\\raw_data/1_113_280_002_39799.png',
                'H:\\Cluster\\raw_data/1_113_280_002_39999.png',
                'H:\\Cluster\\raw_data/1_113_280_002_40199.png',
               # r'H:\Data_1\113\280_extract\velocity_280_iteration_013_archive\raw_images\img_14236.png',
                #r'H:\Data_1\113\280_extract\velocity_280_iteration_013_archive\raw_images\img_18084.png',
                #r'H:\Data_1\113\280_extract\velocity_280_iteration_013_archive\raw_images\img_26471.png',
            ]
            params.n = 113
            params.s = 280
            params.i = 13
            params.dt = 1
            print(imgs)
            for nb_i, img in enumerate(imgs):
                if nb_i < 0:
                    continue
                if exit_val:
                    break
                print(f"Showing {nb_i+1}/{len(imgs)}", img)
                exit_val = Window(img, imgs, params).show()#files.all_imgs, params).show()
        # elif r == "o":
        #     optim.find_params()
        #     optim.save_params()
        elif r == "a": # faire multiprocessing
            begint = time.time()
            #with Pool(mp.cpu_count()) as p:
            for n in [90,94,99,104,107,113]:#[90,94,99,100,103,104,105,107,109,110,111,112,113,114]:
                for s in range(240,260,10):
                    for it in range(0,20,4):
                        compute_experiment(n, s, it, files, params)
                            # try if exist

                            #p.apply_async(compute_experiment, (n, s, it, files))
                #p.close()
                #p.join()
            endt = time.time()
            print(endt-begint)
        elif r == "a2":
            begint = time.time()
            #compute_experiment(99, 250, 6, files, params, True)

            with Pool(mp.cpu_count()) as p:
                r = p.map(high_speed_count, [(files, 99, 250, 1),
                                             (files, 99, 240, 0),
                                             (files, 113, 320, 0)])
                res = [r[0], r[1], r[2]]
            print("fin")
                #high_speed_count(99, 250, 1)

            # with Pool(mp.cpu_count()) as p:
            # for n in [90, 94, 99, 104, 107, 113]:  # [90,94,99,100,103,104,105,107,109,110,111,112,113,114]:
            #     for s in range(240, 260, 10):
            #         for it in range(0, 20, 4):
            #             compute_experiment(n, s, it, files, params)
                        # try if exist

                        # p.apply_async(compute_experiment, (n, s, it, files))
                # p.close()
                # p.join()
            endt = time.time()
            print(endt - begint)
    except Exception as e:
        err = str(e) + "\n" + str(traceback.format_exc())
        logging.error(err)
        print(err)

# optimize parameters, other implemention of hough transfomr, machine learning
# todo, look for radius in matlab return types

# images à échantillonner

# change easily type particles
# msg adel : matlab dans code, opencv essayé, slides et precision avec function matlab
# tune parameters on many examples, make group per experiment, see if it is worth it
# refactor code
# save data automatically
# try machine learning

# check total right number for particles