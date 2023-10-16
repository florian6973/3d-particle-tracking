import os
import glob
import logging
import natsort
import numpy as np

from Data.Models import Session, Experiment, Frame
from analysis.data_low_fps import infer_cnn, init_model


class Navigator:
    def __init__(self, params):
        self.params = params
        self.nb = 5 #2 #5 # 20

        self.imgs = []

    def select(self, n=-1, s=-1, i=-1, all=True, dt=2): #, s=250, n=90e, i=13):
        print("Select n_beads, speed, iteration")
        #n, s, i = map(int, input().split(','))
        if n == -1 or s == -1 or i == -1:
            dt, n, s, i = 1,113,260,2#1,113,320,2#1,113,310,15#2,110,270,1#1,113,270,0#1,113,270,12#1,113,280,2#310,2#350,2#107,250, 1#107, 290, 11 #113,320,2#107, 290, 9
            #dt, n, s, i = 1,113,260,8#1,113,280,2#310,2#350,2#107,250, 1#107, 290, 11 #113,320,2#107, 290, 9
            self.nb = 2#90, 280, 4#114, 350, 1#114, 350, 1
        self.params.dt, self.params.s, self.params.n, self.params.i = dt, s, n, i
        print(f"{n}, {s}, {i}")
        logging.debug(f"{n}, {s}, {i} configuration")

        all = True


        filesauv= f'imgs_ids_{dt}_{n}_{s}_{i}.npz'

        if False:#if os.path.exists(filesauv):
            self.folder = np.load(filesauv)['folder']
            self.imgs = np.load(filesauv)['imgs']
            self.all_imgs = self.imgs
            if dt == 2:
                self.ids = np.load(filesauv)['ids']
        else:
            if dt==2:
                # request get img path
                session = Session()
                res = session.query(Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
                                    Experiment.exp_folder,
                                    Frame.frame_number, Frame.frame_img_path, Frame.id) \
                    .join(Frame, Experiment.id == Frame.experiment_id) \
                    .where((Experiment.exp_type == 2)
                           & (Experiment.exp_nbeads == n)
                           & (Experiment.exp_velocity == s)
                           & (Experiment.exp_iteration == i)
                           & (Frame.frame_img_path != None)) \
                    .order_by(Frame.frame_number.desc())
                if not all:
                    res = res.limit(self.nb)
                    logging.info(f"Using {self.nb} last")
                print(res)
                res = res.all()
                self.folder = res[0][3]
                #print(res)
                self.imgs = list(map(lambda x : os.path.join(self.folder, 'countDump', x[5] if x[5] is not None else "Dump_" + str(x[4]) + '.png'), res))
                self.ids = dict(list(map(lambda x : (x[4], x[6]), res)))
                self.all_imgs = self.imgs
                print(self.imgs)
                    #imgsbv.append(cv2.imread(img))
            else:
                session = Session()
                res = session.query(Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
                                    Experiment.exp_folder) \
                    .where((Experiment.exp_type == 1)
                           & (Experiment.exp_nbeads == n)
                           & (Experiment.exp_velocity == s)
                           & (Experiment.exp_iteration == i))

                step = 20
                inter = 200
                max_nb = step*inter
                # if not all:
                #     res = res.limit(max_nb)
                #     logging.info(f"Using {self.nb} last")
                self.folder = res.all()[0][3]
                res = list(reversed(sorted(glob.glob(os.path.join(self.folder, '*.png')))))
                self.all_imgs = res.copy()

                #print(res)
                # idx = sorted(list(set(list(range(0, step)) + list(range(0, max_nb, inter)) + \
                #       list(range(0, max_nb*10, 10*inter)) + \
                #                       list(range(0, max_nb//10, inter//10))))) #*4

                if not all:
                    idx = sorted(list(set(list(range(0, step)) + list(range(0, max_nb, inter)))))
                else:
                    idx = sorted(list(set(list(list(range(0, len(res), inter))))))

                idx = list(range(0, 1000, 50))

                #2000)))))

                print(idx)
                #input()
                res = np.array(res)[idx]
                print(res)
                #input()
                #print(res)
                self.imgs = res
                #self.ids = dict(list(map(lambda x : (x[4], x[6]), res)))

                print(self.imgs)
                pass


            if False:
                print('analysing')
                imgsbv = []
                idsbv = []
                init_model()
                for i0, img in enumerate(self.imgs):
                    print(f"\r{i0}/{len(self.imgs)}", end='')
                    print(img)
                    nt = infer_cnn(img)
                    if 0 < nt < 10:
                        imgsbv.append(img)
                        if dt == 2:
                            idsbv.append(self.ids[i0])

                self.imgs = imgsbv
                if dt == 2:
                    self.ids = idsbv

            # save in the same file imgs and ids as a numpy array
            #np.savez(filesauv, imgs=imgsbv, ids=idsbv, folder=self.folder)

        return True

        # subpath = f"n_beads_{n:03}_Delrin_white_0.25_Vmin_210_Vmax_350_Step_10_Cooling_140_humidity_55_65\\velocity_{s:03}\\iteration_{i:03}\\countDump"
        # print("f for first and l for last")
        # #c = input()
        # c= "l"
        # print(c)
        # full_path = os.path.join(self.folder, subpath, "*.png")
        # files = glob.glob(full_path)
        # if len(files) > 0:
        #     files = natsort.natsorted(files)
        #     if c == "f":
        #         files = files[:self.nb]
        #         logging.info(f"Using {self.nb} first " + subpath)
        #     elif c == "l":
        #         files = files[-self.nb:]
        #         logging.info(f"Using {self.nb} last " + subpath)
        #     self.imgs = files
        #     return True
        # else:
        #     return False

        # robust data write

    @staticmethod
    def get_percentage(n, s, i = 0):
        percentage = 0
        file = fr"G:\2020_10_06\General_output\n_beads_{n:03}_experiment_boiling_time.txt"
        #print("Reading", file)
        data = np.loadtxt(file, delimiter='\t')
        same_speed = data[np.where((data[:, 0] == s))]
        #print(data)
        bt = data[np.where((data[:, 0] == s) & (data[:, 1] == i))][0][2]
        nb_bt = 0
        nb_nbt = 0
        for exp in same_speed:
            if exp[2] < 0.:
                nb_nbt += 1
            else:
                nb_bt += 1
        percentage = nb_nbt/(nb_nbt+nb_bt)*100
        return percentage, bt

    def check_boiling(self): # todo update with databqse
        p, bt = Navigator.get_percentage(self.params.n, self.params.s, self.params.i)
        print(f"{p:.2f}% proportion of non-boiling experiments with the same parameters")
        if bt < 0:
            print("No boiling for this experiment")
            self.params.is_boiling = False
        else:
            print("Boiling for this experiment at", bt)
            self.params.is_boiling = True

    def get_imgs(self):
        return self.imgs