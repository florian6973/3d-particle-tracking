import numpy as np
import cv2
import os
from PIL import Image
import time
import logging
# import matlab
# import matlab.engine as ml  # C:\Program Files\MATLAB\R2020b\extern\engines\python> admin python setup.py install

import ml.histo

import sys

from Data.Experiment import get_frame_nid

if sys.version.startswith('3.10'):
    from Data.DL_Predict import custom_predict, get_predictor

from Data.Models import Session, Experiment, Frame, Estimate, Bead

predictorcurr = None

class Counter:
    def __init__(self, name, folder, raw_img, params, folder_auto):
        self.loaded_or_saved = False
        self.circles = None

        self.params = params
        self.p = "default"

        self.name = name
        self.folder = folder
        self.folder_auto = folder_auto
        self.raw_img = raw_img

        #print("\t\tInitializing counter")

    def try_count(self, only_bright=False):
        #print("\t\tTrying to count", self.name)

        if not only_bright:
            if True:#not self.load_count():
                self.count_beads()
        else:
            self.circles = np.empty((self.params.n, 4))
            nb_bright = int(round(self.params.sck_reg.predict([ml.cnn.get_features((self.params.n, self.params.s), self.name)])[0]))
            for i in range(nb_bright):
                self.circles[i] = [-1,-1,17,1]
            for i in range(self.params.n-nb_bright):
                self.circles[i+nb_bright] = [-1,-1,17,0]
            self.loaded_or_saved = False

    def nb_dark(self):
        if self.circles.size != 0:
            return np.count_nonzero(self.circles[:, 3]==0)
        else:
            return 0

    def nb_bright(self):
        return self.circles.shape[0] - self.nb_dark()  # avoid ==1

    def load_count(self):
        # res = session.query(Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
        #                     Frame.frame_number, Frame.frame_nbtop, Estimate.name, Estimate.estimate_value_int) \
        #     .join(Experiment, Experiment.id == Frame.experiment_id) \
        #     .join(Estimate, Estimate.frame_id == Frame.id) \
        #     .where((Experiment.exp_type == 2)
        #            & (Experiment.exp_nbeads == self.params.n)
        #            & (Experiment.exp_velocity == self.params.s)
        #            & (Experiment.exp_iteration == self.params.i)
        #            & (Estimate.estimate_name == 'nb_top')) \
        #     .order_by(Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
        #               Frame.frame_number)
        session = Session()
        res, _ = self.get_beads(session)
        res = res.order_by(Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
                      Frame.frame_number)
        print(res)
        res = res.all() # do not exist yet
        self.circles=np.ones((len(res), 4))
        print('len(res)',len(res))
        if len(res) == 0:
            session.close()
            return False
        for k, (n, v, i, fn, ft, x, y, r, top) in enumerate(res):
            if k >= self.params.n:
                print("\t\t\tWarning: more beads than expected")
                break
            self.circles[k,:] = ([x, y, r, int(top)])
        self.circles= self.circles.astype(np.int32)
        print(self.circles)
        print(res)

        #self.circles = np.loadtxt(self.get_savefile_name(), dtype=int)
        self.loaded_or_saved = True
        print("\t\tLoaded", self.name)
        session.close()
        return True

    def get_beads(self, session):
        nb = int(os.path.basename(self.name).split('_')[1].replace('.png', ''))
        res = session.query(Experiment.exp_nbeads, Experiment.exp_velocity, Experiment.exp_iteration,
                            Frame.frame_number, Frame.frame_nbtop,
                            Bead.bead_x, Bead.bead_y, Bead.bead_r, Bead.bead_top) \
            .join(Frame, Experiment.id == Frame.experiment_id) \
            .join(Bead, Bead.frame_id == Frame.id) \
            .where((Experiment.exp_type == self.params.dt)
                   & (Experiment.exp_nbeads == self.params.n)
                   & (Experiment.exp_velocity == self.params.s)
                   & (Experiment.exp_iteration == self.params.i)
                   & (Frame.frame_number == nb))
              # for all frames
        return res, nb

    def save_count(self, auto = False):
        if self.params.optim is not None and self.params.optim.best_params is not None:
            p = str(self.params.optim.best_params)

        np.savetxt(self.name.replace('raw_images', '') + '.txt', np.delete(self.circles, 2, 1), fmt='%d')

        # clean db
        session = Session()
        nb = int(os.path.basename(self.name).split('_')[1].replace('.png', ''))

        fid = None
        fid = session.query(Frame) \
            .join(Experiment, Experiment.id == Frame.experiment_id) \
            .where((Experiment.exp_type == self.params.dt)
                   & (Experiment.exp_nbeads == self.params.n)
                   & (Experiment.exp_velocity == self.params.s)
                   & (Experiment.exp_iteration == self.params.i)
                   & (Frame.frame_number == nb))
        print('dt', self.params.dt)
        if self.params.dt == 2 or fid.first() is not None:
            fid = fid.first() # unique should be
            print('getting new fid')
        else:
            # get exp
            exp = session.query(Experiment) \
                .where((Experiment.exp_type == self.params.dt)
                       & (Experiment.exp_nbeads == self.params.n)
                       & (Experiment.exp_velocity == self.params.s)
                       & (Experiment.exp_iteration == self.params.i)).first()
            expid = exp.id
            print("ajout", expid, nb, self.name, get_frame_nid(exp, nb))
            fid = Frame(experiment_id=expid, frame_number=nb, frame_img_path=self.name,
                       frame_nid=get_frame_nid(exp, nb))
            session.add(fid)
            session.commit()


        if not auto:
            fid.frame_nbtop = self.nb_bright()
            fid = fid.id
            session.commit()

            try:
                q = session.query(Bead). \
                    filter(Bead.frame_id == fid)
                q.delete(synchronize_session=False)
                session.commit()
            except Exception as e:
                print(e)
                print("\t\tNo beads to delete")

            # add new pos
            print(fid)
            for k, (x, y, r, top) in enumerate(self.circles):
                if top == 1:
                    top = True
                else:
                    top = False
                bead = Bead(frame_id=fid, bead_x=x, bead_y=y, bead_r=r, bead_top=top)
                session.add(bead)
        else:
            fid = fid.id
            try:
                q = session.query(Estimate). \
                    filter((Estimate.frame_id == fid) & (Estimate.estimate_name == 'nb_top_ml1'))
                q.delete(synchronize_session=False)
                session.commit()
            except Exception as e:
                print(e)
                print("\t\tNo beads to delete")

            estim = Estimate(frame_id=fid, estimate_name='nb_top_ml1', estimate_value_int=self.nb_bright())
            session.add(estim)

        session.commit()
        session.close()

        #np.savetxt(self.get_savefile_name(auto), self.circles, fmt='%d', header=f"{self.params.matlab}_{self.p}")
        logging.info("Saving " + self.name + " ; auto " + str(auto))
        self.loaded_or_saved = True
        print(f"\t\tSaved auto: {auto}", self.name)

    def save_auto(self):
        self.save_count(True)

    # optim redundant with self.params.is_optimizing
    def count_beads(self, dp=1.28, minDist=10,param1=87, param2=0.00634, sizeaverage=1, threshold=170, tb=0.06, sb=0.95, td=0.06, sd=0.95, optim=False, eng=0, type=-1): # avoid default params
        global predictorcurr
        if self.params.ml_detect:
            self.circles = []
            if predictorcurr is None:
                predictorcurr = get_predictor(self.params.n)
            centers, classes = custom_predict(predictorcurr, self.name, False)
            print(self.name, self.params.n)
            for e in range(len(centers)):
                self.circles.append([centers[e][0], centers[e][1], 17, int(classes[e])])
            self.circles = np.array(self.circles)
            print('circles', self.circles)
            return

        if not optim:
            if self.params.optim is not None:
                try:
                    dp = self.params.optim.best_params["dp"]
                    minDist = self.params.optim.best_params["minDist"]
                    param1 = self.params.optim.best_params["param1"]
                    param2 = self.params.optim.best_params["param2"]
                    sizeaverage = self.params.optim.best_params["sizeaverage"]
                    threshold = self.params.optim.best_params["threshold"]
                except:
                    sb = self.params.optim.best_params["sb"]
                    tb = self.params.optim.best_params["tb"]
                    sd = self.params.optim.best_params["sd"]
                    td = self.params.optim.best_params["td"]
                print("Using", self.params.optim.best_params)

        def call_matlab(k):
            def adding(type, lst, lst_r):
                for i, (x, y) in enumerate(lst):
                    self.circles.append([x, y, float(lst_r[i]), type])
            #print("Calling Matlab")
            beg = time.time()
            center_forground, rad_forground, center_background, rad_background = self.params.eng[k].count_circles_opti(self.name, sb, tb, sd, td, type, nargout=4)

            rad_forground = np.asarray(rad_forground).flatten()
            rad_background = np.asarray(rad_background).flatten()
            end = time.time()
            if not optim:
                self.circles = []
                adding(0, center_background, rad_background)
                adding(1, center_forground, rad_forground)
                self.circles = np.array(self.circles, dtype=np.float)
                self.circles = np.round(self.circles).astype("int")
            else:
                return len(rad_forground), len(rad_background)
            #print((end - beg))

        if self.params.matlab:# You may need to convert the color.
            if not optim:
                call_matlab(eng)
            else:
                return call_matlab(eng)
        else:
            self.circles = cv2.HoughCircles(self.raw_img, cv2.HOUGH_GRADIENT_ALT, dp, minDist, param1=param1, param2=param2,
                                            minRadius=13,
                                            maxRadius=23)  # pip install opencv-contrib-python # 0.4 p2 works fine too

            if self.circles is not None and self.circles != []:
                old_circles = np.round(self.circles[0, :]).astype("int")
                self.circles = []
                for x, y, r in old_circles:
                    self.circles.append([x, y, r, self.classify(x, y, sizeaverage, threshold)])
                self.circles = np.array(self.circles)

        if self.circles is None or self.circles == []:
            self.circles = np.array([])

        self.loaded_or_saved = False
        #return len(rad_forground), len(rad_background)
        pass

    def delete_point(self, mx, my):
        idx = self.get_nearest_point(mx, my)
        old_circles = self.circles
        self.circles = []
        for i in range(len(old_circles)):
            if i != idx:
                self.circles.append(old_circles[i])
        self.circles = np.array(self.circles)

    def switch_point(self, mx, my):
        idx = self.get_nearest_point(mx, my)
        old_circles = self.circles
        self.circles = []
        for i in range(len(old_circles)):
            if i == idx:
                old_circles[i][3] = 1 - old_circles[i][3]
            self.circles.append(old_circles[i])
        self.circles = np.array(self.circles)

    def get_nearest_point(self, mx, my):
        dist = np.inf
        idx = 0
        for i, (x, y, _, _) in enumerate(self.circles):
            d = (mx - x) ** 2 + (my - y) ** 2
            if d < dist:
                dist = d
                idx = i
        return idx

    def add_point(self, mx, my, type):
        self.circles = np.array(self.circles.tolist() + [[mx, my, 17, type]])
        pass

    def get_savefile_name(self, auto = False):
        folder = self.folder
        if auto:
            folder = self.folder_auto
        return os.path.join(folder, f"{self.params.n:03}_{self.params.s:03}_{self.params.i:03}_" + os.path.basename(self.name)) + ".txt"

    def classify(self, x, y, size_avg=1, threshold=170):
        meanv = 0.
        for ki in range(-size_avg, size_avg + 1):
            for kj in range(-size_avg, size_avg + 1):
                try:
                    meanv += float(self.raw_img[y + ki, x + kj].mean())
                except:
                    print("Box overflow", x, y)
                    pass
        meanv = meanv / ((size_avg + 1) ** 2)

        if not threshold >= meanv:
            return 1  # enum to do
        else:
            return 0
