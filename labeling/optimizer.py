import os.path

# import optuna
from image import Image
import json
from joblib import parallel_backend
import logging
# import neptune.new as neptune
# import neptune.new.integrations.optuna as optuna_utils
import time

# # Neptune doesn't work
# from optuna.visualization import plot_contour
# from optuna.visualization import plot_edf
# from optuna.visualization import plot_intermediate_values
# from optuna.visualization import plot_optimization_history
# from optuna.visualization import plot_parallel_coordinate
# from optuna.visualization import plot_param_importances
# from optuna.visualization import plot_slice


class Optimizer:
    def __init__(self, imgs, params):
        self.imgs = imgs
        self.params = params
        self.best_params = {}
        self.file_def = "params/optim_gas_def.json"
        self.name_file()
        self.n_trials = 200 #200

        #self.run = neptune.init(api_token='ANONYMOUS',
        #                  project='common/optuna-integration') # your credentials
        #self.neptune_callback = optuna_utils.NeptuneCallback(self.run)

    def loss_nomatlab(self, param1, param2, dp, minDist, sizeaverage, threshold):
        tot = 0.
        for name_img in self.imgs:  # todo vectorization performance
            try:
                img = Image(name_img, self.params)
                img.counter.try_count()
                if img.counter.loaded_or_saved:
                    o_b, o_d = img.counter.nb_bright(), img.counter.nb_dark()
                    img.counter.count_beads(dp=dp, minDist=minDist, param1=param1, param2=param2, sizeaverage=sizeaverage, threshold=threshold, optim=True)
                    n_b, n_d = img.counter.nb_bright(), img.counter.nb_dark()
                    tot += (n_b - o_b) ** 2 + (n_d - o_d) ** 2
            except Exception as e:
                print("erreur", e)
                raise e
        print(tot)
        return tot

    def objective_nomatlab(self, trial):
        param1 = trial.suggest_int('param1', 1, 1000)
        param2 = trial.suggest_float('param2', 0., 1.5)
        minDist = trial.suggest_int('minDist', 1, 10)
        dp = trial.suggest_float('dp', 0.01, 2.)
        sizeaverage = trial.suggest_int('sizeaverage', 1,20)
        threshold = trial.suggest_int('threshold', 0, 255)
        return self.loss_nomatlab(param1, param2, dp, minDist, sizeaverage, threshold)

    def loss_matlab(self, tb, sb, td, sd, k, type):
        tot = 0.
        for name_img in self.imgs:  # todo vectorization performance
            try:
                self.params.is_optimizing = True
                img = Image(name_img, self.params)
                img.counter.try_count()
                if img.counter.loaded_or_saved:
                    o_b, o_d = img.counter.nb_bright(), img.counter.nb_dark()
                    n_b, n_d = img.counter.count_beads(sb=sb, tb=tb, td=td, sd=sd, optim=True, eng=k, type=type)
                    tot += (n_b - o_b) ** 2 + (n_d - o_d) ** 2
            except Exception as e:
                print("erreur", e)
                raise e
        print(tot)
        return tot

    def objective_matlab_bright(self, trial):
        k_tot = len(self.params.eng)
        tb = trial.suggest_float('tb', 0., 1.)
        sb = trial.suggest_float('sb', 0., 1.)
        return self.loss_matlab(tb, sb, -1, -1, trial.number%k_tot, 1)

    def objective_matlab_dark(self, trial):
        k_tot = len(self.params.eng)
        td = trial.suggest_float('td', 0., 1.)
        sd = trial.suggest_float('sd', 0., 1.)
        return self.loss_matlab(-1, -1, td, sd, trial.number%k_tot, 0)

    def name_file(self):
        self.file = f"params/optim_gas_{self.params.n}_{self.params.s}_{self.params.i}.json"

    def find_params(self):
        self.name_file() # before load and save only

        storage = "sqlite:///example.db"
        study = optuna.create_study(direction="minimize",
                                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        begint = time.time()
        with parallel_backend('multiprocessing'):
            if self.params.matlab:
                study.optimize(self.objective_matlab_bright, n_trials=self.n_trials, n_jobs=12) # load one matlab instance per jobs?
                self.best_params = study.best_params
                study.optimize(self.objective_matlab_dark, n_trials=self.n_trials, n_jobs=12) # load one matlab instance per jobs?
            else:
                study.optimize(self.objective_nomatlab, n_trials=self.n_trials, n_jobs=12, callbacks=[self.neptune_callback])
        endint = time.time()
        print("Temps", endint - begint)
        print(study.best_params)
        self.best_params = dict(self.best_params, **study.best_params)
        print(self.best_params)
        #plot_optimization_history(study)
        #plot_contour(study)
        ## not seen?

    def save_params(self):
        with open(self.file, "w") as outfile:
            logging.info(f"Saving {self.best_params}") # {'t': 0.013381938062835774, 's': 0.9310735302772035}
            json.dump(self.best_params, outfile)
        print("Parameters saved")

    def load_params(self):
        file = self.file
        if not os.path.isfile(file):
            file = self.file_def
        with open(file) as json_file:
            self.best_params = json.load(json_file)
        print("Parameters loaded")

