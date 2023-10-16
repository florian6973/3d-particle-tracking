# import matlab
# import matlab.engine as ml  # C:\Program Files\MATLAB\R2020b\extern\engines\python> admin python setup.py install
import logging
import joblib

class Params:
    def __init__(self):
        logging.basicConfig(filename='actions.log', level=logging.DEBUG,
                            format='%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s')
        # TailViewer to visualize log file

        logging.debug("Starting Shaking-Beads")

        self.fullscreen = True
        self.help = False
        self.overlay = True
        self.only_bright = False
        self.only_dark = False
        self.only_centers = True
        self.matlab = False  # True
        self.optim = None
        self.is_optimizing = False
        self.full_circle = False
        self.folder_save_auto = 'counts_auto_ml'

        self.is_boiling = False

        if self.matlab:
            print("Starting Matlab, please wait...")
            self.eng = [ml.start_matlab()]#, ml.start_matlab(), ml.start_matlab(), ml.start_matlab()]
            logging.debug("Matlab started")

        self.ml_detect = True

        self.sck_reg = joblib.load("ml/histogram.pkl")
        print('Model loaded')






