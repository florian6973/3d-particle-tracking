import os
import shutil
import zipfile

import numpy as np
from PIL import Image

from img_processing.hexagon import frameAnalysis
from multiprocessing import Pool

# https://github.com/soumyaiitkgp/Custom_MaskRCNN

hexagon_settings = {}
hexagon_settings['blurval'] = 17
hexagon_settings['threshA'] = 10
hexagon_settings['threshB'] = 255
hexagon_settings['epsilon'] = 0.05
hexagon_settings['res'] = 1
hexagon_settings['sizeOfhexagonArmInPx'] = 326
hexagon_settings['y0'] = 136
hexagon_settings['x0'] = 165
hexagon_settings['h'] = 420
hexagon_settings['w'] = 485
hexagon_settings['N'] = 113 # todo update


def extract_frames(args):
    v, it = args
    path = rf"H:\Data_1\113\{v:03}_extract"
    if not os.path.exists(path):
        os.makedirs(path)
    folder = os.path.join(path, f"velocity_{v:03}_iteration_{it:03}_archive/raw_images")
    if v == 300:
        return None # already extracted
    # print(folder)
    # todo mutliprocessing
    if os.path.exists(folder):
        print(f"{folder} exists")
        files = list(os.walk(folder))[0]  # or glob.glob
        folder = files[0]
        counter = 0
        for file in files[2]:
            if file.endswith(".npz"):
                full_path = os.path.join(folder, file)
                print(f"\tExtracting {full_path}")
                data = np.load(full_path)['arr']
                for img in data:
                    if counter % 250 == 0:
                        print(f"\t\r{v} - {it:03} - {counter:05} images extracted", end="")
                    output = os.path.join(folder, f'img_{counter:05}.png')
                    try:
                        frameAnalysis(img, hexagon_settings, output)
                    except:
                        print(f"\t\tERROR for {file} - {v} - {it:03} - {counter:05}")
                    counter += 1
            print()  # newline
    print(f"END {v} - {it:03} extracted")

if __name__ == '__main__':
    #its = [i for i in range(0, 20)]
    its = [i for i in range(1, 20)]
    #vs = [v for v in range(240, 350, 10)]
    vs = [310]#[250,260,270]#[330, 340, 350]
    with Pool(processes=4) as pool:
        pool.map(extract_frames, [(v, it) for v in vs for it in its])
        pool.close()
    print("Done")