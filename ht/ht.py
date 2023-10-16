import cv2
import cv2 as cv
import optuna
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from skimage import img_as_ubyte, color
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

# import matlab

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def get_bottom_beads_hough(img: str,
                           dp: float = 1.5,
                           min_dist: int = 17,
                           param1 = 100,
                           param2 = 1e-6,
                           minR=14,
                           maxR=25) -> list:
    raw_img = cv2.imread(img)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    #raw_img = cv2.addWeighted(raw_img, 4, raw_img, 0, 4)
    raw_img = unsharp_mask(raw_img, sigma=1., amount=10.0, threshold=0)
    #raw_img = cv.GaussianBlur(raw_img, (1, 1), 0)
    #raw_img = cv.medianBlur(raw_img, 5)

    circles = cv2.HoughCircles(raw_img, cv2.HOUGH_GRADIENT_ALT, dp, min_dist, param1=param1, param2=param2,
                                    minRadius=minR,
                                    maxRadius=maxR)
    if circles is None:
        circles = np.array([[]])
    return circles


def get_bottom_beads_hough2(img, s=0.1, lt=3, ht=10):
    if lt > ht:
        return np.array([[]])
    img = skimage.io.imread(img)
    image = img_as_ubyte(img)
    edges = canny(image, sigma=s, low_threshold=lt, high_threshold=ht)

    # Detect two radii
    hough_radii = np.arange(15, 25, 10)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=113,
                                               min_xdistance=23,
                                               min_ydistance=23)
                                               #threshold=0.5*max(hough_res))
    # centers = zip(cy, cx, radii)
    # centers_f = []
    # for center_y, center_x, radius in centers:
    #     adding = True
    #     for center_y2, center_x2, radius2 in centers_f:
    #         if np.sqrt((center_y - center_y2) ** 2 + (center_x - center_x2) ** 2) < radius + radius2:
    #             adding = False
    #             break
    #     if adding:
    #         centers_f.append((center_y, center_x, radius))
    #centers_f = zip(cy, cx, radii)
    centers_f = zip(cx, cy, radii)
    return np.array(list(centers_f))

    # Draw them
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
#     image = color.gray2rgb(image)
#     for center_y, center_x, radius in centers_f:
#         circy, circx = circle_perimeter(center_y, center_x, radius,
#                                         shape=image.shape)
#         image[circy, circx] = (220, 20, 20)
#
#     ax.imshow(image, cmap=plt.cm.gray)
#     plt.show()
# #
# https://scikit-image.org/docs/stable/auto_examples/edges/plot_circular_elliptical_hough_transform.html

def objective(img, trial):
    p1 = trial.suggest_float('p1', 1e-15, 1e12, log=True)
    p2 = trial.suggest_float('p2', 1e-15, 1e12, log=True)
    dp = trial.suggest_float('dp', 1e-15, 1e12, log=True)
    # p1 = trial.suggest_float('p1', 100, 5000)
    # p2 = trial.suggest_float('p2', 0.001, 1, log=True)
    # dp = trial.suggest_float('dp', 1., 5.)
    md = trial.suggest_int('md', 25, 40) # 25,40
    minR = trial.suggest_int('minR', 14, 17)
    maxR = trial.suggest_int('maxR', 18, 22)
    res = get_bottom_beads_hough(img, dp, md, p1, p2, minR, maxR).shape[1]
    print(res)
    return np.abs(res - 113)


def objective2(img, trial):
    s = trial.suggest_float('s', 1e-2, 1e2, log=False)
    lt = trial.suggest_float('lt', 1e-2, 1e2, log=False)
    ht = trial.suggest_float('ht', 1e-2, 1e2, log=False)
    res = get_bottom_beads_hough2(img, s, lt, ht).shape[0]
    print(res)
    return np.abs(res - 113)

def optimize(img) -> list:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0)) # seed=0
    study.optimize(lambda trial: objective(img, trial), n_trials=500)
    print("end", study.best_params)
    return study.best_params


def convert(row):
    x, y, r = row
    x2 = x * np.cos(np.pi / 2) - y * np.sin(np.pi / 2)
    y2 = x * np.sin(np.pi / 2) + y * np.sin(np.pi / 2)
    return x2, y2, r

# end {'p1': 5.455347684689475e-07, 'p2': 1.679706738521841e-05, 'dp': 2.998540473245123, 'md': 19}

if __name__ == '__main__':
    # {'p1': 7.269926566511257e-07, 'p2': 1.4023274386209231e-06, 'dp': 2.4517921606252218, 'md': 19}
    # end {'p1': 0.003310930363173787, 'p2': 3.0118795724144425e-08, 'dp': 4.686903114605374, 'md': 16}
    path = r'H:\Data_1\113\310_extract\velocity_310_iteration_006_archive\raw_images\img_00105.png'
    path = r'H:\Data_1\113\310_extract\velocity_310_iteration_006_archive\raw_images\img_00054.png'
    path = r'H:\Data_1\113\290_extract\velocity_290_iteration_001_archive\raw_images\img_00100.png'
    path = r'H:\Data_1\113\290_extract\velocity_290_iteration_001_archive\raw_images\img_00200.png'
    #get_bottom_beads_hough2(skimage.io.imread(path))
    params = optimize(path)
    # params = {}
    # params['p1'] = 1
    # params['p2'] = 0.0009
    # params['dp'] = 4
    # params['md'] = 15
    # params['minR'] = 16
    # params['maxR'] = 19
    circles = get_bottom_beads_hough(path, params['dp'], params['md'], params['p1'], params['p2'],
                                     params['minR'],
                                     params['maxR'])[0]
    # path2 = 'temp.png'
    # img2 = cv2.imread(path)
    # img2= cv2.rotate(img2, cv2.cv2.ROTATE_90_CLOCKWISE)
    # cv2.imwrite(path2, img2)
    # circles2 = get_bottom_beads_hough(path2, params['dp'], params['md'], params['p1'], params['p2'])[0]
    # circles3 = []
    # for circle in circles2:
    #     circles3.append(convert(circle))
    # circles2 = np.array(circles3)
    #
    # tot = np.concatenate((circles, circles2))
    # print(tot.shape)
    # centers_f = []
    # for center_y, center_x, radius in tot:
    #     adding = True
    #     for center_y2, center_x2, radius2 in centers_f:
    #         if np.sqrt((center_y - center_y2) ** 2 + (center_x - center_x2) ** 2) < 20: # radius + radius2:
    #             adding = False
    #             break
    #     if adding:
    #         centers_f.append((center_y, center_x, radius))
    # circles = np.array(centers_f)

    #circles = get_bottom_beads_hough2(path, params['s'], params['lt'], params['ht'])

    print(circles)
    print(circles.shape)

    cv2.setWindowTitle('image', "test")

    i = 200
    while True:
        output = cv2.imread(path)
        for k, (x, y, r) in enumerate(circles):
            x = int(x)
            y = int(y)
            r = int(r)
            #cv2.rectangle(output, (x - 1, y - 1), (x + 1, y + 1), (255, 128, 0), -1)# draw the outer circle
            cv.circle(output,(x,y),r,(0,255,0),1)
            # draw the center of the circle
            cv.circle(output,(x,y),r,(0,0,255),1)

        cv2.imshow('image', output)

        key = cv2.waitKey(0)
        i+=1
        path = fr'H:\Data_1\113\290_extract\velocity_290_iteration_001_archive\raw_images\img_{i:05}.png'
        params = optimize(path)
        circles = get_bottom_beads_hough(path, params['dp'], params['md'], params['p1'], params['p2'],
                                         params['minR'],
                                         params['maxR'])[0]
        print(circles.shape[0])