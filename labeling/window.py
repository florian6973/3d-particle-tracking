import numpy as np
import cv2
import os

from counter import Counter
from image import Image, distance_nearest
import time

class Window:
    def __init__(self, path_img, imgs, params):
        self.image = None

        self.screen_h = 1080  # todo auto set?
        self.screen_w = 1920

        self.params = params

        self.imgs = imgs

        self.create(path_img)

    def mouseRGB(self, event, x, y, flags, param):
        r = 1.
        if self.params.fullscreen:
            r = (float(self.screen_h) / float(self.image.shape()[0]))
        self.image.mx = int(x / r)
        self.image.my = int(y / r)

        if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
            colors = self.image.output[y, x]  # be careful if fullscreen
            print("BRG Format: ", colors)
            print("Coordinates of pixel: X: ", self.image.mx, "Y: ", self.image.my)
        elif event == cv2.EVENT_MOUSEMOVE:
            pass

    def create(self, path):
        self.image = Image(path, self.params)

        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
        cv2.setWindowTitle('image', 'Beads counter - h for help')
        # draw image

    def show(self):
        exit_val = False
        cv2.setWindowTitle('image', 'Beads counter - h for help - ' + os.path.basename(self.image.path) + f"- {self.params.n},{self.params.s},{self.params.i}")

        self.image.counter.try_count()

        begin_time = time.time()
        circles_arch = {}

        while True:
            cv2.setMouseCallback('image', self.mouseRGB)

            self.image.background()
            if not self.params.help and self.params.overlay:
                self.image.draw()
            if self.params.fullscreen:
                ns = (int(float(self.screen_h) / float(self.image.shape()[0]) * float(self.image.shape()[1])), self.screen_h)
                self.image.resize(ns)
                cv2.resizeWindow('image', self.screen_w, self.screen_h)

            if self.params.help:
                self.image.display_help()
            cv2.imshow('image', self.image.output)

            key = cv2.waitKey(1)
            if key == 27 or key == ord("e"):  # escape
                end_time = time.time()
                print(f"{(end_time-begin_time):.2f} s spent on this image")
                begin_time = end_time
                break
            elif key == 32:  # space
                self.image.counter.count_beads()
            elif key == ord("f"):
                self.params.fullscreen = not self.params.fullscreen
            elif key == ord("r"):
                self.image.counter.delete_point(self.image.mx, self.image.my)
            elif key == ord("b"):
                self.image.counter.add_point(self.image.mx, self.image.my, 1)
            elif key == ord("d"):
                self.image.counter.add_point(self.image.mx, self.image.my, 0)
            elif key == ord("g"):
                self.params.overlay = not self.params.overlay
            elif key == ord("w"):
                self.params.only_bright = not self.params.only_bright
            elif key == ord("x"):
                self.params.only_dark = not self.params.only_dark
            elif key == ord("t"):
                self.image.counter.switch_point(self.image.mx, self.image.my)
            elif key == ord("s"):
                self.image.counter.save_count()
            elif key == ord("l"):  # autoload if exist?
                self.image.counter.load_count()
            elif key == ord("h"):
                self.params.help = not self.params.help
            elif key == ord("o"):
                self.params.full_circle = not self.params.full_circle
            elif key == ord("y") or key == ord('6'):
                print(self.image.path)
                idx = self.imgs.index(self.image.path)
                print(idx)
                cur_delt = 1 if key == ord("y") else -1
                while True:
                    path_prec = self.imgs[idx+cur_delt]
                    tmpimg = cv2.imread(path_prec)

                    if cur_delt not in circles_arch:
                        counter = Counter(path_prec, "counts", cv2.cvtColor(tmpimg, cv2.COLOR_RGB2GRAY),
                                          self.params, self.params.folder_save_auto)
                        if counter.load_count():
                            circles = sorted(counter.circles, key=lambda x: x[3])
                            circles_arch[cur_delt] = circles
                        else:
                            circles_arch[cur_delt] = None

                    if cur_delt in circles_arch and circles_arch[cur_delt] is not None:
                        circles = circles_arch[cur_delt]
                        thickness = 1
                        alpha = 0.1
                        overlay = tmpimg.copy()
                        output = tmpimg.copy()

                        for k, (x, y, r, c) in enumerate(circles): # CODE DUPLICATE TO CLEAN
                            d = distance_nearest(circles, x, y, c, k)
                            if d < 10 and not self.params.only_dark and not self.params.only_bright:
                                cv2.circle(overlay, (x, y), 3, (255, 255, 0), thickness)
                            if c == 1:
                                if not self.params.only_dark:
                                    cv2.rectangle(overlay, (x - 1, y - 1), (x + 1, y + 1), (255, 128, 0), -1)
                                    if not self.params.only_centers:
                                        cv2.circle(overlay, (x, y), r, (255, 0, 0), thickness)
                            else:
                                if not self.params.only_bright:
                                    cv2.rectangle(overlay, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)
                                    if not self.params.only_centers:
                                        cv2.circle(overlay, (x, y), r, (0, 255, 0), thickness)

                        for k, (x, y, r, c) in enumerate(self.image.counter.circles):  # CODE DUPLICATE TO CLEAN
                            if c == 1:
                                cv2.circle(overlay, (x, y), 1, (255, 128, 128), -1)
                            else:
                                cv2.circle(overlay, (x, y), 1, (70, 70, 255), -1)


                        if self.params.full_circle:
                            cv2.addWeighted(tmpimg, alpha, output, 1 - alpha,
                                            0, output)
                        else:
                            tmpimg = overlay


                    print(path_prec)
                    if self.params.fullscreen:
                        ns = (int(float(self.screen_h) / float(tmpimg.shape[0]) * float(tmpimg.shape[1])),
                              self.screen_h)
                        tmpimg = cv2.resize(tmpimg, ns, tmpimg)
                        cv2.resizeWindow('image', self.screen_w, self.screen_h)
                    cv2.putText(tmpimg, f'{path_prec.split("_")[-1].replace(".png", "")}', (0, 50), self.image.font, 2,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('image', tmpimg)
                    key = cv2.waitKey(0)
                    if key == ord('y'):
                        cur_delt += 1
                    elif key == ord('6'):
                        cur_delt -= 1
                    else:
                        break
                # if self.image_old is None:
                #     self.image_old = self.image
                #     #self.image = Image(self.imgs[self.basei-1], self.params)
                #
                #     cv2.imshow('image', self.image.output)
                # else:
                #     self.image = self.image_old
                #     self.image_old = None
            elif key == ord("q"):
                exit_val = True
                break
            elif key == ord("c"):
                self.image.params.only_centers = not self.image.params.only_centers  # todo improve

        cv2.destroyAllWindows()
        return exit_val