import numpy as np
import cv2
import os

import counter

# charger dans ram si besoin


class Image:
    def __init__(self, path, params):
        self.path = path
        self.params = params
        print("\tReading image file", self.path)

        if not self.params.is_optimizing:
            self.img_init = cv2.imread(self.path)
            self.raw = cv2.cvtColor(self.img_init, cv2.COLOR_RGB2GRAY)
            self.output = self.img_init.copy()
            self.font = cv2.FONT_HERSHEY_SIMPLEX
        else:
            self.raw = None

        self.counter = counter.Counter(self.path, "counts", self.raw, self.params, self.params.folder_save_auto)

        self.help_msg = """
h: toggle help
f: toggle fullscreen mode
g: toggle display
r: remove circle
d: add dark circle
b: add bright circle
s: save circles
l, v: load circles
escape, n: next photo
c: toggle circle display
space: recount beads
q: quit
"""

        self.mx = 0
        self.my = 0

    def shape(self):
        return self.raw.shape

    def count_info(self):
        nb_d = self.counter.nb_dark()
        nb_b = self.counter.nb_bright()
        sm = nb_d + nb_b
        if sm == self.params.n:
            cv2.putText(self.output, "T" + str(nb_d+nb_b) + "/" + str(self.params.n), (0, 50), self.font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        elif sm > self.params.n:
            cv2.putText(self.output, "T" + str(nb_d+nb_b) + "/" + str(self.params.n), (0, 50), self.font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(self.output, "T" + str(nb_d+nb_b) + "/" + str(self.params.n), (0, 50), self.font, 2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.output, "D" + str(nb_d), (0, 100), self.font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(self.output, "B" + str(nb_b), (0, 150), self.font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        if self.counter.loaded_or_saved:
            cv2.putText(self.output, "S", (0, 200), self.font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(self.output, "N", (0, 200), self.font, 2, (255, 0, 0), 2, cv2.LINE_AA)

        if self.params.is_boiling:
            cv2.putText(self.output, "B", (0, 250), self.font, 2, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(self.output, "NB", (0, 250), self.font, 2, (255, 0, 255), 2, cv2.LINE_AA)

    def display_help(self):
        self.output = np.ones(self.raw.shape, np.uint8) * 255
        for i0, ln in enumerate(self.help_msg.split('\n')):
            cv2.putText(self.output, ln, (0, i0 * 50), self.font, 2, (0, 255, 0), 2, cv2.LINE_AA)

    def background(self):
        self.output = self.img_init.copy()

    def draw(self):
        alpha = 0.1
        #print("Draw", self.path)

        overlay = self.output.copy()
        output = self.output.copy()

        circles = sorted(self.counter.circles, key=lambda x: x[3])

        thickness = 1
        if self.params.full_circle:
            thickness = -1

        for k, (x, y, r, c) in enumerate(circles):
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
        if self.params.full_circle:
            cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                        0, output)
        else:
            output = overlay
        self.output = output
        self.count_info()

    def resize(self, new_size):
        self.output = cv2.resize(self.output, new_size)



def distance_nearest(circles, x0, y0, c0, k0):
    d_min = np.inf
    for k, (x, y, r, c) in enumerate(circles):
        d = np.sqrt((x0 - x)**2 + (y0 - y)**2)
        if k != k0 and d < d_min: # and c == c0
            d_min = d
    return d_min