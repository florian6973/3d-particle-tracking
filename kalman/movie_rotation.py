import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon
import numpy as np
from numpy import radians as rad


mx = 0
my = 0
def mouse(self, x, y, flags, param):
    global mx, my
    mx = x
    my = y

def select(path, i, step, nbs, read=True):
    global mx, my
    pos = []

    if read:
        data = np.loadtxt(f'H:\Data_1\pos_{nbs}_{step}_{i}.txt').astype(int)

    files = sorted(glob.glob(os.path.join(path, '*.png')))

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    cv2.setWindowTitle('image', 'Beads counter - h for help')


    video_name = r'H:\Data_2/trajectory.mp4'
    images = [img for img in os.listdir(path) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    cut = 185
    height = height - cut - 70
    width = width - cut
    print(height, width)

    video = cv2.VideoWriter('H:/Data_1/trajectory_after.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    # lab or dish frame, otherwise open archives



    #step = 4#2#4#8#2#8
    #nbs = 50#100#50#100#50#30 # compute the number for x revolutions

    nbsc = np.linspace(0, 255, nbs) #np.logspace(0, np.log10(255), nbs)

    def get_color(j):
        return tuple((255-nbsc[j], 0, nbsc[j]))
        #return (255-255/nbs*j, 125/nbs*j, 255/nbs*j)#(0, 255, int(255/nbs*j))

    for k in range(i, i+nbs*step, step):
        img = cv2.imread(files[k])
        #img = cv2.addWeighted(img, 2, np.zeros(img.shape, img.dtype),0, 1)

        if not read:
            cv2.setMouseCallback('image', mouse)
            cv2.imshow('image', img)

            key = cv2.waitKey(0)
            pos.append([mx, my])
            print(pos)
        else:
            pos = data[:(k-i)//step+1].copy()
            pos[:, 1] = 600 - pos[:, 1]
            #print(pos)

        print(get_color((k-i)//step))
        for j, (xt, yt) in enumerate(pos):
            print(j, get_color(j))
            img = cv2.circle(img, (xt, yt), 7, get_color(j), -1)
        for j in range(len(pos)-1):
            img = cv2.line(img, pos[j], pos[j+1], get_color(j), 6)
        img = img[cut//2+35:-cut//2-35, cut//2:-cut//2]
        video.write(img)


    cv2.destroyAllWindows()
    video.release()

    pos = np.array(pos)
    pos[:,1] = 600-pos[:,1]
    # plt.plot(pos[:,0], pos[:,1])
    # # set axis limits
    # plt.xlim([0, 600])
    # plt.ylim([0, 600])
    # plt.show()

    if not read:
        np.savetxt(f'H:\Data_1\pos_{nbs}_{step}_{i}.txt', pos, fmt='%d')

def drawCirc(ax,radius,centX,centY,angle_,theta2_,color_='black'):
    #========Line
    arc = Arc([centX,centY],radius,radius,angle=angle_,
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=10,color=color_)
    ax.add_patch(arc)


    #========Create the arrow head
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/9,                # radius
            rad(angle_+theta2_),     # orientation
            color=color_
        )
    )
    ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius])
    # Make sure you keep the axes scaled or else arrow will distort

def unique_plot(i):
    data = np.loadtxt(f'H:\Data_1\pos_{i}.txt')

    plt.figure(figsize=(10,10))

    nbs = len(data)
    def get_color(j):
        return matplotlib.colors.to_hex((1-1/nbs*j, 0.5, 1/nbs*j))

    for k in range(nbs):
        #print(get_color(k))
        plt.scatter([data[k, 0]], [data[k, 1]], color=get_color(k), s=0.1)


    for k in range(nbs-1):
        plt.quiver(data[k, 0], data[k, 1],
                   data[k+1, 0] - data[k, 0],
                   data[k+1, 1] - data[k, 1], color=get_color(k), width=0.001, scale=1000)
        plt.plot(data[k:k+2, 0], data[k:k+2, 1], color=get_color(k), linewidth=0.2)

    plt.plot([300], [300], marker=r'$\circlearrowright$', ms=100)
    #ax = plt.gca()
    #drawCirc(ax, 300, 300, 300, 0, -250)

    plt.scatter([data[0, 0]], [data[0, 1]], marker='+')
    plt.scatter([data[-1, 0]], [data[-1, 1]], marker='*')
    plt.xlim([0, 600])
    plt.ylim([0, 600])
    plt.show()
    pass

def read_trajectory():
    file = r'H:\Data_1\100\velocity_200\trajectories/iteration_0_trajectories.npz'
    data = np.load(file)
    print(data.files)

    image_path = r'H:\Data_1\100\velocity_200\iteration_0\centered_images/'
    #image_path = r'H:\Data_1\100\velocity_200\iteration_0\raw_images/'
    files = glob.glob(image_path + '*.tif')

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(len(data["X"])/100)


    nbsc = np.linspace(0, 1, 101) #np.logspace(0, np.log10(255), nbs)

    # def get_color(j):
    #     return tuple((1-nbsc[j], 0, nbsc[j]))
    #
    # for i in range(0, 1000, 10):
    #     print(i)
    #     plt.plot([data['X'][i], data['X'][i+10]],
    #              [data['Y'][i], data['Y'][i+10]], color=get_color(i//10), linewidth=0.2)
    # plt.show()
    # exit()

    for i in range(100):
        img = cv2.imread(files[i])
        img = cv2.addWeighted(img, 3, np.zeros(img.shape, img.dtype), 0, 1)


        #pos_frame_x = data['X'][i::69994]
        #pos_frame_y = data['Y'][i::69994]


        for j in range(100):
            img = cv2.circle(img, (int(pos_frame_x[j]), int(pos_frame_y[j])), 3, (0, 255, 0), -1)

        cv2.imshow('image', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    path_folder = r'H:\Data_1\113\350_extract\velocity_350_iteration_002_archive\raw_images'
    #path_folder = r'H:\Data_1\113\280_extract\velocity_280_iteration_006_archive\raw_images'
    pos_before = []
    pos_after = []

    #i_bef = 11000
    #i_after = 40000


    i_bef = 5000
    i_after = 15000

    #read_trajectory()

    #pos_before = select(path_folder, i_bef, 4, 50, False) # 50 at 4
    #pos_after = select(path_folder, i_after, 2, 100, False) # 100 at 2

    #pos_before = select(path_folder, i_bef, 8, 100, True) # 50 at 4
    #pos_after = select(path_folder, i_after, 2, 100, False) # 100 at 2

    pos_after = select(path_folder, 30000, 1, 90, True) # 100 at 2

    #unique_plot(i_bef)
    #unique_plot(i_after)

    #read_positions()

# save it and reload it
# [[479, 395], [500, 400], [516, 412], [526, 420], [518, 426], [523, 428], [522, 433], [516, 439], [512, 447], [503, 458], [498, 469], [491, 476], [483, 481], [479, 483], [467, 488], [464, 491], [458, 485], [452, 481], [445, 474], [445, 471], [445, 463], [448, 447], [453, 439], [466, 430], [476, 427], [490, 424], [500, 425], [504, 437], [508, 454], [505, 461], [500, 472], [495, 481], [490, 487], [487, 494], [480, 505], [471, 518], [462, 528], [453, 538], [443, 540], [433, 540], [420, 535], [412, 530], [404, 521], [401, 513], [402, 505], [406, 496], [413, 488], [423, 488], [439, 488], [459, 494], [467, 507], [467, 528], [458, 534], [461, 539], [460, 540], [458, 540], [457, 534], [453, 534], [449, 539], [446, 539], [434, 539], [424, 536], [416, 532], [407, 524], [406, 517], [408, 513], [411, 504], [425, 491], [436, 480], [450, 475], [472, 465], [500, 465], [504, 458], [504, 464], [496, 474], [489, 489], [483, 497], [478, 506], [474, 511], [465, 513], [462, 515], [459, 519], [451, 523], [441, 525], [433, 526], [422, 526], [413, 519], [406, 511], [399, 502], [399, 488], [394, 473], [403, 468], [412, 464], [426, 461], [438, 461], [445, 478], [455, 491], [465, 515], [469, 530], [460, 526]]

def read_positions():
    pos_file = r'H:\Data_1\113\310\data_310\velocity_310_iteration_001_output\output/data_merged.txt'
    pos = np.loadtxt(pos_file, delimiter='\t')
    print(pos.shape)
    return pos

from trajectories.Tracker import Tracker

def format_centers(x,y):
    b = list(np.array(([x], [y])).T)
    for i in range(len(b)):
        b[i] = b[i].T
    return b
def tracking_pos(pos, start_i, length):
    n = 113

    distanceThresh = 200  # 430
    dt = 1. / 200.
    conversion = 6.35 / (1000 * 33)  # pixels to m, dbeads(mm)/m/(nbpixelsdiameter)

    tracker = Tracker(distanceThresh, 5, 100000, n)
    trajs = []
    trajsv = []
    onetrack = []

    old_centers = None
    for k in range(start_i, start_i + length):
        print(f"\r\tTracking {len(onetrack)+1:04}/{length:04} ", end='')

        loc_pos = pos[pos[:, 0] == k].copy()
        centers = loc_pos[:, 1:3] # or groupby frame?
        print(centers.shape)
        #print(centers)
        #exit()

        centersf = format_centers(centers[:, 0], centers[:, 1])
        tracker.Update(centersf)

        def coords(tr):
            coordinates = tr
            coordinatesD = np.array(coordinates)
            # coordinatesT = coordinatesD.reshape((coordinatesD.shape[0], 2))
            coordinatesT = coordinatesD.flatten()
            coordinatesTT = coordinatesT.T
            return coordinatesTT[0], coordinatesTT[1]

        def get_point(c, x, y):
            distances = np.sqrt((x - c[:, 0]) ** 2 + (y - c[:, 1]) ** 2)
            return c[np.argmin(distances)]

        if len(tracker.tracks) == 0:
            print("No tracks")

        # if len(tracker.tracks) != n:
        #     print("Not enough tracks")
        #     print('Stopping')
        #     break

        vls = []
        outputs = []
        discard = []
        for i in range(len(tracker.tracks)):
            # print(tracker.tracks[i].trace[-1])
            try:
                x, y = coords(tracker.tracks[i].trace[-1])

                if i == 0:
                    onetrack.append([x, y])

                outputs.append((x, y))
            except:
                print("Error", i)
                discard.append(i)
                break
            #print(x,y)

            if (len(tracker.tracks[i].trace) > 1):  # when boiling it crashes
                x1, y1 = coords(tracker.tracks[i].trace[-2])
                x2, y2 = coords(tracker.tracks[i].trace[-1])

                #poss.append([x2, y2])

                x1, y1 = get_point(old_centers, x1, y1)
                x2, y2 = get_point(centers, x2, y2)

                vel = ((x2 - x1)*conversion / dt) ** 2 + ((y2 - y1)*conversion / dt) ** 2
                #vel = dx / dt
                # check conversion
                vls.append(vel) # no data for first frame
            else:
                print("\rno trace", end='')
                pass
        old_centers = centers.copy()
        trajs.append(outputs)
        if vls != []:
            trajsv.append(vls)

        # if i0 > 10:
        #     cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
        #     ofile = ofile.replace('\\', '/').replace('.txt', '.png')
        #     #print(ofile)
        #     img = cv2.imread(ofile)
        #     #print(img.shape)
        #     outputsc = np.array(outputs).astype(int)
        #     for i, (x, y) in enumerate(outputsc):
        #         cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
        #         # put text near the circle
        #         cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        #     cv2.putText(img, str(np.mean(vls)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        #     cv2.imshow("image", img)
        #     cv2.waitKey(0)

    tracks = []
    for i in range(len(tracker.tracks)-10):
        print(len(tracker.tracks[i].trace))
        onetrack = []
        for tr in tracker.tracks[i].trace[10:]:
            onetrack.append(coords(tr))
        tracks.append(onetrack)
    tracks = np.array(tracks)
    #print(onetrack)
    #np.savetxt(f'onetrack-{start_i}-{length}.txt', onetrack)
    np.save(f'tracks-{start_i}-{length}.npy', tracks)

    #trajs = np.array(trajs)[10:, :, :] # remove first beads kalman filters
    #print(trajsv)
    #trajsv = np.array(trajsv)[(10 - 1):, :] # remove first beads kalman filters
    #print(trajs.shape)
    #print(trajsv.shape)

    # folder_crop = rf'H:\Data_2\n_beads_{n:03}_Delrin_white_0.25_Vmin_210_Vmax_350_Step_10_Cooling_140_humidity_55_65\velocity_{v:03}\iteration_{it:03}\raw_images_sanity_check\crop'
    #
    # avg = np.array(np.mean(np.mean(trajsv, axis=1)))
    # savef = os.path.join(folder_crop, f'trajs_{n}_{v}_{it}_2.npz')
    # np.savez(savef, trajs=trajs, trajsv=trajsv,
    #          avg=avg,
    #          std=np.array(np.std(np.mean(trajsv, axis=1))),
    #          fmt='%d')
    # print(f"\tSaved {savef} - {avg}")
    #print(np.array(np.mean(np.mean(trajsv, axis=1))))
    #np.savez(savef, trajs, fmt='%d')

def plot_pos(start_i, length):
    data = np.load(f'tracks-{start_i}-{length}.npy', allow_pickle=True)
    print(data.shape)
    data = data[0]
    nbsc = np.linspace(0, 1, length+1)  # np.logspace(0, np.log10(255), nbs)
    print(data)
    data= np.array(data)
    print(data.shape)

    def get_color(j):
        return tuple((1-nbsc[j], 0, nbsc[j]))

    for i in range(0, length-20, 10):
        #print(i)
        plt.plot([data[i, 0], data[i+10, 0]],
                 [data[i, 1], data[i+10, 1]], color=get_color(i//10), linewidth=0.2)
    plt.show()
    exit()
    pass

def movie_traj(start_i, length, step=10):
    data = np.load(f'tracks-{start_i}-{length}.npy', allow_pickle=True)
    print(data.shape)
    data = data[10]
    nbsc = np.linspace(0, 1, length+1)  # np.logspace(0, np.log10(255), nbs)
    print(data)
    data= np.array(data)
    print(data.shape)

    def get_color(j):
        return tuple((1-nbsc[j], 0, nbsc[j]))

    # for i in range(0, length-20, 10):
    #     #print(i)
    #     plt.plot([data[i, 0], data[i+10, 0]],
    #              [data[i, 1], data[i+10, 1]], color=get_color(i//10), linewidth=0.2)
    # plt.show()
    # exit()

    image_path = r'H:\Data_1\113\310_extract\velocity_310_iteration_001_archive\raw_images/'
    # image_path = r'H:\Data_1\100\velocity_200\iteration_0\raw_images/'
    files = glob.glob(image_path + '*.png')

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    images = [img for img in os.listdir(image_path) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width, layers = frame.shape
    cut = 185
    height = height - cut - 70
    width = width - cut
    print(height, width)

    video = cv2.VideoWriter('H:/Data_1/trajector_before.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 9, (width, height))

    # lab or dish frame, otherwise open archives

    # step = 4#2#4#8#2#8
    # nbs = 50#100#50#100#50#30 # compute the number for x revolutions

    step = 10
    nbsc = np.linspace(0, 255, length//step+1)  # np.logspace(0, np.log10(255), nbs)

    def get_color(j):
        return tuple((255 - nbsc[j], 0, nbsc[j]))
        # return (255-255/nbs*j, 125/nbs*j, 255/nbs*j)#(0, 255, int(255/nbs*j))

    for k in range(start_i, start_i+length, step):#start_i + length * step, step):
        img = cv2.imread(files[k+10])
        img = cv2.addWeighted(img, 2, np.zeros(img.shape, img.dtype), 0, 1)

        #pos = data.copy()
        pos = data[:(k - start_i):step].copy()
        #pos = data[:(k - start_i) // step + 1:step].copy()
        pos = pos.astype(int)
        pos[:, 1], pos[:, 0] = pos[:, 1].copy()+135, pos[:, 0].copy()+110
            # print(pos)

        print(get_color((k - start_i) // step))

        for j, (xt, yt) in enumerate(pos):
            xt, yt = int(xt), int(yt)
            #print(j, get_color(j))
            img = cv2.circle(img, (xt, yt), 7, get_color(j), -1)
        for j in range(len(pos) - 1):
            img = cv2.line(img, pos[j], pos[j + 1], get_color(j), 6)
        img = img[cut//2+35:-cut//2-35, cut//2:-cut//2]
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    pass

print("Loading pos")
#pos = read_positions()
#tracking_pos(pos, 4000, 2000)#1000)

#movie_traj(4000, 2000)#1000)
#plot_pos(4000, 2000)
