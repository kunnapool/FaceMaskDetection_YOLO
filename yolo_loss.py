import numpy as np
from helpers import iou # TODO Do I need this?
import math


def sse(x, y):

    s = 0
    for i in range(x):
        s += (x[i] - y[i])**2

    return s


def _get_sqrt(x):
    l = []
    for xi in x:
        l.append(math.sqrt(xi))
    
    return l


def loss(pred, target, B=2, S=13):
    """
    The target/pred size should be (SxSx(B*E + C))
        S - grid size
        B - Number of bounding boxes (number of predictions) per grid cell
        E - Number of elements predicted per box
        C - Number of classes

    Format of predicted elements, E - [x1, y1, w1, h1, conf1, x2... conf_n, class_probs]
        x, y - center point of the BB relative to the grid cell; relative range of [0, 1]
        w, h - width, height of the BB relative to the image; relative range of [0, 1]
        conf - Defined as Pr(obj) * IOU(pred, truth)
               0 if no obj
    """

    def get_best_box(arr):
        st = 0
        end = 5
        max_bb = []
        max_iou = -1

        # get the BB with the box prob
        for b in range(B):
            x1, y1, x2, y2, p = arr[st: end]
            st = end
            end += 5

            bb = [x1, y1, x2, y2]
            i = iou(bb, t_bb)

            if i > max_iou:
                max_bb = bb
                max_iou = i
        
        return max_bb

    lmbda = 0.5

    l1 = 0
    l2 = 0
    l3 = 0
    l4 = 0

    # iterate over each grid (per one image)
    for s1 in range(S):
        for s2 in range(S):    
            # this should have B instances of [x1, y1, x2, y2, conf]
            # and then class probabilites

            arr = pred[s1, s2]
            arr2 = target[s1, s2]
            x1, y1, x2, y2, _ = arr2
            t_bb = [x1, y1, x2, y2]

            # TODO check if there are any boxes at all
            best_bb = get_best_box(arr)

            x1, y1, x2, y2 = best_bb

            # The BB with the highest IOU (with the GD) will be the box responsible
            # for detecting

            l1 += (x1 - t_bb[0])**2 + \
                  (x2 - t_bb[2])**2 + \
                  (y1 - t_bb[1])**2 + \
                  (y2 - t_bb[3])**2

            bb_h = abs(y1 - y2)
            bb_w = abs(x1 - x2)

            t_bb_h = abs(t_bb[1] - t_bb[3])
            t_bb_w = abs(t_bb[0] - t_bb[2])

            l2 += (math.sqrt(bb_h) - math.sqrt(t_bb_h))**2 + \
                  (math.sqrt(bb_w) - math.sqrt(t_bb_w))**2
            

    return 0
