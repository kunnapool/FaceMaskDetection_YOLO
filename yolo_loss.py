import numpy as np
from helpers import iou
import math


def _get_grid_BBs(target, S=13):
    """
    The incoming target variable is a list of all the
    BBs present in an image

    Assume that the image is divided into SxS grids and figure out of
    which grid does the BB land in
    """

    grid_target = -1*np.ones((S, S, 5))

    for b in target:
        x1, y1, x2, y2, l = b
        mid_x = abs(x1 - x2)/2
        mid_y = abs(y1 - y2)/2

        s1 = int(mid_y/S)
        s2 = int(mid_x/S)

        grid_target[s1, s2] = [x1, y1, x2, y2, l]

    return grid_target

def _get_softmax(x):
    l = []
    s = 0
    for xi in x:
        s += np.exp(xi)
    
    for xi in x:
        l.append(xi/s)

    return l


def loss(pred, target, B=2, S=13):
    """
    The pred size should be (SxSx(B*E + C))
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
        best_conf = -1 # confidence of the best box

        # get the BB with the box prob
        for b in range(B):
            x1, y1, x2, y2, conf = arr[st: end]
            st = end
            end += 5

            bb = [x1, y1, x2, y2]
            i = iou(bb, t_bb)

            if i > max_iou:
                max_bb = bb
                max_iou = i
                best_conf = conf

        return max_bb, max_iou, best_conf

    lmbda = 0.5

    target = _get_grid_BBs(target, S=7)

    loss_1 = 0 # TODO Also update these names
    loss_2 = 0
    loss_3 = 0
    loss_4 = 0

    # iterate over each grid (per image)
    for s1 in range(S):
        for s2 in range(S):    
            # this should have B instances of [x1, y1, x2, y2, label] # TODO Update this
            # and then class probabilites

            arr = pred[s1, s2]
            arr2 = target[s1, s2]
            x1, y1, x2, y2, gd_label = arr2
            t_bb = [x1, y1, x2, y2]

            # TODO check if there are any boxes at all
            # TODO_UPDATE - this might not be needed as every grid predicts B number of BBs
            best_bb, best_iou, pred_conf = get_best_box(arr)

            x1, y1, x2, y2 = best_bb

            # The BB with the highest IOU (with the GD) will be the box responsible
            # for detecting

            if gd_label != -1:
                loss_1 += (x1 - t_bb[0])**2 + \
                    (x2 - t_bb[2])**2 + \
                    (y1 - t_bb[1])**2 + \
                    (y2 - t_bb[3])**2

            bb_h = abs(y1 - y2)
            bb_w = abs(x1 - x2)

            t_bb_h = abs(t_bb[1] - t_bb[3])
            t_bb_w = abs(t_bb[0] - t_bb[2])

            if gd_label != -1:
                loss_2 += (math.sqrt(bb_h) - math.sqrt(t_bb_h))**2 + \
                    (math.sqrt(bb_w) - math.sqrt(t_bb_w))**2
            

            # # TODO What should we do about these???
            # if gd_label != -1:
            #     loss_3 += (gd_conf - best_iou)
            # else:
            #     # TODO add lambda constant here
            #     loss_3 += (gd_conf - best_iou)
            

            # TODO Does this operation still make sense?
            # TODO_UPDATE now calculating cross entropy loss with softmax instead of sse
            if gd_label != -1:
                pred_class_probs = _get_softmax(arr[5*B:])
                gd_class_label = arr[5:]

                # -1 because all the probability goes to the gd label
                loss_4 += -1 * math.log(pred_class_probs[gd_class_label])


    return (loss_1 + loss_2 + loss_3 + loss_4)


"""
References:
    https://hackernoon.com/understanding-yolo-f5a74bbc7967
    https://towardsdatascience.com/yolov1-you-only-look-once-object-detection-e1f3ffec8a89
    https://arxiv.org/pdf/1506.02640.pdf
"""