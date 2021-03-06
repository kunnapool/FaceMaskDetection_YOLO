import cv2


def iou(b1, b2, is_midpt=False):
    """
    :params: b1 is the first bounding box - b1 = (x1, y1, x2, y2)
    :params: b2 is the second bounding box
    :return: The intersection over union area of b1 and b2
    """

    if is_midpt:
        b1, b2 = get_box_cords_from_midpt(b1, b2)

    b1_x1, b1_y1, b1_x2, b1_y2 = b1
    b2_x1, b2_y1, b2_x2, b2_y2 = b2


    # the following 4 points (i_ ) represent the intersection box
    i_x1 = max(b1_x1, b2_x1)
    i_y1 = max(b1_y1, b2_y1)

    i_x2 = min(b1_x2, b2_x2)
    i_y2 = min(b1_y2, b2_y2)

    i_area = abs(i_x1 - i_x2) * abs(i_y1 - i_y2)

    u_area = abs(b1_x1 - b1_x2) * abs(b1_y1 - b1_y2) + \
             abs(b2_x1 - b2_x2) * abs(b2_y1 - b2_y2)


    return i_area/u_area


def get_box_cords_from_midpt(b1, b2):
    xm1, ym1, w1, h1 = b1
    xm2, ym2, w2, h2 = b2

    b1_x1 = xm1 - w1/2
    b1_y1 = ym1 - h1/2
    b1_x2 = xm1 + w1/2
    b1_y2 = ym1 + h1/2

    b2_x1 = xm2 - w2/2
    b2_y1 = ym2 - h2/2
    b2_x2 = xm2 + w2/2
    b2_y2 = ym2 + h2/2
    

    return (b1_x1, b1_y1, b1_x2, b1_y2), (b2_x1, b2_y1, b2_x2, b2_y2)

def nonmax_sup(b1, b2, b1_p, b2_p, is_midpt=False, iou_threshold=0.5):
    """
    Non-max suppression for bounding boxes.
    Function calculates IOU, and if the value if above the threshold, rejects the
    box with lower probability

    :return: -1 indicates the lower probability box should be rejected
              0 means it should be kept
    """
    max_p = max(b1_p, b2_p)
    iou = iou(b1, b2, is_midpt)

    if iou > iou_threshold:
        return -1
    
    return 0


def resize_image_with_BB(image, width, height, list_BB):
    h = image.shape[0]
    w = image.shape[1]

    new_size = (width, height)

    new_BB_list = []

    for i in range(len(list_BB)):

        x1, y1, x2, y2, l = list_BB[i]

        r_w = width/w
        r_h = height/h


        x1 *= r_w
        x2 *= r_w

        y1 *= r_h
        y2 *= r_h

        new_BB = (int(x1), int(y1), int(x2), int(y2), l)

        new_BB_list.append(new_BB)

    return cv2.resize(image, new_size, interpolation = cv2.INTER_AREA), new_BB_list

"""
References:
    https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    https://arxiv.org/pdf/1506.02640.pdf
"""