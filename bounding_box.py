import pickle
from skimage.transform import resize
import numpy as np
import cv2

EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("C:\codes\choisaab\parking\model\model.p", "rb"))


def empty_or_not(spot_bgr):

    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_place_boxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    BB = []
    coef = 1
    for i in range(1, totalLabels):

        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        BB.append([x1, y1, w, h])

    return BB