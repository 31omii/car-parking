import cv2
import matplotlib.pyplot as plt
import numpy as np

from bounding_box import get_parking_place_boxes, empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask = "C:\codes\choisaab\parking\mask\mask_1920_1080.png"
VidPath = "C:\codes\choisaab\parking\data\parking_1920_1080.mp4"

mask = cv2.imread(mask, 0)

cap = cv2.VideoCapture(VidPath)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

places = get_parking_place_boxes(connected_components)

places_status = [None for j in places]
diffs = [None for j in places]

previous_frame = None

frame_nmr = 0
X = True
step = 30
while X:
    X, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for place_indx, place in enumerate(places):
            x1, y1, w, h = place

            place_crop = frame[y1:y1 + h, x1:x1 + w, :]

            diffs[place_indx] = calc_diff(place_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        print([diffs[j] for j in np.argsort(diffs)][::-1])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(places))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for place_indx in arr_:
            place = places[place_indx]
            x1, y1, w, h = place

            place_crop = frame[y1:y1 + h, x1:x1 + w, :]

            place_status = empty_or_not(place_crop)

            places_status[place_indx] = place_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for place_indx, place in enumerate(places):
        place_status = places_status[place_indx]
        x1, y1, w, h = places[place_indx]

        if place_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Available places: {format(str(sum(places_status)))} / {str(len(places_status))}", (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow("Parking Camera", cv2.WINDOW_NORMAL)
    cv2.imshow("Parking Camera", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()