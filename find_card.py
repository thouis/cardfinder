import cv2
import numpy as np
from scipy.ndimage.filters import median_filter
import os.path

def open_camera():
    camera = cv2.VideoCapture()
    camera.open(1)
    return camera

def center_subwindow(im, sz):
    start = [(d - sz) / 2 for d in im.shape]
    return im[[slice(st, (st + sz)) for st in start]]


def getframe(camera):
    status, im = camera.read()
    assert status, "Error reading image %d" % status
    return center_subwindow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 500)

def kmeans(data, lo, hi, iter=20):
    for i in range(iter):
        distlo = abs(data - lo)
        disthi = abs(data - hi)
        masklo = distlo < disthi
        lo = np.mean(data[masklo])
        hi = np.mean(data[~ masklo])
    return lo, hi, (~ masklo)

def find_cards():
    camera = open_camera()
    background = getframe(camera)
    kernel = np.ones((5,5),np.uint8)

    have_taken = False
    previous_snapshot = None
    image_idx = 0
    prev_images = []

    while True:
        im = getframe(camera)
        diff = abs(im.astype(np.float32) - background.astype(np.float32)).astype(np.uint8)
        cv2.imshow("difference image", diff)

        l, h, m = kmeans(diff, diff.min(), diff.max())
        contours, hierarchy = cv2.findContours(255 * m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_idx = [(cv2.contourArea(i), idx) for idx, i in enumerate(contours)]
        area_idx.sort()
        biggest = area_idx[-1]
        rect = cv2.minAreaRect(contours[biggest[1]])

        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)

        # make first edge short
        short = np.linalg.norm(box[0, :] - box[1, :])
        long = np.linalg.norm(box[1, :] - box[2, :])
        if long < short:
            box = np.roll(box, 1, 0)
            short, long = long, short

        keypress = cv2.waitKey(100)
        if keypress == ord('r'):
            print "reaquire"
            background = getframe(camera)
            continue
        elif keypress == ord('q'):
            camera.release()
            return


        # check aspect ratio
        mtg_aspect = 56.0 / 81.0
        if abs(short / long - mtg_aspect) / mtg_aspect > 0.15:
            have_taken = False
            continue

        mat = cv2.getAffineTransform(box[:3, :].astype(np.float32),
                                     np.array([[0, 0],
                                               [short, 0],
                                               [short, long]], np.float32))
        rotated = cv2.warpAffine(im, mat, (int(short), int(long)), flags=cv2.INTER_LINEAR)

        small_rot = cv2.resize(rotated, (56, 81))

        if previous_snapshot is not None:
            prev_diff = abs(small_rot.astype(float) - previous_snapshot.astype(np.float))
            print "PD", np.median(prev_diff)
        else:
            prev_diff = np.zeros_like(small_rot, np.float32)
        STABLE_THRESHOLD = 3
        NEW_CARD_THRESHOLD = 20
        if (not have_taken) and (np.median(prev_diff) <= STABLE_THRESHOLD):
            have_taken = True
            while os.path.exists("card_%05d.png" % image_idx): image_idx += 1
            cv2.imwrite("card_%05d.png" % image_idx, rotated)
            image_idx += 1

            prev_images.append(rotated)
            prev_images = prev_images[-5:]
            stack_height = max(p.shape[0] for p in prev_images)
            prev_stack = np.hstack([np.pad(p, ((0, stack_height - p.shape[0]), (0, 0)), 'constant') for p in prev_images])
            cv2.imshow("captures", prev_stack)

        elif have_taken and (np.median(prev_diff) > NEW_CARD_THRESHOLD):
            have_taken = False
        previous_snapshot = small_rot


if __name__ == '__main__':
    find_cards()
