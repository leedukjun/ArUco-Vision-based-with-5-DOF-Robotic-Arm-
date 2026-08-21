#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ArUco workspace + unknown-object detection, v3 (background subtraction only).

The camera and mat are strictly static, so segmentation is pure background
subtraction: press 'b' to capture a 15-frame median reference of the
workspace, and from then on foreground = |current - reference|. Grid lines,
mat texture, marker leftovers and anything else static cancel out exactly —
they are invisible by construction, no tuning needed.

Background-capture discipline: everything that should be IGNORED must be in
the reference. Capture with the mat in its resting state (foam block too, if
it lives on the mat), and re-press 'b' whenever anything static is moved —
otherwise its old location shows up as a "ghost" in the diff.

Two lessons baked into this file (they are why v1 kept locking onto the
wrong things):

  * Never paint masks onto the INPUT image — a black rectangle on a lighter
    mat is itself a giant step edge. The ignore-mask is applied AFTER
    segmentation, to the binary map, where zeroing pixels can only delete
    evidence, never create it.

  * Detection must run on a PRISTINE warped frame. v1 drew crosshairs and
    marker annotations first and then detected on that image, so it was
    tracking its own drawings.

Contour selection is a rejection gauntlet, not "largest wins":
  - area inside [Min Area, 50% of the field]
  - bounding box must not touch the masked border strip -> kills the robot
    arm reaching in and the boundary-marker leftovers
  - minAreaRect short side must exceed ~12 px -> kills anything long + thin
  - largest survivor wins.

The IK target is the mean of the contour's bottom pixel band, pulled up by
PARALLAX_OFFSET_PX, then EMA-smoothed.

Keys:  b = capture the background reference (remove the unknown object first!)
       o = print unknown-object base position
       q = quit
"""

import time
import cv2
import numpy as np
# import robot_control_python

# Dummy function required by OpenCV trackbars
def empty(a):
    pass

def getMarkerCoordinates(markers, ids, point=0):
    marker_array = []
    if markers:
        for marker in markers:
            marker_array.append([int(marker[0][point][0]), int(marker[0][point][1])])
    return marker_array, ids

def draw_corners(img, corners):
    for corner in corners:
        cv2.circle(img, (corner[0], corner[1]), 10, (0, 255, 0), thickness=-1)

def draw_numbers(img, corners, ids):
    if not ids or not corners: return
    number = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 4
    for corner in corners:
        if number < len(ids):
            cv2.putText(img, str(ids[number]), (corner[0]+10, corner[1]+10), font, 2, (0, 0, 0), thickness)
        number += 1

def show_spec(img, corners):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    amountOfCorners = len(corners)
    spec_string = str(amountOfCorners) + " markers found."
    cv2.putText(img, spec_string, (15, 15), font, 0.5, (0, 0, 250), thickness)

def draw_field(img, corners, ids):
    if len(corners) == 4 and ids is not None:
        markers_sorted = [0, 0, 0, 0]
        for sorted_corner_id in [0, 1, 2, 3]:
            if sorted_corner_id in ids:
                index = ids.index(sorted_corner_id)
                markers_sorted[sorted_corner_id] = corners[index]
        contours = np.array(markers_sorted)
        overlay = img.copy()
        cv2.fillPoly(overlay, pts=[contours], color=(255, 215, 0))
        alpha = 0.4
        img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        squarefound = True
    else:
        img_new = img
        squarefound = False
    return img_new, squarefound

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def get_markers(vid_frame, aruco_dictionary, aruco_parameters):
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
    bboxs, ids, rejected = detector.detectMarkers(vid_frame)
    if ids is not None:
        ids_sorted = ids.flatten().tolist()
    else:
        ids_sorted = []
    return bboxs, ids_sorted

# =================================================================
# UNKNOWN OBJECT DETECTION v3 (background subtraction only)
# =================================================================

BORDER_PAD_PX          = 10    # ignore this many px along each field edge
MIN_OBJECT_THICKNESS   = 12    # px; minAreaRect short side below this = noise sliver
MAX_AREA_FRACTION      = 0.50  # contours bigger than half the field = warp noise
PARALLAX_OFFSET_PX     = 15    # pull the grip point up from the silhouette's
                               # bottom edge; calibrate to object height * px/mm
ARM_BASE_POLYGONS      = []    # if the arm base sits at a FIXED spot inside the
                               # warped field, list its polygon(s) here (warped-
                               # image px) to make it invisible, e.g.
                               # ARM_BASE_POLYGONS = [[(0,150),(60,150),(60,260),(0,260)]]


def build_ignore_mask(shape_hw, border_pad=BORDER_PAD_PX, extra_polygons=None):
    """255 = region invisible to the detector. Applied AFTER segmentation,
    to the binary map — never painted onto the camera image, so it can
    only delete evidence, never fabricate edges."""
    h, w = shape_hw
    ignore = np.zeros((h, w), np.uint8)

    if border_pad > 0:                       # field boundary markers + warp fringe
        ignore[:border_pad, :] = 255
        ignore[h - border_pad:, :] = 255
        ignore[:, :border_pad] = 255
        ignore[:, w - border_pad:] = 255

    if extra_polygons:                       # fixed arm-base region, if any
        for p in extra_polygons:
            cv2.fillPoly(ignore, [np.asarray(p, np.int32)], 255)

    return ignore


def _segment_bgdiff(warped_bgr, background_bgr, thresh):
    """Foreground = |current - empty-workspace reference|. Grid lines and
    mat texture are identical in both images, so they vanish exactly."""
    if background_bgr.shape != warped_bgr.shape:
        background_bgr = cv2.resize(
            background_bgr, (warped_bgr.shape[1], warped_bgr.shape[0]))
    diff = cv2.absdiff(warped_bgr, background_bgr).max(axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return binary


def _clean_binary(binary, ignore_mask):
    binary[ignore_mask > 0] = 0
    # opening kills specks / thin slivers, closing fills holes in the object
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return binary


def select_object_contour(binary, min_area,
                          max_area_frac=MAX_AREA_FRACTION,
                          min_thickness=MIN_OBJECT_THICKNESS,
                          border_margin=BORDER_PAD_PX + 4):
    # border_margin must reach past the masked border strip: anything the
    # mask truncated there (the arm reaching in, boundary markers) still
    # has its bounding box pressed against the strip.
    """Run every contour through the rejection gauntlet; return the
    largest survivor (or None)."""
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    max_area = max_area_frac * h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if (x <= border_margin or y <= border_margin or
                x + bw >= w - border_margin or y + bh >= h - border_margin):
            continue                          # touches field edge: arm / boundary markers
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        if min(rw, rh) < min_thickness:
            continue                          # long + thin = noise sliver
        if area > best_area:
            best, best_area = cnt, area
    return best


def contour_grip_point(cnt, parallax_offset=PARALLAX_OFFSET_PX, band=4):
    """Bottom of the silhouette, averaged over a small pixel band so one
    noisy pixel can't drag the target, then pulled up to compensate the
    perspective 'lean' of a 3D object in the flattened view."""
    pts = cnt.reshape(-1, 2)
    y_max = int(pts[:, 1].max())
    bottom = pts[pts[:, 1] >= y_max - band]
    tx = int(round(float(bottom[:, 0].mean())))
    ty = max(0, y_max - parallax_offset)
    return tx, ty


class UnknownObjectDetector:
    def __init__(self, smoothing_factor=0.1, jump_reset_dist=100,
                 parallax_offset=PARALLAX_OFFSET_PX, miss_reset_frames=15):
        self.background_bgr = None
        self._bg_pool = []
        self._bg_wanted = 0
        self.smoothed = None
        self.smoothing_factor = smoothing_factor
        self.jump_reset_dist = jump_reset_dist
        self.parallax_offset = parallax_offset
        self.miss_reset_frames = miss_reset_frames
        self._miss_count = 0

    # ---------- background reference ----------
    def request_background(self, n_frames=15):
        self._bg_pool, self._bg_wanted = [], n_frames

    def collecting_background(self):
        return self._bg_wanted > 0

    def has_background(self):
        return self.background_bgr is not None

    def _feed_background(self, warped_bgr):
        if self._bg_pool and self._bg_pool[0].shape != warped_bgr.shape:
            self._bg_pool = []                # warp size changed mid-capture
        self._bg_pool.append(warped_bgr.copy())
        if len(self._bg_pool) >= self._bg_wanted:
            stack = np.stack(self._bg_pool).astype(np.float32)
            self.background_bgr = np.median(stack, axis=0).astype(np.uint8)
            self._bg_pool, self._bg_wanted = [], 0
            print("[INFO] Background reference captured. Detection is now active.")

    # ---------- smoothing ----------
    def reset_smoothing(self):
        self.smoothed = None

    def _smooth(self, tx, ty):
        if self.smoothed is None:
            self.smoothed = [float(tx), float(ty)]
        else:
            if np.hypot(tx - self.smoothed[0], ty - self.smoothed[1]) > self.jump_reset_dist:
                self.smoothed = [float(tx), float(ty)]   # new object: snap
            else:
                a = self.smoothing_factor
                self.smoothed[0] = a * tx + (1 - a) * self.smoothed[0]
                self.smoothed[1] = a * ty + (1 - a) * self.smoothed[1]
        return int(round(self.smoothed[0])), int(round(self.smoothed[1]))

    # ---------- main entry ----------
    def detect(self, warped_bgr, seg_thresh, min_area,
               extra_ignore_polygons=None, border_pad=BORDER_PAD_PX,
               annotate_on=None):
        """warped_bgr MUST be a pristine warped frame — nothing drawn on it.
        Returns (target_xy_or_None, binary_debug_image)."""
        h, w = warped_bgr.shape[:2]
        ignore = build_ignore_mask((h, w), border_pad,
                                   extra_polygons=extra_ignore_polygons)

        if self.collecting_background():
            self._feed_background(warped_bgr)
            if annotate_on is not None:
                cv2.putText(annotate_on, "CAPTURING BACKGROUND...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return None, np.zeros((h, w), np.uint8)

        if not self.has_background():
            if annotate_on is not None:
                cv2.putText(annotate_on, "NO BACKGROUND - press 'b' with the field empty",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return None, np.zeros((h, w), np.uint8)

        binary = _segment_bgdiff(warped_bgr, self.background_bgr, seg_thresh)
        binary = _clean_binary(binary, ignore)

        cnt = select_object_contour(binary, min_area,
                                    border_margin=border_pad)

        target = None
        if cnt is not None:
            self._miss_count = 0
            tx, ty = contour_grip_point(cnt, self.parallax_offset)
            target = self._smooth(tx, ty)
        else:
            self._miss_count += 1
            if self._miss_count > self.miss_reset_frames:
                self.reset_smoothing()        # object gone: stop the stale crosshair

        # ---------- annotation (display copy only) ----------
        if annotate_on is not None:
            annotate_on[ignore > 0] //= 2     # dim the ignored zones
            cv2.putText(annotate_on, "BG DIFF (ref OK)", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if cnt is not None and target is not None:
                cv2.drawContours(annotate_on, [cnt], -1, (0, 255, 255), 2)
                cv2.line(annotate_on, (target[0], 0), (target[0], h), (0, 0, 255), 2)
                cv2.line(annotate_on, (0, target[1]), (w, target[1]), (0, 0, 255), 2)
                cv2.circle(annotate_on, (target[0], target[1]), 12, (0, 255, 0), -1)

        return target, binary


# MAIN PROGRAM VARIABLES
CALIBRATION_FILE = 'calibration.npz'

desired_aruco_dictionary1 = "DICT_4X4_50"

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50
}

init_loc_1 = [10, 400]
init_loc_2 = [400, 400]
init_loc_3 = [400, 10]
init_loc_4 = [10, 10]

current_square_points = [init_loc_1, init_loc_2, init_loc_3, init_loc_4]
marker_location_hold = True

def main():
    start_time = time.time()

    print("[INFO] detecting '{}' markers...".format(desired_aruco_dictionary1))
    this_aruco_dictionary1 = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary1])
    this_aruco_parameters1 = cv2.aruco.DetectorParameters()

    # --- SETUP UI WINDOW ---
    cv2.namedWindow('Unknown Object Detection')
    cv2.createTrackbar("Seg Thresh", "Unknown Object Detection", 45, 255,  empty)
    cv2.createTrackbar("Min Area",   "Unknown Object Detection", 500, 5000, empty)

    detector = UnknownObjectDetector(smoothing_factor = 0.02)
    print("[INFO] Keys: 'b' capture empty-workspace background, "
          "'o' print object pos, 'q' quit.")
    print("[INFO] Press 'b' once the field is locked and the unknown object is OFF the mat.")

    # --- CAMERA & REMAP SETUP ---
    cap = cv2.VideoCapture(0)

    calibration_loaded = False
    mapx, mapy = None, None

    try:
        with np.load(CALIBRATION_FILE) as calib_data:
            # UPDATED: Using the exact keys found in your .npz file
            mtx = calib_data['camera_matrix']
            dist = calib_data['dist_coeffs']

            # Read a test frame to get resolution for the remapping matrix
            ret, test_frame = cap.read()
            if ret:
                h, w = test_frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)
                calibration_loaded = True
                print("[INFO] Camera calibration loaded and remap matrices initialized successfully.")
    except Exception as e:
        print(f"[WARNING] Could not load calibration data from '{CALIBRATION_FILE}'. Exception: {e}")
        print("[WARNING] Processing raw, distorted frames.")

    square_points = current_square_points

    while True:
        current_time = time.time()
        delay = 0

        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera.")
            break

        # Apply the fast camera undistortion remap
        if calibration_loaded:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        markers, ids = get_markers(frame, this_aruco_dictionary1, this_aruco_parameters1)
        frame_clean = frame.copy()
        left_corners, corner_ids = getMarkerCoordinates(markers, ids, 0)

        if marker_location_hold == True:
            if corner_ids is not None and len(corner_ids) > 0:
                count = 0
                for id in corner_ids:
                    if id > 3:
                        break
                    current_square_points[id] = left_corners[count]
                    count += 1
            left_corners = current_square_points
            corner_ids = [0, 1, 2, 3]

        if (start_time + delay * 1) < current_time and (start_time + delay * 2) > current_time:
            cv2.aruco.drawDetectedMarkers(frame, markers)
        if (start_time + delay * 2) < current_time:
            draw_corners(frame, left_corners)
        if (start_time + delay * 3) < current_time:
            draw_numbers(frame, left_corners, corner_ids)
        if (start_time + delay * 4) < current_time:
            show_spec(frame, left_corners)

        frame_with_square, squareFound = draw_field(frame, left_corners, corner_ids)

        obj_target = None

        # Unknown Object Detection
        if (start_time + delay * 6) < current_time:
            if squareFound:
                square_points = left_corners

            img_wrapped = four_point_transform(frame_clean, np.array(square_points))
            h, w, c = img_wrapped.shape

            seg_thresh = cv2.getTrackbarPos("Seg Thresh", "Unknown Object Detection")
            min_area   = cv2.getTrackbarPos("Min Area",   "Unknown Object Detection")

            # detect() must always receive the pristine warped frame;
            # annotations go on a separate display copy.
            img_with_object = img_wrapped.copy()
            obj_target, seg_debug = detector.detect(
                img_wrapped,
                seg_thresh=seg_thresh, min_area=max(1, min_area),
                extra_ignore_polygons=ARM_BASE_POLYGONS,
                annotate_on=img_with_object
            )
            cv2.imshow('Unknown Object Detection', img_with_object)
            cv2.imshow('Segmentation Debug', seg_debug)

        cv2.imshow('frame_with_square', frame_with_square)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('b'):
            print("[INFO] Capturing background... remove the UNKNOWN object from the field. "
                  "Anything left on the mat becomes part of the background and will be ignored.")
            detector.request_background(15)

        elif key == ord('o') and obj_target is not None:
            x_coordinate = int((obj_target[1] / h) * 600) - 300
            y_coordinate = int((obj_target[0] / w) * 400)
            print("Unknown Object Base Optical position: ", x_coordinate, ", ", y_coordinate)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # robot_control_python.home()
    main()
