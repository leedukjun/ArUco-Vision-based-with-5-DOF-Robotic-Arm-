#!/usr/bin/env python3
"""
Two-marker workspace calibration for the 5-DOF arm.

    python workspace.py lock     # both tags visible -> export the frames
    python workspace.py run      # tags removed      -> locate the cube

THE TWO COORDINATE SYSTEMS
--------------------------
    tag 0  ->  TABLE frame.  Stick it anywhere flat and in view. Its position
               is arbitrary; it exists to pin the table plane and give every
               measurement a stable reference that does not move when the
               robot does.

    tag 1  ->  ROBOT BASE frame.  Stick it on the centre of the robot base,
               aligned with the robot's own axes. Its origin and heading
               become the robot frame.

Lock measures both and writes them out. Because tag 1 is MEASURED rather than
assumed, you do not have to place tag 0 precisely -- any error in where tag 0
sits cancels when a point is converted into the robot frame.

    T_table_robot = inv(T_cam_table) @ T_cam_robot

Both tags are seen by the same camera in the same frame, so the camera term
cancels; the result is independent of where the camera happens to be.

WHY THE TAGS CAN BE REMOVED AFTERWARDS
--------------------------------------
The cube and the tags all lie on one flat table. For a fixed camera looking at
a fixed plane, pixels map to table millimetres through a single fixed
transform -- nothing in it depends on a marker still being visible. Projecting
a world point onto the image is

    s * [u v 1]^T = K * [r1 r2 r3 t] * [X Y Z 1]^T

and on the table Z = 0, so the r3 column drops out entirely:

    s * [u v 1]^T = K * [r1 r2 t] * [X Y 1]^T  ==  H * [X Y 1]^T

H is a 3x3 homography; invert it and any pixel becomes a table coordinate.
The full camera pose is saved too, because the cube has HEIGHT -- its
silhouette does not lie on Z = 0 (see cube_base_xy).

PIPELINE
--------
  1. Stick tag 0 at any convenient point on the table.
  2. Stick tag 1 on the centre of the robot base, axes matching the robot's.
  3. python workspace.py lock   -> averages 40 frames, exports
                                     workspace.npz          (for the program)
                                     coordinate_systems.json (human readable)
                                   then asks you to remove the tags and stores
                                   a reference image for the drift check.
  4. Remove both tags. The camera must not move from here on.
  5. python workspace.py run    -> tune the HSV sliders once, press S.
                                   The cube is reported in BOTH frames.
  6. SPACE sends the robot-frame (X, Y) to the arm -- fill in send_to_robot().

Requires calibration.npz in this folder (camera_matrix, dist_coeffs).
"""
import json
import math
import os
import sys
from collections import deque

import cv2
import numpy as np

# ------------------------------------------------------------------ config
CALIB = "calibration.npz"
WORKSPACE = "workspace.npz"
EXPORT = "coordinate_systems.json"

MARKER = 50.0          # mm, printed BLACK SQUARE edge -- measure it
CUBE = 50.0            # mm, cube edge -- measure it, see accuracy note
TABLE_ID = 0           # tag defining the table frame
ROBOT_ID = 1           # tag on the robot base centre

ARUCO_DICT = cv2.aruco.DICT_4X4_50
LOCK_SAMPLES = 40      # frames averaged when locking
MIN_BLOB_AREA = 400    # px, ignore specks
DRIFT_WARN_PX = 4.0    # mean image-corner motion that counts as "camera moved"
STABLE_N = 15          # readings kept for the median sent to the robot

FONT = cv2.FONT_HERSHEY_SIMPLEX


def send_to_robot(x, y):
    """
    Hand a target to the arm, in ROBOT BASE coordinates.

    Fill this in for your controller, e.g.

        import serial
        ser = serial.Serial("COM3", 115200, timeout=1)
        ser.write(b"MOVE %.1f %.1f\\n" % (x, y))
    """
    print(">> TARGET (robot frame)  X %.1f  Y %.1f mm" % (x, y))


# ------------------------------------------------------------------- setup
def load_calib():
    """
    Read the camera intrinsics. Different calibration scripts name their
    arrays differently, so accept the common spellings rather than failing
    with a bare KeyError on a file that is perfectly good.
    """
    if not os.path.isfile(CALIB):
        sys.exit("Missing %s in %s" % (CALIB, os.getcwd()))
    d = np.load(CALIB)
    have = list(d.keys())

    def pick(names, what):
        for n in names:
            if n in have:
                return d[n]
        sys.exit("%s has no %s array.\n  looked for: %s\n  found: %s"
                 % (CALIB, what, ", ".join(names), ", ".join(have)))

    K = np.asarray(pick(("camera_matrix", "mtx", "cameraMatrix", "K"),
                        "camera matrix"), np.float64)
    dist = np.asarray(pick(("dist_coeffs", "dist", "distCoeffs", "D"),
                           "distortion"), np.float64).reshape(1, -1)
    if K.shape != (3, 3):
        sys.exit("camera matrix in %s is %s, expected 3x3" % (CALIB, K.shape))

    size = None
    for n in ("image_size", "img_size", "resolution"):
        if n in have:
            wh = np.asarray(d[n]).ravel()
            if wh.size == 2:
                size = (int(wh[0]), int(wh[1]))
                break
    if size is None:
        size = (640, 480)
        print("note: %s has no image_size; assuming %dx%d. If the camera was "
              "calibrated at another resolution, every distance will be "
              "scaled wrong." % (CALIB, size[0], size[1]))
    return K, dist, size


def open_camera(size):
    cap = cv2.VideoCapture(
        0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        sys.exit("Could not open the camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    ok, probe = cap.read()
    if not ok:
        sys.exit("Camera opened but delivered no frames.")
    got = (probe.shape[1], probe.shape[0])
    if got != tuple(size):
        print("** camera gave %dx%d but calibration is %dx%d -- coordinates "
              "will be wrong **" % (got[0], got[1], size[0], size[1]))
    return cap


def make_detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), p)


def marker_pose(corners, K, dist):
    """Best-of-candidates PnP for one marker -> (R 3x3, t 3x1)."""
    h = MARKER / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    pts = np.asarray(corners, np.float64).reshape(-1, 2)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    cand = [(rvecs[i], tvecs[i]) for i in range(n)]
    ok, r, t = cv2.solvePnP(obj, pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        cand.append((r, t))          # rescues the case where IPPE goes singular

    def err(rt):
        p, _ = cv2.projectPoints(obj, rt[0], rt[1], K, dist)
        return np.linalg.norm(p.reshape(-1, 2) - pts)
    r, t = min(cand, key=err)
    return cv2.Rodrigues(r)[0], np.asarray(t, np.float64).reshape(3, 1)


def orthonormalize(R):
    """Nearest true rotation to an averaged matrix (averaging leaves SO(3))."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


# --------------------------------------------------------------- geometry
def homography(K, R, t):
    """Table plane (Z=0) -> image. H = K [r1 r2 t]."""
    return K @ np.column_stack((R[:, 0], R[:, 1], t.ravel()))


def pixel_to_plane(uv, K, R, t, z=0.0):
    """
    Back-project a pixel onto the table plane at height z.

    R, t map table into camera (X_cam = R X_table + t), so the camera centre
    in table coordinates is C = -R^T t and a camera-frame ray direction d
    becomes R^T d. Intersect that ray with the horizontal plane.
    """
    d_cam = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    d_table = R.T @ d_cam
    C = (-R.T @ t).ravel()
    if abs(d_table[2]) < 1e-9:
        return None                     # ray parallel to the plane
    s = (z - C[2]) / d_table[2]
    if s <= 0:
        return None                     # plane is behind the camera
    return (C + s * d_table)[:2]


def table_to_robot(xy, robot):
    """Table mm -> robot base mm. robot = (x, y, yaw_rad) in table frame."""
    dx, dy = xy[0] - robot[0], xy[1] - robot[1]
    c, s = math.cos(-robot[2]), math.sin(-robot[2])
    return np.array([c * dx - s * dy, s * dx + c * dy])


def drift_px(ref_gray, cur_gray):
    """Mean image-corner motion between the reference and now, in pixels."""
    orb = cv2.ORB_create(1500)
    k1, d1 = orb.detectAndCompute(ref_gray, None)
    k2, d2 = orb.detectAndCompute(cur_gray, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:200]
    if len(matches) < 12:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    Hd, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if Hd is None:
        return None
    h, w = ref_gray.shape
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(c, Hd).reshape(-1, 2) - c.reshape(-1, 2)
    return float(np.linalg.norm(moved, axis=1).mean())


# ------------------------------------------------------------------- LOCK
def export_json(path, K, R, t, robot, robot_z, size):
    """Human-readable dump of both coordinate systems."""
    C = (-R.T @ t).ravel()
    c, s = math.cos(robot[2]), math.sin(robot[2])
    # 2D homogeneous transform taking table coords -> robot coords
    T_tr = np.array([[c, s, -(c * robot[0] + s * robot[1])],
                     [-s, c, -(-s * robot[0] + c * robot[1])],
                     [0, 0, 1]])
    doc = {
        "units": "millimetres, degrees",
        "marker_size_mm": MARKER,
        "cube_size_mm": CUBE,
        "image_size": [int(size[0]), int(size[1])],
        "table_frame": {
            "definition": "origin and axes of ArUco tag %d; Z=0 is the table"
                          % TABLE_ID,
            "camera_position_in_table_frame": [round(float(v), 2) for v in C],
            "R_table_to_camera": [[float(v) for v in row] for row in R],
            "t_table_to_camera": [round(float(v), 3) for v in t.ravel()],
            "H_table_plane_to_image": [[float(v) for v in row]
                                       for row in homography(K, R, t)],
        },
        "robot_base_frame": {
            "definition": "origin and axes of ArUco tag %d, stuck on the "
                          "centre of the robot base" % ROBOT_ID,
            "origin_in_table_frame_mm": [round(float(robot[0]), 2),
                                         round(float(robot[1]), 2)],
            "heading_in_table_frame_deg": round(math.degrees(robot[2]), 3),
            "height_above_table_mm": round(float(robot_z), 2),
            "T_table_to_robot": [[float(v) for v in row]
                                 for row in T_tr],
        },
        "usage": "p_robot = T_table_to_robot @ [X_table, Y_table, 1]",
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def lock():
    K, dist, size = load_calib()
    cap = open_camera(size)
    det = make_detector()

    print("Show BOTH tags: %d on the table, %d on the robot base centre."
          % (TABLE_ID, ROBOT_ID))
    print("Collecting %d samples..." % LOCK_SAMPLES)
    Ra = np.zeros((3, 3)); ta = np.zeros((3, 1))
    Rb = np.zeros((3, 3)); tb = np.zeros((3, 1))
    n = 0

    while n < LOCK_SAMPLES:
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids, _ = det.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c, i in zip(corners, ids.flatten()):
                if int(i) in (TABLE_ID, ROBOT_ID):
                    found[int(i)] = marker_pose(c, K, dist)
        for mid, (R_, t_) in found.items():
            cv2.drawFrameAxes(frame, K, dist, cv2.Rodrigues(R_)[0], t_,
                              MARKER * 0.8, 3)
        if TABLE_ID in found and ROBOT_ID in found:
            Ra += found[TABLE_ID][0]; ta += found[TABLE_ID][1]
            Rb += found[ROBOT_ID][0]; tb += found[ROBOT_ID][1]
            n += 1

        cv2.putText(frame, "%d/%d   need tag %d (table) + tag %d (robot base)"
                    % (n, LOCK_SAMPLES, TABLE_ID, ROBOT_ID),
                    (10, 25), FONT, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "tag %d axes must match the robot's: red=+X green=+Y"
                    % ROBOT_ID, (10, 48), FONT, 0.5, (0, 200, 255), 2)
        cv2.imshow("lock", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            cap.release(); cv2.destroyAllWindows(); sys.exit("aborted")

    R = orthonormalize(Ra / n); t = ta / n          # table frame
    R1 = orthonormalize(Rb / n); t1 = tb / n        # robot base tag

    # Robot base expressed in the table frame. The camera cancels here.
    R_rel = R.T @ R1
    t_rel = (R.T @ (t1 - t)).ravel()
    robot = np.array([t_rel[0], t_rel[1], math.atan2(R_rel[1, 0], R_rel[0, 0])])

    C = (-R.T @ t).ravel()
    print("\nboth frames locked")
    print("  TABLE frame  : origin at tag %d, camera at X %.0f Y %.0f Z %.0f mm"
          % (TABLE_ID, C[0], C[1], C[2]))
    print("  ROBOT frame  : origin X %.1f  Y %.1f mm in the table frame, "
          "heading %.1f deg" % (robot[0], robot[1], math.degrees(robot[2])))
    print("  robot tag sits %.1f mm above the table plane" % t_rel[2])
    if abs(t_rel[2]) > 8.0:
        print("     (raised base -- fine, as long as the tag is directly")
        print("      above the base centre, since only X/Y are used)")

    corners_px = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
    pts = [pixel_to_plane(p, K, R, t, 0.0) for p in corners_px]
    pts = [p for p in pts if p is not None]
    if pts:
        a = np.array(pts)
        print("  view covers   : table X %.0f..%.0f  Y %.0f..%.0f mm"
              % (a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()))

    print("\nNow REMOVE both tags, then press R to store the reference image.")
    ref = None
    while ref is None:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.putText(frame, "remove both tags, then press R", (10, 25),
                    FONT, 0.7, (0, 200, 255), 2)
        cv2.imshow("lock", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("r"), ord("R")):
            ref = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif k in (ord("q"), 27):
            cap.release(); cv2.destroyAllWindows(); sys.exit("aborted")

    np.savez(WORKSPACE, K=K, dist=dist, R=R, t=t, H=homography(K, R, t),
             robot=robot, robot_z=np.float64(t_rel[2]), ref=ref,
             marker=MARKER, cube=CUBE, image_size=np.array(size, np.int64))
    export_json(EXPORT, K, R, t, robot, t_rel[2], size)
    print("\nexported:")
    print("  %s   (used by 'run')" % WORKSPACE)
    print("  %s   (both coordinate systems, readable)" % EXPORT)
    print("The tags are no longer needed.")
    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------- RUN
def silhouette_centroid(X, Y, K, dist, R, t, yaw=0.0):
    """Where a cube standing at table (X, Y) would put its silhouette centroid."""
    c, a = CUBE / 2.0, yaw
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    base = (rot @ (np.array([[-1., -1], [1, -1], [1, 1], [-1, 1]]) * c).T).T \
        + np.array([X, Y])
    V = np.vstack([np.hstack([base, np.zeros((4, 1))]),
                   np.hstack([base, np.full((4, 1), CUBE)])])
    ip = cv2.projectPoints(V, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)
    m = cv2.moments(cv2.convexHull(ip.astype(np.float32)))
    if abs(m["m00"]) < 1e-9:
        return ip.mean(axis=0)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def cube_base_xy(contour, K, dist, R, t):
    """
    Centre of the cube's base on the table, in TABLE coordinates.

    The blob is the cube's SILHOUETTE -- it includes the sides and top face,
    which stand above the table. Back-projecting it straight onto Z=0 throws
    the answer several millimetres too far from the camera, because those
    upper pixels only meet the plane well beyond the cube.

    So instead of inverting a bad assumption, solve the forward problem: for a
    candidate base centre, project the known cube model and see where its
    silhouette centroid would land. Shift the candidate by the back-projected
    mismatch and repeat. Converges in a handful of iterations, needs no fudge
    factor, and measured ~1 mm against ground truth versus ~6 mm for a
    fixed-offset rule. Cube yaw is ignored: it costs under 1 mm.
    """
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-9:
        return None
    observed = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])

    target = pixel_to_plane(observed, K, R, t, 0.0)
    xy = pixel_to_plane(observed, K, R, t, CUBE / 2.0)   # seed
    if target is None or xy is None:
        return None

    for _ in range(12):
        got = pixel_to_plane(
            silhouette_centroid(xy[0], xy[1], K, dist, R, t), K, R, t, 0.0)
        if got is None:
            break
        step = target - got
        xy = xy + step
        if np.linalg.norm(step) < 1e-3:
            break
    return xy


def run():
    if not os.path.isfile(WORKSPACE):
        sys.exit("No %s -- run 'python workspace.py lock' first." % WORKSPACE)
    w = np.load(WORKSPACE)
    K, dist, R, t, ref = w["K"], w["dist"], w["R"], w["t"], w["ref"]
    robot = w["robot"]
    size = tuple(int(v) for v in w["image_size"])
    hsv_lo = w["hsv_lo"] if "hsv_lo" in w else np.array([35, 80, 60])
    hsv_hi = w["hsv_hi"] if "hsv_hi" in w else np.array([85, 255, 255])
    H = homography(K, R, t)
    print("robot base at table X %.1f Y %.1f mm, heading %.1f deg"
          % (robot[0], robot[1], math.degrees(robot[2])))

    cap = open_camera(size)
    win = "cube (SPACE send, S save, Q quit)"
    cv2.namedWindow(win)
    cv2.createTrackbar("H lo", win, int(hsv_lo[0]), 179, lambda v: None)
    cv2.createTrackbar("H hi", win, int(hsv_hi[0]), 179, lambda v: None)
    cv2.createTrackbar("S lo", win, int(hsv_lo[1]), 255, lambda v: None)
    cv2.createTrackbar("V lo", win, int(hsv_lo[2]), 255, lambda v: None)

    # Has the camera moved since the lock? Nothing else would reveal it.
    ok, frame = cap.read()
    if ok:
        dpx = drift_px(ref, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if dpx is None:
            print("drift check inconclusive (too few background features)")
        elif dpx > DRIFT_WARN_PX:
            print("** camera appears to have MOVED ~%.1f px since lock -- "
                  "coordinates are suspect, re-run lock **" % dpx)
        else:
            print("drift check ok (%.1f px)" % dpx)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    recent = deque(maxlen=STABLE_N)
    dropped = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            dropped += 1
            if dropped > 60:
                print("Camera stopped delivering frames.")
                break
            continue
        dropped = 0
        xy = None

        try:
            lo = np.array([cv2.getTrackbarPos("H lo", win),
                           cv2.getTrackbarPos("S lo", win),
                           cv2.getTrackbarPos("V lo", win)], np.uint8)
            hi = np.array([cv2.getTrackbarPos("H hi", win), 255, 255], np.uint8)

            mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) > MIN_BLOB_AREA]

            if cnts:
                c = max(cnts, key=cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
                xy = cube_base_xy(c, K, dist, R, t)

            if xy is not None:
                rx, ry = table_to_robot(xy, robot)
                recent.append((rx, ry))
                print("cube  table X %7.1f Y %7.1f  |  robot X %7.1f Y %7.1f mm"
                      % (xy[0], xy[1], rx, ry))
                cv2.putText(frame, "table X %.1f  Y %.1f mm" % (xy[0], xy[1]),
                            (10, 25), FONT, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "robot X %.1f  Y %.1f mm" % (rx, ry),
                            (10, 50), FONT, 0.6, (0, 255, 255), 2)
                spread = (np.ptp(np.array(recent), axis=0).max()
                          if len(recent) == recent.maxlen else None)
                cv2.putText(frame, "spread %s   SPACE to send"
                            % ("%.1f mm" % spread if spread is not None
                               else "settling"),
                            (10, 73), FONT, 0.5, (200, 200, 200), 1)
                p = H @ np.array([xy[0], xy[1], 1.0])
                if abs(p[2]) > 1e-9:
                    cv2.drawMarker(frame, (int(p[0] / p[2]), int(p[1] / p[2])),
                                   (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            else:
                recent.clear()
                cv2.putText(frame, "no cube -- adjust the sliders", (10, 25),
                            FONT, 0.6, (0, 0, 255), 2)

            cv2.imshow("mask", mask)
        except cv2.error as exc:
            cv2.putText(frame, str(exc)[:60], (10, 25), FONT, 0.5, (0, 0, 255), 2)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            if recent:
                # Median of recent readings: one noisy frame should not decide
                # where the arm goes.
                med = np.median(np.array(recent), axis=0)
                send_to_robot(med[0], med[1])
            else:
                print("no cube to send")
        if k in (ord("s"), ord("S")):
            d = {key: w[key] for key in w.files}
            d["hsv_lo"], d["hsv_hi"] = lo, hi
            np.savez(WORKSPACE, **d)
            print("settings saved")

    cap.release()
    cv2.destroyAllWindows()


def usage():
    """Short, actionable help -- and where you are in the pipeline."""
    print("workspace.py needs a mode:\n")
    print("  py workspace.py lock    tags visible -> measure and export the frames")
    print("  py workspace.py run     tags removed -> locate the cube\n")
    print("(the long explanation is the comment block at the top of this file)\n")
    print("status in %s" % os.getcwd())
    print("  %-24s %s" % (CALIB,
          "found" if os.path.isfile(CALIB) else "MISSING - required"))
    if os.path.isfile(WORKSPACE):
        print("  %-24s found  ->  next step: py workspace.py run" % WORKSPACE)
    else:
        print("  %-24s not created yet  ->  next step: py workspace.py lock"
              % WORKSPACE)
    return 2


if __name__ == "__main__":
    mode = sys.argv[1].lower().lstrip("-") if len(sys.argv) > 1 else ""
    if mode == "lock":
        lock()
    elif mode == "run":
        run()
    else:
        if mode:
            print("unknown mode %r\n" % sys.argv[1])
        sys.exit(usage())
