#!/usr/bin/env python3
"""
Real-time pose of the robotic arm (ArUco ID 1) relative to the world
origin (ArUco ID 0), from a webcam.

Prints and displays X/Y/Z in mm and Roll/Pitch/Yaw in degrees.
Press q to quit.
"""
import math
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------- settings
CALIB = "calibration.npz"
MARKER = 50.0                       # mm, printed black square edge
ORIGIN, ARM = 0, 1
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# ------------------------------------------------------------- calibration
_c = np.load(CALIB)
K = _c["camera_matrix"].astype(np.float64)
DIST = _c["dist_coeffs"].astype(np.float64)
# fx/fy/cx/cy are in PIXELS, so the camera must run at the resolution it was
# calibrated at or every distance below is scaled wrong.
CW, CH = (int(v) for v in _c["image_size"]) if "image_size" in _c else (640, 480)

# ---------------------------------------------------------------- detector
_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # big accuracy win
detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(ARUCO_DICT), _params)

# Marker corners in the marker's own frame, ordered to match detectMarkers()
_h = MARKER / 2.0
OBJ = np.array([[-_h, _h, 0], [_h, _h, 0], [_h, -_h, 0], [-_h, -_h, 0]], np.float64)


def marker_pose(corners):
    """Corners -> (rvec, tvec). Keeps whichever PnP candidate reprojects best."""
    pts = np.asarray(corners, np.float64).reshape(-1, 2)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        OBJ, pts, K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    cands = [(rvecs[i], tvecs[i]) for i in range(n)]
    ok, r, t = cv2.solvePnP(OBJ, pts, K, DIST, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        cands.append((r, t))          # rescues a case where IPPE goes singular

    def err(rt):
        p, _ = cv2.projectPoints(OBJ, rt[0], rt[1], K, DIST)
        return np.linalg.norm(p.reshape(-1, 2) - pts)
    return min(cands, key=err)


def to_matrix(rvec, tvec):
    """(rvec, tvec) -> 4x4 transform, marker frame into camera frame."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = np.asarray(tvec).ravel()
    return T


def euler_zyx(R):
    """Rotation matrix -> (roll, pitch, yaw) degrees. R = Rz(y)Ry(p)Rx(r)."""
    cp = math.hypot(R[0, 0], R[1, 0])          # == cos(pitch)
    if cp > 1e-6:
        r, p, y = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], cp), \
            math.atan2(R[1, 0], R[0, 0])
    else:                                       # gimbal lock: roll/yaw merge
        r, p, y = 0.0, math.atan2(-R[2, 0], cp), math.atan2(-R[0, 1], R[1, 1])
    return math.degrees(r), math.degrees(p), math.degrees(y)


# ------------------------------------------------------------------ camera
cap = cv2.VideoCapture(
    0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
if not cap.isOpened():
    sys.exit("Could not open the camera.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CH)
print("calibrated at %dx%d, fx=%.1f -- keep markers within ~%.0f mm"
      % (CW, CH, K[0, 0], MARKER * K[0, 0] / 110.0))

font = cv2.FONT_HERSHEY_SIMPLEX
dropped = 0
while True:
    ok, frame = cap.read()
    if not ok:
        # Tolerate the odd dropped frame, but do not spin forever if the
        # camera is unplugged or taken by another program.
        dropped += 1
        if dropped > 60:
            print("Camera stopped delivering frames.")
            break
        continue
    dropped = 0

    try:
        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        poses = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c, i in zip(corners, ids.flatten()):
                if int(i) in (ORIGIN, ARM):
                    rvec, tvec = marker_pose(c)
                    poses[int(i)] = (rvec, tvec)
                    cv2.drawFrameAxes(frame, K, DIST, rvec, tvec, MARKER * 0.6, 2)

        if ORIGIN in poses and ARM in poses:
            T0 = to_matrix(*poses[ORIGIN])
            T1 = to_matrix(*poses[ARM])

            # Pose of the arm marker in the origin marker's frame.
            # Same as inv(T0) @ T1, written out so the camera term visibly
            # cancels: inv([R|t]) == [R^T | -R^T t]
            R0 = T0[:3, :3]
            T = np.eye(4)
            T[:3, :3] = R0.T @ T1[:3, :3]
            T[:3, 3] = R0.T @ (T1[:3, 3] - T0[:3, 3])

            x, y, z = T[:3, 3]
            roll, pitch, yaw = euler_zyx(T[:3, :3])

            print("X %8.1f  Y %8.1f  Z %8.1f mm   |   R %7.1f  P %7.1f  Y %7.1f deg"
                  % (x, y, z, roll, pitch, yaw))
            cv2.putText(frame, "X %.1f  Y %.1f  Z %.1f mm" % (x, y, z),
                        (10, 25), font, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "R %.1f  P %.1f  Y %.1f deg" % (roll, pitch, yaw),
                        (10, 50), font, 0.6, (0, 255, 255), 2)
        else:
            missing = [str(m) for m in (ORIGIN, ARM) if m not in poses]
            cv2.putText(frame, "waiting for marker(s): " + ", ".join(missing),
                        (10, 25), font, 0.6, (0, 0, 255), 2)

    except cv2.error as exc:          # never let one bad frame kill the loop
        cv2.putText(frame, str(exc)[:60], (10, 25), font, 0.5, (0, 0, 255), 2)

    cv2.imshow("ArUco relative pose (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()