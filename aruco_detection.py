#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Vision-to-Motion Pipeline:
ArUco Workspace Tracking + Foreground Object Detection + 4-DoF Robot Inverse Kinematics (IK).

Controls:
    'b' : Capture median background reference (ensure mat is clear).
    'o' : Compute target 3D base pose and solve 4-DoF Inverse Kinematics.
    'q' : Terminate pipeline.
"""

from dataclasses import dataclass
import time
import cv2
import numpy as np


# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
@dataclass(frozen=True)
class Config:
    # Calibration & Workspace
    calib_file: str = "calibration.npz"
    aruco_dict: int = cv2.aruco.DICT_4X4_50
    mat_width_m: float = 0.40    # Physical mat width along X (meters)
    mat_height_m: float = 0.60   # Physical mat height along Y (meters)

    # Robot Transformation & Geometry (Robot Base relative to Mat Top-Left)
    robot_offset_xyz: tuple = (-0.20, 0.15, 0.08)  # [X, Y, Z] offset in meters
    dh_lengths: tuple = (0.30, 0.35, 0.25, 0.10)   # [L0, L1, L2, L_grip] in meters

    # Vision & Segmentation Tuning
    border_pad_px: int = 10
    min_thickness_px: int = 12
    max_area_fraction: float = 0.50
    smoothing_factor: float = 0.05
    bg_frames_count: int = 15


# =============================================================================
# 4-DoF ROBOT KINEMATICS ENGINE
# =============================================================================
class Kinematics4DoF:
    """Analytical FK and Numerical Damped Least Squares (DLS) IK Engine."""

    @staticmethod
    def _dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d],
            [0,   0,        0,       1]
        ], dtype=float)

    @classmethod
    def forward_kinematics(cls, q: np.ndarray, lengths: tuple) -> tuple[np.ndarray, list[np.ndarray]]:
        l0, l1, l2, lg = lengths
        dh = [
            (q[0], l0, 0.0,  np.pi / 2),
            (q[1], 0.0, l1,  0.0),
            (q[2], 0.0, l2,  np.pi / 2),
            (q[3], lg,  0.0, 0.0)
        ]
        t_curr = np.eye(4)
        chain = []
        for params in dh:
            t_curr = t_curr @ cls._dh_matrix(*params)
            chain.append(t_curr)
        return t_curr, chain

    @classmethod
    def solve_ik(cls, target_xyz: np.ndarray, lengths: tuple,
                    q_seed: np.ndarray = None, max_iter: int = 150,
                    tol: float = 1e-4, lam: float = 0.02) -> tuple[np.ndarray, bool]:
        """Solves numerical IK to align the end-effector position with target_xyz."""
        q = np.array([0.1, 0.4, -0.4, 0.0] if q_seed is None else q_seed, dtype=float)
        p_target = np.asarray(target_xyz, dtype=float)

        for _ in range(max_iter):
            t_end, chain = cls.forward_kinematics(q, lengths)
            p_curr = t_end[:3, 3]
            err = p_target - p_curr

            if np.linalg.norm(err) <= tol:
                return q, True

            # Extract joint frame origins & Z-axes for geometric Jacobian
            origins = [np.zeros(3)] + [t[:3, 3] for t in chain[:-1]]
            z_axes  = [np.array([0, 0, 1])] + [t[:3, 2] for t in chain[:-1]]

            j_pos = np.column_stack([
                np.cross(z, p_curr - o) for z, o in zip(z_axes, origins)
            ])

            # DLS Pseudo-Inverse Update
            dq = j_pos.T @ np.linalg.inv(j_pos @ j_pos.T + (lam ** 2) * np.eye(3)) @ err
            q += dq

        return q, False


# =============================================================================
# WORKSPACE TRACKING & COMPUTER VISION
# =============================================================================
class WorkspaceTransformer:
    """Manages ArUco workspace detection and four-point perspective warping."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cfg.aruco_dict),
            cv2.aruco.DetectorParameters()
        )
        self.field_corners = None

    def detect_markers(self, frame: np.ndarray) -> tuple[list, list]:
        corners, ids, _ = self.detector.detectMarkers(frame)
        return corners, (ids.flatten().tolist() if ids is not None else [])

    def update_field(self, corners: list, ids: list):
        if ids and all(i in ids for i in range(4)):
            self.field_corners = [corners[ids.index(i)][0][0] for i in range(4)]

    def warp(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.field_corners is None:
            return frame, False

        pts = np.array(self.field_corners, dtype="float32")
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        rect = np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")

        (tl, tr, br, bl) = rect
        width = int(max(np.hypot(br[0] - bl[0], br[1] - bl[1]), np.hypot(tr[0] - tl[0], tr[1] - tl[1])))
        height = int(max(np.hypot(tr[0] - br[0], tr[1] - br[1]), np.hypot(tl[0] - bl[0], tl[1] - bl[1])))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        m = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, m, (width, height)), True


class ObjectDetector:
    """Background-subtracted object detector locating the topmost feature point."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bg_reference = None
        self.bg_buffer = []
        self.bg_requested = 0
        self.smoothed_pt = None

    def request_background(self):
        self.bg_buffer = []
        self.bg_requested = self.cfg.bg_frames_count

    def update_background(self, warped: np.ndarray) -> bool:
        if self.bg_requested <= 0:
            return False
        self.bg_buffer.append(warped.copy())
        if len(self.bg_buffer) >= self.bg_requested:
            self.bg_reference = np.median(np.stack(self.bg_buffer), axis=0).astype(np.uint8)
            self.bg_requested = 0
            print("[INFO] Background reference updated successfully.")
        return True

    def _get_topmost_point(self, cnt: np.ndarray, band: int = 4) -> tuple[int, int]:
        pts = cnt.reshape(-1, 2)
        y_min = int(pts[:, 1].min())
        band_pts = pts[pts[:, 1] <= y_min + band]
        return int(round(float(band_pts[:, 0].mean()))), y_min

    def detect(self, warped: np.ndarray, thresh: int, min_area: int) -> tuple[tuple | None, np.ndarray]:
        h, w = warped.shape[:2]
        if self.bg_reference is None:
            return None, np.zeros((h, w), dtype=np.uint8)

        # Background Differencing & Thresholding
        diff = cv2.absdiff(warped, cv2.resize(self.bg_reference, (w, h))).max(axis=2)
        _, binary = cv2.threshold(cv2.GaussianBlur(diff, (5, 5), 0), thresh, 255, cv2.THRESH_BINARY)

        # Apply Ignore Border
        pad = self.cfg.border_pad_px
        binary[:pad, :] = binary[-pad:, :] = binary[:, :pad] = binary[:, -pad:] = 0
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        # Filter Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt, max_area = None, 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > (self.cfg.max_area_fraction * h * w):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if x <= pad or y <= pad or (x + bw) >= (w - pad) or (y + bh) >= (h - pad):
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            if min(rw, rh) < self.cfg.min_thickness_px:
                continue
            if area > max_area:
                best_cnt, max_area = cnt, area

        if best_cnt is None:
            return None, binary

        # Compute and Smooth Topmost Point
        tx, ty = self._get_topmost_point(best_cnt)
        if self.smoothed_pt is None or np.hypot(tx - self.smoothed_pt[0], ty - self.smoothed_pt[1]) > 80:
            self.smoothed_pt = [float(tx), float(ty)]
        else:
            a = self.cfg.smoothing_factor
            self.smoothed_pt[0] = a * tx + (1 - a) * self.smoothed_pt[0]
            self.smoothed_pt[1] = a * ty + (1 - a) * self.smoothed_pt[1]

        target = (int(round(self.smoothed_pt[0])), int(round(self.smoothed_pt[1])))
        return target, binary


# =============================================================================
# PIPELINE EXECUTION ENTRYPOINT
# =============================================================================
def main():
    cfg = Config()
    transformer = WorkspaceTransformer(cfg)
    detector = ObjectDetector(cfg)

    # Initialize Video Capture & Camera Matrix Remapping
    cap = cv2.VideoCapture(0)
    mapx, mapy = None, None
    try:
        with np.load(cfg.calib_file) as data:
            mtx, dist = data["camera_matrix"], data["dist_coeffs"]
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
                print("[INFO] Undistortion calibration profile loaded.")
    except Exception as err:
        print(f"[WARNING] Calibration file unavailable ({err}). Running on raw feed.")

    cv2.namedWindow("Workspace View")
    cv2.createTrackbar("Seg Thresh", "Workspace View", 45, 255, lambda a: None)
    cv2.createTrackbar("Min Area",   "Workspace View", 500, 5000, lambda a: None)

    print("\n--- Command Keys ---")
    print(" [b] Capture background reference (remove objects from field)")
    print(" [o] Solve Inverse Kinematics for tracked object")
    print(" [q] Exit application\n")

    latest_target = None
    warped_shape = (1, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mapx is not None:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # Field Calibration & Warping
        corners, ids = transformer.detect_markers(frame)
        transformer.update_field(corners, ids)
        warped, is_warped = transformer.warp(frame)

        if is_warped:
            warped_shape = warped.shape[:2]
            if detector.update_background(warped):
                cv2.putText(warped, "RECORDING BACKGROUND...", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                thresh = cv2.getTrackbarPos("Seg Thresh", "Workspace View")
                area = cv2.getTrackbarPos("Min Area", "Workspace View")
                target, seg_debug = detector.detect(warped, thresh, area)
                latest_target = target

                # Visual Feedback
                if target is not None:
                    u, v = target
                    cv2.drawMarker(warped, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                    cv2.circle(warped, (u, v), 8, (0, 0, 255), -1)
                    cv2.putText(warped, f"TOP ({u},{v})", (u + 12, v - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("Segmentation Mask", seg_debug)
            cv2.imshow("Workspace View", warped)

        cv2.imshow("Raw Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('b'):
            print("[INFO] Capturing clean background baseline...")
            detector.request_background()
        elif key == ord('o') and latest_target is not None:
            # Map Pixel Space -> Physical Workspace Coordinates (Meters)
            h, w = warped_shape
            u, v = latest_target
            table_x = (u / w) * cfg.mat_width_m
            table_y = (v / h) * cfg.mat_height_m

            # Map Workspace Frame -> Robot Base Frame
            target_xyz = np.array([
                table_x + cfg.robot_offset_xyz[0],
                table_y + cfg.robot_offset_xyz[1],
                cfg.robot_offset_xyz[2]
            ])

            print("-" * 60)
            print(f"[TARGET] Image Feature  : u={u} px, v={v} px")
            print(f"[TARGET] Cartesian Pos  : X={target_xyz[0]:.4f}m, Y={target_xyz[1]:.4f}m, Z={target_xyz[2]:.4f}m")

            # Execute Numerical IK Solver
            q_sol, converged = Kinematics4DoF.solve_ik(target_xyz, cfg.dh_lengths)
            if converged:
                print("[IK SOLVER] Converged Successfully:")
                joint_names = ["Base Yaw", "Shoulder Pitch", "Elbow Pitch", "Gripper Roll"]
                for i, (name, val) in enumerate(zip(joint_names, q_sol)):
                    print(f"  Joint {i+1} ({name:<14}): {np.rad2deg(val):+7.2f} deg  ({val:+.4f} rad)")
            else:
                print("[IK SOLVER] Error: Pose unreachable or solver stalled.")
            print("-" * 60)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()