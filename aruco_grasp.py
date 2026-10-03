#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision-guided pick pipeline for a yaw-pitch-pitch-roll arm + gripper (5x MG996R on PCA9685).

ArUco workspace -> object footprint (centre, length, width, orientation)
-> pick the object side that fits inside the gripper -> closed-form IK + roll alignment
-> interpolated PWM stream to the Arduino.

Controls:
    'b' : Capture background reference (mat clear, arm at HOME).
    'o' : Plan and execute the grasp of the detected object.
    'r' : Release (open the gripper fully).
    'h' : Return the arm to HOME.
    'q' : Quit.

Serial packet: '<yaw,shoulder,elbow,roll,gripper>\\n' in microseconds.
Every value marked MEASURE in Config is a placeholder - replace it with your arm's numbers.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import time

import cv2
import numpy as np
import serial  # Requires: pip install pyserial


# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
@dataclass(frozen=True)
class Config:
    # ---- Camera & workspace -------------------------------------------------
    calib_file: str = "calibration.npz"
    camera_index: int = 0
    aruco_dict: int = cv2.aruco.DICT_4X4_50
    marker_ids: tuple = (0, 1, 2, 3)   # mat corners: (top-left, top-right, bottom-right, bottom-left)
    # Distances between MARKER CENTRES (not the mat edges), metres. MEASURE
    mat_width_m: float = 0.40          # bottom-left -> bottom-right marker  (mat +X)
    mat_height_m: float = 0.60         # bottom-left -> top-left marker      (mat +Y)
    px_per_m: float = 1000.0           # warped image resolution (1000 -> 1 px = 1 mm)

    # ---- Robot placement in the mat frame (MEASURE) -------------------------
    # Mat frame: origin at the bottom-left marker centre, X right, Y towards the
    # top-left marker, Z up out of the mat (right-handed).
    base_xy_in_mat_m: tuple = (0.20, -0.05)  # where the yaw axis meets the mat plane
    base_z_in_mat_m: float = 0.0             # robot frame origin height above the mat surface
    base_yaw_in_mat_deg: float = 90.0        # direction the arm points at yaw = 0, CCW from mat +X

    # ---- Arm geometry (MEASURE, metres) -------------------------------------
    l0_m: float = 0.10                 # robot frame origin -> shoulder pitch axis (vertical)
    l1_m: float = 0.14                 # shoulder axis -> elbow axis
    l2_m: float = 0.10                 # elbow axis -> roll joint
    lg_m: float = 0.11                 # roll joint -> fingertip centre (TCP)
    # "inline":        roll axis + gripper continue straight along the forearm
    # "perpendicular": gripper points 90 deg below the forearm (the old DH model)
    wrist_type: str = "inline"
    min_elbow_z_m: float = 0.02        # keep the elbow at least this high above the mat

    # ---- Gripper ------------------------------------------------------------
    gripper_max_open_m: float = 0.07   # EDIT: jaw gap when fully open (gripper_open_us)
    gripper_open_us: int = 1000        # MEASURE: pulse for the fully open jaws
    gripper_closed_us: int = 2000      # MEASURE: pulse for jaws touching (0 gap)
    grip_clearance_m: float = 0.005    # jaws must open at least this much wider than the object
    grip_squeeze_m: float = 0.005      # close this much tighter than the object
    max_tilt_deg: float = 45.0         # warn when the gripper is further than this from vertical

    # ---- Grasp --------------------------------------------------------------
    object_height_m: float = 0.05      # default object height (also the "Obj H mm" trackbar)
    grip_depth_m: float = 0.02         # fingertips go this far below the object's top
    approach_clearance_m: float = 0.05 # pre-grasp height above the object's top
    min_grasp_z_m: float = 0.01        # never put the TCP lower than this above the mat
    descend_steps: int = 10            # straight-line waypoints between pre-grasp and grasp

    # ---- Servos & serial ----------------------------------------------------
    serial_port: str = "COM3"
    baud_rate: int = 115200
    # Per joint: (us_at_zero, us_at_ref, ref_deg, us_min, us_max) - measure with calib/calib_servo.py.
    #   us_at_zero : pulse that puts the link in its joint-zero pose
    #   us_at_ref  : pulse that puts the link exactly ref_deg away from zero (+90 or -90)
    #   us_min/max : safe pulse limits (end stops / collisions)
    # Joint zero pose: yaw = arm along robot +X, shoulder = upper arm horizontal,
    #   elbow = forearm in line with upper arm, roll = jaws close along the shoulder axis.
    # Positive joint angles: yaw CCW seen from above, shoulder/elbow lift the link upwards.
    servo_calib: tuple = (
        (1500, 2500,  90.0, 500, 2500),   # Base Yaw        PLACEHOLDER
        (2500, 1500,  90.0, 500, 2500),   # Shoulder Pitch  PLACEHOLDER (1500 = upper arm vertical)
        (2500, 1500, -90.0, 500, 2500),   # Elbow Pitch     PLACEHOLDER (1500 = elbow bent 90 deg down)
        (1500, 2500,  90.0, 500, 2500),   # Gripper Roll    PLACEHOLDER
    )
    home_joints_deg: tuple = (0.0, 90.0, -30.0, 0.0)  # arm raised, out of the camera's way
    max_speed_deg_s: float = 60.0      # fastest servo speed while streaming
    stream_hz: float = 50.0

    # ---- Vision tuning (px; 1 px = 1 mm at px_per_m = 1000) -----------------
    border_pad_px: int = 10
    min_thickness_px: int = 12
    max_area_fraction: float = 0.50
    bg_frames_count: int = 15
    filter_len: int = 15               # detections averaged before planning
    filter_min: int = 5
    filter_jump_m: float = 0.02        # a centre jump larger than this restarts the average


def wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


# =============================================================================
# SERVO MAPPING
# =============================================================================
class ServoMap:
    """Joint angle <-> pulse width from a two-point calibration per servo, plus pulse limits."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        for i, (us0, us_ref, ref_deg, lo, hi) in enumerate(cfg.servo_calib):
            if ref_deg == 0 or us_ref == us0 or lo >= hi:
                raise ValueError(f"servo_calib[{i}] is invalid: {cfg.servo_calib[i]}")

    def us(self, i: int, q_rad: float) -> float:
        us0, us_ref, ref_deg, _, _ = self.cfg.servo_calib[i]
        return us0 + (us_ref - us0) * math.degrees(q_rad) / ref_deg

    def us_per_deg(self, i: int) -> float:
        us0, us_ref, ref_deg, _, _ = self.cfg.servo_calib[i]
        return abs((us_ref - us0) / ref_deg)

    def joint_ok(self, i: int, q_rad: float) -> bool:
        _, _, _, lo, hi = self.cfg.servo_calib[i]
        return lo - 0.5 <= self.us(i, q_rad) <= hi + 0.5

    def gripper_us(self, gap_m: float) -> int:
        """Linear jaw gap -> pulse. Recalibrate with a lookup table if your linkage is non-linear."""
        c = self.cfg
        f = min(max(gap_m / c.gripper_max_open_m, 0.0), 1.0)
        return int(round(c.gripper_closed_us + f * (c.gripper_open_us - c.gripper_closed_us)))


# =============================================================================
# KINEMATICS (closed form)
# =============================================================================
class ArmKinematics:
    """
    Yaw-pitch-pitch-roll arm. Joints q = [yaw, shoulder, elbow, roll] (rad).
    Both wrist types reduce to a planar 2-link problem: shoulder link l1 plus a rigid
    'virtual forearm' from the elbow to the TCP of length l2e, rotated by beta.
    """

    def __init__(self, cfg: Config, servos: ServoMap):
        self.cfg, self.servos = cfg, servos
        if cfg.wrist_type == "inline":
            self.l2e, self.beta, self.tool_angle = cfg.l2_m + cfg.lg_m, 0.0, 0.0
        elif cfg.wrist_type == "perpendicular":
            self.l2e = math.hypot(cfg.l2_m, cfg.lg_m)
            self.beta, self.tool_angle = math.atan2(cfg.lg_m, cfg.l2_m), -math.pi / 2
        else:
            raise ValueError(f"wrist_type must be 'inline' or 'perpendicular', got {cfg.wrist_type!r}")

    def fk(self, q) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (TCP xyz, approach axis, jaw closing axis, elbow xyz) in the robot frame."""
        c = self.cfg
        radial = np.array([math.cos(q[0]), math.sin(q[0]), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        pitch_axis = np.array([math.sin(q[0]), -math.cos(q[0]), 0.0])

        phi = q[1] + q[2]                 # forearm angle above horizontal
        tool = phi + self.tool_angle      # gripper axis angle above horizontal
        er, ez = c.l1_m * math.cos(q[1]), c.l0_m + c.l1_m * math.sin(q[1])
        r = er + c.l2_m * math.cos(phi) + c.lg_m * math.cos(tool)
        z = ez + c.l2_m * math.sin(phi) + c.lg_m * math.sin(tool)

        approach = math.cos(tool) * radial + math.sin(tool) * up
        closing = math.cos(q[3]) * pitch_axis + math.sin(q[3]) * np.cross(approach, pitch_axis)
        return r * radial + z * up, approach, closing, er * radial + ez * up

    def tilt_deg(self, q) -> float:
        """Angle between the gripper axis and straight down."""
        return math.degrees(math.acos(min(1.0, max(-1.0, -self.fk(q)[1][2]))))

    def ik(self, xyz) -> dict[str, np.ndarray]:
        """Valid [yaw, shoulder, elbow, 0] solutions keyed by 'up' / 'down' elbow branch."""
        c = self.cfg
        x, y, z = xyz
        r, zs = math.hypot(x, y), z - c.l0_m
        d = (r * r + zs * zs - c.l1_m ** 2 - self.l2e ** 2) / (2 * c.l1_m * self.l2e)
        if abs(d) > 1.0:
            return {}

        sols = {}
        for branch, sign in (("up", -1.0), ("down", 1.0)):
            q2e = sign * math.acos(d)
            q1 = math.atan2(zs, r) - math.atan2(self.l2e * math.sin(q2e), c.l1_m + self.l2e * math.cos(q2e))
            q = np.array([wrap(math.atan2(y, x)), wrap(q1), wrap(q2e + self.beta), 0.0])
            if not all(self.servos.joint_ok(i, q[i]) for i in range(3)):
                continue
            if self.fk(q)[3][2] < c.min_elbow_z_m:
                continue
            sols[branch] = q
        return sols

    def roll_for(self, q, close_dir: np.ndarray, prefer: float = 0.0) -> tuple[float, float] | None:
        """
        Roll that turns the jaw closing axis as close as possible to close_dir.
        Returns (roll, alignment) where alignment = |cos| of the remaining angle error
        (1.0 = perfect; it drops when the gripper is tilted and close_dir points radially).
        """
        _, a, _, _ = self.fk([q[0], q[1], q[2], 0.0])
        p = np.array([math.sin(q[0]), -math.cos(q[0]), 0.0])
        b = np.cross(a, p)
        dp, db = float(close_dir @ p), float(close_dir @ b)
        psi = math.atan2(db, dp)
        # Jaws are symmetric: psi and psi +/- 180 deg grip the same line.
        options = [wrap(psi + k * math.pi) for k in (-1, 0, 1)]
        options = [o for o in options if self.servos.joint_ok(3, o)]
        if not options:
            return None
        return min(options, key=lambda o: abs(wrap(o - prefer))), math.hypot(dp, db)


# =============================================================================
# GRASP PLANNING: choose the side that fits, then solve the path
# =============================================================================
@dataclass
class ObjectObs:
    center_mat: np.ndarray        # (x, y) metres in the mat frame
    short_m: float                # footprint width
    long_m: float                 # footprint length
    short_axis_mat: np.ndarray    # unit vector along the short side, mat frame
    box_px: np.ndarray = field(default=None, repr=False)


@dataclass
class GraspPlan:
    side: str                     # "short" or "long"
    width_m: float                # object size between the jaws
    span_m: float                 # width corrected for residual roll misalignment
    axis_mat: np.ndarray          # jaw closing axis in the mat frame (for drawing)
    path: list                    # joint waypoints, pre-grasp -> grasp
    open_m: float
    close_m: float
    tilt_deg: float


class GraspPlanner:
    def __init__(self, cfg: Config, kin: ArmKinematics):
        self.cfg, self.kin = cfg, kin
        th = math.radians(cfg.base_yaw_in_mat_deg)
        self.rot = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])  # mat -> robot

    def mat_to_robot(self, p_mat: np.ndarray) -> np.ndarray:
        return self.rot @ (np.asarray(p_mat) - np.asarray(self.cfg.base_xy_in_mat_m))

    def plan(self, obs: ObjectObs, object_height_m: float) -> tuple[GraspPlan | None, str]:
        c = self.cfg
        xy = self.mat_to_robot(obs.center_mat)
        z_grasp = max(object_height_m - c.grip_depth_m, c.min_grasp_z_m) - c.base_z_in_mat_m
        z_above = object_height_m + c.approach_clearance_m - c.base_z_in_mat_m
        long_axis = np.array([-obs.short_axis_mat[1], obs.short_axis_mat[0]])

        reasons = []
        # Narrowest side first: it leaves the most clearance inside the jaws.
        for side, width, axis in (("short", obs.short_m, obs.short_axis_mat), ("long", obs.long_m, long_axis)):
            if width + c.grip_clearance_m > c.gripper_max_open_m + 1e-9:
                reasons.append(f"{side} {width * 1000:.0f}mm too wide")
                continue
            close_dir = np.r_[self.rot @ axis, 0.0]
            plan = self._solve_path(xy, z_above, z_grasp, close_dir)
            if plan is None:
                reasons.append(f"{side}: unreachable")
                continue
            path, alignment = plan
            span = width / max(alignment, 1e-3)
            if span + c.grip_clearance_m > c.gripper_max_open_m + 1e-9:
                reasons.append(f"{side}: roll can't align ({span * 1000:.0f}mm span)")
                continue
            return GraspPlan(side, width, span, axis, path,
                             open_m=min(span + c.grip_clearance_m, c.gripper_max_open_m),
                             close_m=max(span - c.grip_squeeze_m, 0.0),
                             tilt_deg=self.kin.tilt_deg(path[-1])), "ok"
        return None, "; ".join(reasons)

    def _solve_path(self, xy, z_above, z_grasp, close_dir) -> tuple[list, float] | None:
        """Straight vertical descent, same elbow branch throughout, roll re-aligned per waypoint."""
        at_grasp = self.kin.ik((xy[0], xy[1], z_grasp))
        for branch in sorted(at_grasp, key=lambda b: self.kin.tilt_deg(at_grasp[b])):
            path, roll, alignment = [], 0.0, 1.0
            for z in np.linspace(z_above, z_grasp, self.cfg.descend_steps + 1):
                q = self.kin.ik((xy[0], xy[1], z)).get(branch)
                rolled = None if q is None else self.kin.roll_for(q, close_dir, prefer=roll)
                if rolled is None:
                    break
                roll, alignment = rolled
                q[3] = roll
                path.append(q)
            else:
                return path, alignment
        return None


# =============================================================================
# SERIAL DRIVER (PYTHON -> ARDUINO -> PCA9685)
# =============================================================================
class ArmDriver:
    """Streams interpolated servo pulses so the MG996Rs never jump at full speed."""

    def __init__(self, cfg: Config, servos: ServoMap):
        self.cfg, self.servos = cfg, servos
        self.ser = None
        try:
            self.ser = serial.Serial(cfg.serial_port, cfg.baud_rate, timeout=1)
            time.sleep(2.0)  # Wait for Arduino auto-reset
            self.ser.reset_input_buffer()
            print(f"[SERIAL] Opened {cfg.serial_port} at {cfg.baud_rate} baud.")
        except Exception as e:
            print(f"[WARNING] Serial connection to {cfg.serial_port} failed ({e}). Running in simulation mode.")
        # The Arduino sketch boots every channel at 1500 us.
        self.cur_us = np.full(4, 1500.0)
        self.cur_grip_us = 1500.0
        self.cur_q = None  # set by the first move (main() homes the arm at start-up)

    def _send(self, us: list[int], log: bool):
        packet = f"<{','.join(map(str, us))}>\n"
        if self.ser and self.ser.is_open:
            self.ser.write(packet.encode("utf-8"))
        if log:
            print(f"[{'SERIAL' if self.ser else 'SIMULATION'} TX] {packet.strip()}")

    def move(self, q, grip_us: int | None = None):
        c = self.cfg
        target = np.array([self.servos.us(i, q[i]) for i in range(4)])
        grip = self.cur_grip_us if grip_us is None else float(grip_us)
        joint_deg = [abs(target[i] - self.cur_us[i]) / self.servos.us_per_deg(i) for i in range(4)]
        duration = max(max(joint_deg) / c.max_speed_deg_s, 0.3 if grip_us else 0.0)
        steps = max(1, math.ceil(duration * c.stream_hz))
        start, start_grip = self.cur_us.copy(), self.cur_grip_us
        for k in range(1, steps + 1):
            f = k / steps
            us = start + f * (target - start)
            packet = [int(round(u)) for u in us] + [int(round(start_grip + f * (grip - start_grip)))]
            self._send(packet, log=(k == steps))
            time.sleep(1.0 / c.stream_hz)
        self.cur_us, self.cur_grip_us, self.cur_q = target, grip, np.array(q, dtype=float)

    def grip(self, grip_us: int):
        self.move(self.cur_q, grip_us)

    def home(self, grip_us: int | None = None):
        self.move(np.radians(self.cfg.home_joints_deg), grip_us)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SERIAL] Port closed.")


def execute_grasp(driver: ArmDriver, servos: ServoMap, plan: GraspPlan):
    driver.move(plan.path[0], grip_us=servos.gripper_us(plan.open_m))   # pre-grasp, jaws open
    for q in plan.path[1:]:                                              # straight descent
        driver.move(q)
    driver.move(plan.path[-1], grip_us=servos.gripper_us(plan.close_m))  # close
    time.sleep(0.3)
    for q in reversed(plan.path[:-1]):                                   # straight lift
        driver.move(q)
    driver.home()


# =============================================================================
# WORKSPACE TRACKING & COMPUTER VISION
# =============================================================================
class WorkspaceTransformer:
    """ArUco corner markers -> metric top-down view of the mat."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cfg.aruco_dict),
            cv2.aruco.DetectorParameters()
        )
        self.size = (int(round(cfg.mat_width_m * cfg.px_per_m)) + 1,
                     int(round(cfg.mat_height_m * cfg.px_per_m)) + 1)
        self.m = None  # kept from the last frame that saw all four markers (survives occlusion)

    def update(self, frame: np.ndarray):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            ids = ids.flatten().tolist()
            if all(m in ids for m in self.cfg.marker_ids):
                src = np.float32([corners[ids.index(m)][0].mean(axis=0) for m in self.cfg.marker_ids])
                w, h = self.size
                dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                self.m = cv2.getPerspectiveTransform(src, dst)

    def warp(self, frame: np.ndarray) -> np.ndarray | None:
        return None if self.m is None else cv2.warpPerspective(frame, self.m, self.size)

    def px_to_mat(self, u: float, v: float) -> np.ndarray:
        return np.array([u, self.size[1] - 1 - v]) / self.cfg.px_per_m

    def mat_to_px(self, p: np.ndarray) -> tuple[int, int]:
        return int(round(p[0] * self.cfg.px_per_m)), int(round(self.size[1] - 1 - p[1] * self.cfg.px_per_m))


class ObjectDetector:
    """Background-subtracted detector returning the object's metric footprint."""

    def __init__(self, cfg: Config, ws: WorkspaceTransformer):
        self.cfg, self.ws = cfg, ws
        self.bg_reference = None
        self.bg_buffer = []
        self.bg_requested = 0

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

    def detect(self, warped: np.ndarray, thresh: int, min_area: int) -> tuple[ObjectObs | None, np.ndarray]:
        h, w = warped.shape[:2]
        if self.bg_reference is None:
            return None, np.zeros((h, w), dtype=np.uint8)

        # Background Differencing & Thresholding
        diff = cv2.absdiff(warped, self.bg_reference).max(axis=2)
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

        # Footprint: centroid + minimum-area rectangle, measured in the mat frame
        mom = cv2.moments(best_cnt)
        center = self.ws.px_to_mat(mom["m10"] / mom["m00"], mom["m01"] / mom["m00"])
        box = cv2.boxPoints(cv2.minAreaRect(best_cnt))
        edges = [(box[1] - box[0]), (box[2] - box[1])]
        edges = [np.array([e[0], -e[1]]) / self.cfg.px_per_m for e in edges]  # px -> mat (v points down)
        short, long = sorted(edges, key=np.linalg.norm)
        obs = ObjectObs(center, float(np.linalg.norm(short)), float(np.linalg.norm(long)),
                        short / np.linalg.norm(short), box.astype(np.int32))
        return obs, binary


class ObservationFilter:
    """Median of recent detections; the side direction is averaged on the doubled angle (180 deg periodic)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.buf = deque(maxlen=cfg.filter_len)

    def add(self, obs: ObjectObs | None):
        if obs is None or (self.buf and np.linalg.norm(obs.center_mat - self.buf[-1].center_mat) > self.cfg.filter_jump_m):
            self.buf.clear()
        if obs is not None:
            self.buf.append(obs)

    def estimate(self) -> ObjectObs | None:
        if len(self.buf) < self.cfg.filter_min:
            return None
        ang = [math.atan2(o.short_axis_mat[1], o.short_axis_mat[0]) for o in self.buf]
        mean = 0.5 * math.atan2(sum(math.sin(2 * a) for a in ang), sum(math.cos(2 * a) for a in ang))
        return ObjectObs(np.median([o.center_mat for o in self.buf], axis=0),
                         float(np.median([o.short_m for o in self.buf])),
                         float(np.median([o.long_m for o in self.buf])),
                         np.array([math.cos(mean), math.sin(mean)]), self.buf[-1].box_px)


def draw_overlay(img: np.ndarray, ws: WorkspaceTransformer, obs: ObjectObs, plan: GraspPlan | None, msg: str):
    cv2.drawContours(img, [obs.box_px], 0, (0, 255, 0), 2)
    cu, cv_ = ws.mat_to_px(obs.center_mat)
    cv2.circle(img, (cu, cv_), 5, (0, 0, 255), -1)
    cv2.putText(img, f"{obs.long_m * 1000:.0f} x {obs.short_m * 1000:.0f} mm", (cu + 10, cv_ - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if plan is None:
        cv2.putText(img, f"NO GRASP: {msg}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return
    half = plan.axis_mat * plan.width_m / 2
    cv2.line(img, ws.mat_to_px(obs.center_mat - half), ws.mat_to_px(obs.center_mat + half), (255, 255, 0), 3)
    color = (0, 165, 255) if plan.tilt_deg > ws.cfg.max_tilt_deg else (255, 255, 0)
    cv2.putText(img, f"GRIP {plan.side} side {plan.width_m * 1000:.0f}mm | roll {math.degrees(plan.path[-1][3]):+.0f}"
                     f" | tilt {plan.tilt_deg:.0f}deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# =============================================================================
# PIPELINE EXECUTION ENTRYPOINT
# =============================================================================
def main():
    cfg = Config()
    servos = ServoMap(cfg)
    kin = ArmKinematics(cfg, servos)
    planner = GraspPlanner(cfg, kin)
    ws = WorkspaceTransformer(cfg)
    detector = ObjectDetector(cfg, ws)
    obs_filter = ObservationFilter(cfg)
    driver = ArmDriver(cfg, servos)
    driver.home(servos.gripper_us(cfg.gripper_max_open_m))

    # Initialize Video Capture & Camera Matrix Remapping
    cap = cv2.VideoCapture(cfg.camera_index)
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
    cv2.createTrackbar("Min Area", "Workspace View", 500, 5000, lambda a: None)
    cv2.createTrackbar("Obj H mm", "Workspace View", int(cfg.object_height_m * 1000), 250, lambda a: None)

    print("\n--- Command Keys ---")
    print(" [b] Capture background (mat clear, arm at home)")
    print(" [o] Grasp the detected object")
    print(" [r] Release object    [h] Home    [q] Quit\n")

    plan, obs = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if mapx is not None:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        ws.update(frame)
        warped = ws.warp(frame)
        if warped is not None:
            if detector.update_background(warped):
                cv2.putText(warped, "RECORDING BACKGROUND...", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                thresh = cv2.getTrackbarPos("Seg Thresh", "Workspace View")
                area = cv2.getTrackbarPos("Min Area", "Workspace View")
                height_m = cv2.getTrackbarPos("Obj H mm", "Workspace View") / 1000.0
                raw_obs, seg_debug = detector.detect(warped, thresh, area)
                obs_filter.add(raw_obs)
                obs = obs_filter.estimate()
                plan, msg = planner.plan(obs, height_m) if obs is not None else (None, "")
                if obs is not None:
                    draw_overlay(warped, ws, obs, plan, msg)
                cv2.imshow("Segmentation Mask", seg_debug)
            cv2.imshow("Workspace View", warped)

        cv2.imshow("Raw Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('b'):
            print("[INFO] Capturing clean background baseline...")
            detector.request_background()
        elif key == ord('r'):
            driver.grip(servos.gripper_us(cfg.gripper_max_open_m))
        elif key == ord('h'):
            driver.home()
        elif key == ord('o'):
            print("-" * 65)
            if obs is None:
                print("[GRASP] No stable object detection yet.")
            elif plan is None:
                print(f"[GRASP] Cannot grasp: {msg}")
            else:
                print(f"[OBJECT] centre (mat) = ({obs.center_mat[0]:.3f}, {obs.center_mat[1]:.3f}) m, "
                      f"footprint {obs.long_m * 1000:.0f} x {obs.short_m * 1000:.0f} mm, height {height_m * 1000:.0f} mm")
                print(f"[GRASP] Gripping the {plan.side} side ({plan.width_m * 1000:.0f} mm, "
                      f"effective span {plan.span_m * 1000:.0f} mm, max open {cfg.gripper_max_open_m * 1000:.0f} mm)")
                for name, val in zip(("Base Yaw", "Shoulder Pitch", "Elbow Pitch", "Gripper Roll"), plan.path[-1]):
                    print(f"  {name:<14}: {math.degrees(val):+7.2f} deg")
                if plan.tilt_deg > cfg.max_tilt_deg:
                    print(f"[WARNING] Gripper is {plan.tilt_deg:.0f} deg from vertical at the grasp pose.")
                execute_grasp(driver, servos, plan)
                obs_filter.add(None)
                for _ in range(5):  # drop frames buffered while the arm was moving
                    cap.grab()
            print("-" * 65)

    cap.release()
    cv2.destroyAllWindows()
    driver.home()
    driver.close()


if __name__ == "__main__":
    main()
