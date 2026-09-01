from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd


# ============================================================
# STAMPEDE DETECTION V3.3
# ============================================================
#
# Single-file pipeline:
#
# Video
#   ↓
# YOLOv8 Head Detection
#   ↓
# BoT-SORT Tracking
#   ↓
# Speed / Acceleration / Direction
#   ↓
# Rapid Motion
#   ↓
# Optical Flow
#   ↓
# Density
#   ↓
# Normal Behaviour Baseline
#   ↓
# Risk Score
#   ↓
# NORMAL / ABNORMAL / HIGH_RISK
#   ↓
# Persistent High Risk
#   ↓
# Event Grouping
#   ↓
# CSV + Team JSON + Plot
#
# Every execution creates a NEW timestamped output folder.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class StampedeConfig:

    # Minimum number of frames a tracking ID must appear in.
    min_track_length: int = 15

    # Smoothing window.
    smoothing_window: int = 7

    # First part of the video used as normal baseline.
    baseline_seconds: float = 10.0

    # Optical-flow threshold.
    flow_magnitude_threshold: float = 0.5

    # Motion entropy bins.
    entropy_bins: int = 36

    # Rapid movement detection.
    rapid_motion_multiplier: float = 1.5
    minimum_rapid_speed: float = 15.0

    # Behaviour states.
    abnormal_threshold: float = 0.55
    high_risk_threshold: float = 0.70

    # High-risk persistence.
    persistence_seconds: float = 1.5

    # Minimum event duration.
    minimum_abnormal_duration: float = 0.30

    # Maximum gap between abnormal frames in one event.
    event_gap_seconds: float = 1.0


# ============================================================
# DETECTOR
# ============================================================

class StampedeDetector:

    def __init__(
        self,
        fps: float,
        config: Optional[StampedeConfig] = None,
    ):

        if fps <= 0:
            raise ValueError(
                "FPS must be greater than zero."
            )

        self.fps = float(fps)

        self.config = (
            config
            if config is not None
            else StampedeConfig()
        )

        self.previous_gray = None

        self.baseline = None

    # ========================================================
    # SPEED
    # ========================================================

    def calculate_speed(
        self,
        tracking_data: pd.DataFrame,
    ) -> pd.DataFrame:

        df = tracking_data.copy()

        df = df.sort_values(
            ["track_id", "frame"]
        )

        dx = (
            df.groupby("track_id")["x"]
            .diff()
        )

        dy = (
            df.groupby("track_id")["y"]
            .diff()
        )

        distance = np.sqrt(
            dx ** 2 +
            dy ** 2
        )

        df["displacement"] = (
            distance.fillna(0.0)
        )

        df["speed"] = (
            df["displacement"]
            * self.fps
        )

        return df

    # ========================================================
    # ACCELERATION
    # ========================================================

    def calculate_acceleration(
        self,
        tracking_data: pd.DataFrame,
    ) -> pd.DataFrame:

        df = tracking_data.copy()

        df = df.sort_values(
            ["track_id", "frame"]
        )

        df["acceleration"] = (
            df.groupby("track_id")["speed"]
            .diff()
            .fillna(0.0)
            * self.fps
        )

        df["acceleration"] = (
            df["acceleration"]
            .abs()
        )

        window = (
            self.config.smoothing_window
        )

        df["acceleration_smoothed"] = (
            df.groupby("track_id")[
                "acceleration"
            ]
            .transform(
                lambda x:
                x.rolling(
                    window=window,
                    min_periods=1,
                    center=True,
                ).median()
            )
        )

        return df

    # ========================================================
    # DIRECTION
    # ========================================================

    def calculate_direction(
        self,
        tracking_data: pd.DataFrame,
    ) -> pd.DataFrame:

        df = tracking_data.copy()

        df = df.sort_values(
            ["track_id", "frame"]
        )

        dx = (
            df.groupby("track_id")["x"]
            .diff()
        )

        dy = (
            df.groupby("track_id")["y"]
            .diff()
        )

        df["direction_degrees"] = (
            np.degrees(
                np.arctan2(
                    dy,
                    dx,
                )
            )
        )

        return df

    # ========================================================
    # RAPID MOTION
    # ========================================================

    def calculate_rapid_motion(
        self,
        tracking_data: pd.DataFrame,
    ) -> pd.DataFrame:

        df = tracking_data.copy()

        valid_speed = (
            df["speed"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if valid_speed.empty:

            reference = 0.0

        else:

            reference = float(
                valid_speed.quantile(0.75)
            )

        threshold = max(
            reference
            * self.config.rapid_motion_multiplier,
            self.config.minimum_rapid_speed,
        )

        df[
            "rapid_motion_threshold"
        ] = threshold

        df[
            "rapid_motion"
        ] = (
            df["speed"]
            > threshold
        )

        print(
            f"Rapid-motion threshold: "
            f"{threshold:.2f} px/s"
        )

        return df

    # ========================================================
    # DENSITY
    # ========================================================

    def calculate_density(
        self,
        tracking_data: pd.DataFrame,
    ) -> pd.DataFrame:

        density = (
            tracking_data
            .groupby("frame")["track_id"]
            .nunique()
            .rename("head_count_density")
            .reset_index()
        )

        density[
            "density_change"
        ] = (
            density[
                "head_count_density"
            ]
            .diff()
            .fillna(0.0)
        )

        density[
            "density_change_smoothed"
        ] = (
            density[
                "density_change"
            ]
            .rolling(
                window=self.config.smoothing_window,
                min_periods=1,
                center=True,
            )
            .median()
            .fillna(0.0)
        )

        return density[
            [
                "frame",
                "head_count_density",
                "density_change",
                "density_change_smoothed",
            ]
        ]

    # ========================================================
    # DIRECTION CONSISTENCY
    # ========================================================

    @staticmethod
    def calculate_direction_consistency(
        tracking_data: pd.DataFrame,
    ) -> pd.DataFrame:

        rows = []

        for frame_id, group in (
            tracking_data.groupby("frame")
        ):

            directions = (
                group[
                    "direction_degrees"
                ]
                .dropna()
                .values
            )

            if len(directions) == 0:

                consistency = 0.0

            else:

                radians = np.radians(
                    directions
                )

                mean_x = np.mean(
                    np.cos(radians)
                )

                mean_y = np.mean(
                    np.sin(radians)
                )

                consistency = float(
                    np.sqrt(
                        mean_x ** 2
                        +
                        mean_y ** 2
                    )
                )

            rows.append(
                {
                    "frame":
                        int(frame_id),

                    "direction_consistency":
                        consistency,
                }
            )

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # OPTICAL FLOW
    # ========================================================

    def calculate_optical_flow(
        self,
        frame: Optional[np.ndarray],
    ) -> dict:

        empty = {

            "optical_flow_magnitude":
                0.0,

            "optical_flow_p95":
                0.0,

            "flow_divergence":
                0.0,

            "motion_entropy":
                0.0,

            "optical_flow_direction":
                0.0,
        }

        if frame is None:

            return empty

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        if self.previous_gray is None:

            self.previous_gray = gray

            return empty

        flow = cv2.calcOpticalFlowFarneback(

            self.previous_gray,
            gray,

            None,

            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

        u = flow[..., 0]

        v = flow[..., 1]

        magnitude = np.sqrt(
            u ** 2 +
            v ** 2
        )

        angle = np.arctan2(
            v,
            u,
        )

        valid = (
            magnitude
            >
            self.config.flow_magnitude_threshold
        )

        if not np.any(valid):

            self.previous_gray = gray

            return empty

        valid_magnitude = (
            magnitude[valid]
        )

        mean_magnitude = float(
            np.mean(
                valid_magnitude
            )
        )

        p95_magnitude = float(
            np.percentile(
                valid_magnitude,
                95,
            )
        )

        # ----------------------------------------------------
        # FLOW DIVERGENCE
        # ----------------------------------------------------

        du_dx = np.gradient(
            u,
            axis=1,
        )

        dv_dy = np.gradient(
            v,
            axis=0,
        )

        divergence = (
            du_dx +
            dv_dy
        )

        mean_divergence = float(
            np.mean(
                np.abs(
                    divergence[valid]
                )
            )
        )

        # ----------------------------------------------------
        # MOTION ENTROPY
        # ----------------------------------------------------

        histogram, _ = np.histogram(

            angle[valid],

            bins=self.config.entropy_bins,

            range=(
                -np.pi,
                np.pi,
            ),
        )

        if histogram.sum() > 0:

            probabilities = (
                histogram
                /
                histogram.sum()
            )

            probabilities = (
                probabilities[
                    probabilities > 0
                ]
            )

            entropy = float(
                -np.sum(
                    probabilities
                    *
                    np.log(
                        probabilities
                    )
                )
            )

        else:

            entropy = 0.0

        # ----------------------------------------------------
        # AVERAGE FLOW DIRECTION
        # ----------------------------------------------------

        mean_u = float(
            np.mean(
                u[valid]
            )
        )

        mean_v = float(
            np.mean(
                v[valid]
            )
        )

        direction = float(
            np.degrees(
                np.arctan2(
                    mean_v,
                    mean_u,
                )
            )
        )

        self.previous_gray = gray

        return {

            "optical_flow_magnitude":
                mean_magnitude,

            "optical_flow_p95":
                p95_magnitude,

            "flow_divergence":
                mean_divergence,

            "motion_entropy":
                entropy,

            "optical_flow_direction":
                direction,
        }

    # ========================================================
    # BASELINE
    # ========================================================

    def establish_baseline(
        self,
        features: pd.DataFrame,
    ) -> Dict:

        baseline_frames = int(
            self.config.baseline_seconds
            * self.fps
        )

        baseline = features[
            features["frame"]
            < baseline_frames
        ].copy()

        if len(baseline) < 10:

            baseline = features.head(
                min(
                    300,
                    len(features),
                )
            )

        columns = [

            "avg_speed",

            "avg_acceleration",

            "rapid_motion_ratio",

            "optical_flow_magnitude",

            "optical_flow_p95",

            "flow_divergence",

            "motion_entropy",

            "density_change_smoothed",
        ]

        stats = {}

        for column in columns:

            if column not in baseline.columns:

                continue

            values = (
                baseline[column]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0.0)
            )

            mean = float(
                values.mean()
            )

            std = float(
                values.std()
            )

            if std < 1e-6:

                std = 1e-6

            stats[column] = {

                "mean":
                    mean,

                "std":
                    std,
            }

        self.baseline = stats

        return stats

    # ========================================================
    # POSITIVE Z SCORE
    # ========================================================

    def positive_zscore(
        self,
        value: float,
        feature_name: str,
    ) -> float:

        if (
            self.baseline is None
            or
            feature_name
            not in self.baseline
        ):

            return 0.0

        mean = (
            self.baseline[
                feature_name
            ]["mean"]
        )

        std = (
            self.baseline[
                feature_name
            ]["std"]
        )

        z = (
            value - mean
        ) / std

        return float(
            max(
                0.0,
                z,
            )
        )

    # ========================================================
    # SUDDEN MOTION CHANGE
    # ========================================================

    def calculate_sudden_change(
        self,
        features: pd.DataFrame,
    ) -> pd.Series:

        columns = [

            "avg_speed",

            "avg_acceleration",

            "optical_flow_magnitude",

            "optical_flow_p95",

            "rapid_motion_ratio",
        ]

        available = [

            column

            for column in columns

            if column in features.columns
        ]

        if not available:

            return pd.Series(
                0.0,
                index=features.index,
            )

        changes = []

        lookback = max(
            5,
            int(
                self.fps
                * 0.5
            ),
        )

        for column in available:

            current = (
                features[column]
                .astype(float)
            )

            previous = (
                current.shift(
                    lookback
                )
            )

            change = (
                current -
                previous
            )

            baseline = (
                self.baseline.get(
                    column,
                    {
                        "mean": 0.0,
                        "std": 1.0,
                    },
                )
            )

            std = max(
                baseline["std"],
                1e-6,
            )

            z = (
                change /
                std
            )

            changes.append(
                z.clip(
                    lower=0
                )
            )

        result = (
            pd.concat(
                changes,
                axis=1,
            )
            .mean(axis=1)
        )

        return result.clip(
            lower=0,
            upper=4,
        )

    # ========================================================
    # RISK SCORE
    # ========================================================

    def calculate_anomaly_score(
        self,
        row: pd.Series,
    ) -> float:

        speed = np.clip(

            self.positive_zscore(
                float(
                    row["avg_speed"]
                ),
                "avg_speed",
            )
            / 3.0,

            0,
            1,
        )

        acceleration = np.clip(

            self.positive_zscore(
                float(
                    row[
                        "avg_acceleration"
                    ]
                ),
                "avg_acceleration",
            )
            / 3.0,

            0,
            1,
        )

        rapid = np.clip(

            self.positive_zscore(
                float(
                    row[
                        "rapid_motion_ratio"
                    ]
                ),
                "rapid_motion_ratio",
            )
            / 2.5,

            0,
            1,
        )

        flow = np.clip(

            self.positive_zscore(
                float(
                    row[
                        "optical_flow_magnitude"
                    ]
                ),
                "optical_flow_magnitude",
            )
            / 3.0,

            0,
            1,
        )

        flow_p95 = np.clip(

            self.positive_zscore(
                float(
                    row[
                        "optical_flow_p95"
                    ]
                ),
                "optical_flow_p95",
            )
            / 3.0,

            0,
            1,
        )

        divergence = np.clip(

            self.positive_zscore(
                float(
                    row[
                        "flow_divergence"
                    ]
                ),
                "flow_divergence",
            )
            / 3.0,

            0,
            1,
        )

        entropy = np.clip(

            self.positive_zscore(
                float(
                    row[
                        "motion_entropy"
                    ]
                ),
                "motion_entropy",
            )
            / 3.0,

            0,
            1,
        )

        density = np.clip(

            self.positive_zscore(
                float(
                    row[
                        "density_change_smoothed"
                    ]
                ),
                "density_change_smoothed",
            )
            / 3.0,

            0,
            1,
        )

        sudden = np.clip(

            float(
                row.get(
                    "sudden_motion_change",
                    0.0,
                )
            )
            / 3.0,

            0,
            1,
        )

        # ----------------------------------------------------
        # FINAL WEIGHTED SCORE
        # ----------------------------------------------------

        score = (

            0.17 * speed

            + 0.08 * acceleration

            + 0.16 * rapid

            + 0.20 * flow

            + 0.09 * flow_p95

            + 0.08 * divergence

            + 0.07 * entropy

            + 0.05 * density

            + 0.10 * sudden
        )

        # ----------------------------------------------------
        # MULTI-SIGNAL AGREEMENT
        # ----------------------------------------------------

        signals = [

            speed,

            acceleration,

            rapid,

            flow,

            flow_p95,

            divergence,

            entropy,

            density,

            sudden,
        ]

        strong_signals = sum(
            value >= 0.55
            for value in signals
        )

        if strong_signals >= 5:

            score += 0.10

        elif strong_signals >= 4:

            score += 0.07

        elif strong_signals >= 3:

            score += 0.03

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # THREE-STATE CLASSIFICATION
    # ========================================================

    def classify_state(
        self,
        score: float,
    ) -> str:

        if (
            score
            >=
            self.config.high_risk_threshold
        ):

            return "HIGH_RISK"

        if (
            score
            >=
            self.config.abnormal_threshold
        ):

            return "ABNORMAL"

        return "NORMAL"


# ============================================================
# NEW RUN FOLDER
# ============================================================

def create_run_directory(
    output_root: str,
) -> Path:

    root = Path(
        output_root
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )

    run_dir = (
        root /
        timestamp
    )

    counter = 1

    while run_dir.exists():

        run_dir = (
            root /
            f"{timestamp}_{counter:02d}"
        )

        counter += 1

    run_dir.mkdir(
        parents=True
    )

    return run_dir


# ============================================================
# MOVEMENT DIRECTION
# ============================================================

def direction_to_compass(
    degrees: float,
) -> str:

    directions = [

        "EAST",

        "NORTH-EAST",

        "NORTH",

        "NORTH-WEST",

        "WEST",

        "SOUTH-WEST",

        "SOUTH",

        "SOUTH-EAST",
    ]

    index = int(
        (
            (degrees % 360)
            + 22.5
        )
        // 45
    ) % 8

    return directions[index]


# ============================================================
# BUILD EVENT JSON
# ============================================================

def build_team_events(
    features: pd.DataFrame,
    video_id: str,
    camera_id: str,
    zone_id: str,
):

    event_rows = features[
        features["event_id"] > 0
    ]

    events = []

    for event_id, group in (
        event_rows.groupby(
            "event_id"
        )
    ):

        peak_index = (
            group[
                "risk_score_smoothed"
            ]
            .idxmax()
        )

        peak = features.loc[
            peak_index
        ]

        # Confirmed stampede only when
        # high-risk behaviour persisted.
        stampede_detected = bool(
            group[
                "persistent_high_risk"
            ]
            .astype(bool)
            .any()
        )

        if stampede_detected:

            severity = "CRITICAL"

        else:

            severity = "WARNING"

        start_time = float(
            group[
                "timestamp"
            ].min()
        )

        end_time = float(
            group[
                "timestamp"
            ].max()
        )

        max_crowd = int(
            group[
                "head_count"
            ]
            .max()
        )

        max_speed = float(
            group[
                "avg_speed"
            ]
            .max()
        )

        direction = (
            direction_to_compass(
                float(
                    peak[
                        "optical_flow_direction"
                    ]
                )
            )
        )

        events.append(
            {

                "event_id":
                    int(event_id),

                "frame_id":
                    int(
                        peak["frame"]
                    ),

                "timestamp":
                    round(
                        float(
                            peak[
                                "timestamp"
                            ]
                        ),
                        3,
                    ),

                "camera_id":
                    camera_id,

                "zone_id":
                    zone_id,

                "stampede_detected":
                    stampede_detected,

                "confidence":
                    round(
                        float(
                            peak[
                                "risk_score_smoothed"
                            ]
                        ),
                        3,
                    ),

                "severity":
                    severity,

                "crowd_count":
                    max_crowd,

                "movement_direction":
                    direction,

                "movement_speed":
                    round(
                        max_speed,
                        3,
                    ),

                "abnormal_movement":
                    True,

                "event_start":
                    round(
                        start_time,
                        3,
                    ),

                "event_end":
                    round(
                        end_time,
                        3,
                    ),

                "event_duration":
                    round(
                        end_time -
                        start_time,
                        3,
                    ),

                "frames_in_event":
                    int(
                        len(group)
                    ),
            }
        )

    return events


# ============================================================
# MAIN VIDEO ANALYSIS
# ============================================================

def analyse_video(
    video_path: str,
    output_dir: Path,
    video_id: str,
    camera_id: str,
    zone_id: str,
):

    from ultralytics import YOLO

    from huggingface_hub import (
        hf_hub_download
    )

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video:\n"
            + video_path
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print()
    print("=" * 70)
    print(
        "VIDEO INFORMATION"
    )
    print("=" * 70)

    print(
        f"Resolution : "
        f"{width} x {height}"
    )

    print(
        f"FPS        : "
        f"{fps:.2f}"
    )

    print(
        f"Frames     : "
        f"{total_frames}"
    )

    print(
        f"Duration   : "
        f"{duration:.2f} seconds"
    )

    # ========================================================
    # YOLO
    # ========================================================

    print()
    print("=" * 70)
    print(
        "LOADING YOLOv8 HEAD MODEL"
    )
    print("=" * 70)

    model_path = hf_hub_download(

        repo_id=(
            "AmineSam/"
            "irail-crowd-counting-yolov8n"
        ),

        filename="best.pt",
    )

    model = YOLO(
        model_path
    )

    print(
        "Model loaded."
    )

    print(
        "Classes:",
        model.names
    )

    # ========================================================
    # TRACKING
    # ========================================================

    print()
    print("=" * 70)
    print(
        "YOLOv8 + BoT-SORT TRACKING"
    )
    print("=" * 70)

    tracking_rows = []

    results = model.track(

        source=video_path,

        device=0,

        conf=0.30,

        imgsz=832,

        iou=0.75,

        tracker="botsort.yaml",

        persist=True,

        stream=True,

        verbose=False,
    )

    for frame_id, result in enumerate(
        results
    ):

        if (
            result.boxes is None
            or
            result.boxes.id is None
        ):

            continue

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        for (
            box,
            track_id,
            confidence,
        ) in zip(
            boxes,
            ids,
            confidences,
        ):

            x1, y1, x2, y2 = box

            cx = (
                x1 +
                x2
            ) / 2.0

            cy = (
                y1 +
                y2
            ) / 2.0

            tracking_rows.append(
                {

                    "frame":
                        frame_id,

                    "track_id":
                        int(
                            track_id
                        ),

                    "x":
                        float(cx),

                    "y":
                        float(cy),

                    "width":
                        float(
                            x2 -
                            x1
                        ),

                    "height":
                        float(
                            y2 -
                            y1
                        ),

                    "confidence":
                        float(
                            confidence
                        ),
                }
            )

    tracking_data = pd.DataFrame(
        tracking_rows
    )

    if tracking_data.empty:

        raise RuntimeError(
            "No tracking detections "
            "were produced."
        )

    print(
        f"Tracking records : "
        f"{len(tracking_data)}"
    )

    print(
        f"Unique track IDs : "
        f"{tracking_data['track_id'].nunique()}"
    )

    print(
        f"Frames tracked   : "
        f"{tracking_data['frame'].nunique()}"
    )

    # ========================================================
    # TRACK FILTERING
    # ========================================================

    config = StampedeConfig()

    lengths = (
        tracking_data
        .groupby("track_id")
        .size()
    )

    valid_ids = lengths[
        lengths
        >= config.min_track_length
    ].index

    tracking_data = (
        tracking_data[
            tracking_data[
                "track_id"
            ].isin(valid_ids)
        ]
        .copy()
    )

    print()
    print("=" * 70)
    print(
        "TRACK FILTERING"
    )
    print("=" * 70)

    print(
        f"Valid tracks    : "
        f"{tracking_data['track_id'].nunique()}"
    )

    # ========================================================
    # MOTION FEATURES
    # ========================================================

    detector = StampedeDetector(
        fps=fps,
        config=config,
    )

    tracking_data = (
        detector.calculate_speed(
            tracking_data
        )
    )

    tracking_data = (
        detector.calculate_acceleration(
            tracking_data
        )
    )

    tracking_data = (
        detector.calculate_direction(
            tracking_data
        )
    )

    tracking_data = (
        detector.calculate_rapid_motion(
            tracking_data
        )
    )

    density = (
        detector.calculate_density(
            tracking_data
        )
    )

    consistency = (
        detector.calculate_direction_consistency(
            tracking_data
        )
    )

    # ========================================================
    # FRAME FEATURES
    # ========================================================

    frame_features = (

        tracking_data

        .groupby("frame")

        .agg(

            head_count=(
                "track_id",
                "nunique",
            ),

            avg_speed=(
                "speed",
                "mean",
            ),

            avg_acceleration=(
                "acceleration_smoothed",
                "mean",
            ),

            rapid_people=(
                "rapid_motion",
                "sum",
            ),
        )

        .reset_index()
    )

    frame_features[
        "rapid_motion_ratio"
    ] = (

        frame_features[
            "rapid_people"
        ]

        /

        frame_features[
            "head_count"
        ].replace(
            0,
            np.nan,
        )
    )

    # IMPORTANT:
    #
    # Density has its own head-count column.
    # We only merge density-change values.
    #
    # This prevents the previous
    # KeyError/head_count collision.

    density_for_merge = density[
        [
            "frame",
            "density_change",
            "density_change_smoothed",
        ]
    ]

    frame_features = (
        frame_features

        .merge(
            density_for_merge,
            on="frame",
            how="left",
        )

        .merge(
            consistency,
            on="frame",
            how="left",
        )
    )

    frame_features = (
        frame_features

        .replace(
            [np.inf, -np.inf],
            np.nan,
        )

        .fillna(0.0)
    )

    # ========================================================
    # OPTICAL FLOW
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STREAMING FARNEBACK OPTICAL FLOW"
    )
    print("=" * 70)

    flow_detector = StampedeDetector(
        fps=fps,
        config=config,
    )

    cap = cv2.VideoCapture(
        video_path
    )

    flow_rows = []

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        flow = (
            flow_detector
            .calculate_optical_flow(
                frame
            )
        )

        flow_rows.append(
            {

                "frame":
                    frame_id,

                "optical_flow_magnitude":
                    flow[
                        "optical_flow_magnitude"
                    ],

                "optical_flow_p95":
                    flow[
                        "optical_flow_p95"
                    ],

                "flow_divergence":
                    flow[
                        "flow_divergence"
                    ],

                "motion_entropy":
                    flow[
                        "motion_entropy"
                    ],

                "optical_flow_direction":
                    flow[
                        "optical_flow_direction"
                    ],
            }
        )

        frame_id += 1

    cap.release()

    flow_data = pd.DataFrame(
        flow_rows
    )

    print(
        f"Optical-flow frames: "
        f"{len(flow_data)}"
    )

    # ========================================================
    # COMBINE
    # ========================================================

    features = (

        frame_features

        .merge(
            flow_data,
            on="frame",
            how="outer",
        )

        .sort_values(
            "frame"
        )

        .reset_index(
            drop=True
        )
    )

    features[
        "timestamp"
    ] = (

        features[
            "frame"
        ]

        /

        fps
    )

    features = (
        features

        .replace(
            [np.inf, -np.inf],
            np.nan,
        )

        .fillna(0.0)
    )

    # ========================================================
    # NORMAL BASELINE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "NORMAL BEHAVIOUR BASELINE"
    )
    print("=" * 70)

    detector.establish_baseline(
        features
    )

    for (
        name,
        values,
    ) in detector.baseline.items():

        print(
            f"{name:30s} "
            f"mean={values['mean']:.4f} "
            f"std={values['std']:.4f}"
        )

    # ========================================================
    # SUDDEN MOTION
    # ========================================================

    features[
        "sudden_motion_change"
    ] = (

        detector.calculate_sudden_change(
            features
        )
    )

    # ========================================================
    # RISK SCORE
    # ========================================================

    scores = []

    for _, row in features.iterrows():

        scores.append(
            detector.calculate_anomaly_score(
                row
            )
        )

    features[
        "risk_score"
    ] = scores

    # ========================================================
    # RISK SMOOTHING
    # ========================================================

    smoothing_frames = max(

        3,

        int(
            fps
            *
            config.smoothing_window
            /
            2
        ),
    )

    features[
        "risk_score_smoothed"
    ] = (

        features[
            "risk_score"
        ]

        .rolling(
            window=smoothing_frames,
            min_periods=1,
            center=True,
        )

        .median()

        .bfill()

        .ffill()
    )

    # ========================================================
    # THREE STATES
    # ========================================================

    features[
        "state"
    ] = (

        features[
            "risk_score_smoothed"
        ]

        .apply(
            detector.classify_state
        )
    )

    # Compatibility column.
    features[
        "status"
    ] = features[
        "state"
    ]

    # ========================================================
    # PERSISTENT HIGH RISK
    # ========================================================

    persistence_frames = max(

        1,

        int(
            config.persistence_seconds
            * fps
        ),
    )

    high = (

        features[
            "risk_score_smoothed"
        ]

        >=

        config.high_risk_threshold
    )

    group_id = (
        (~high)
        .cumsum()
    )

    consecutive_high = (

        high.astype(int)

        .groupby(
            group_id
        )

        .cumsum()
    )

    features[
        "persistent_high_risk"
    ] = (

        consecutive_high

        >=

        persistence_frames
    )

    # ========================================================
    # ABNORMAL EVENTS
    # ========================================================

    abnormal = (

        features[
            "state"
        ]

        !=

        "NORMAL"
    )

    gap_frames = max(

        1,

        int(
            config.event_gap_seconds
            * fps
        ),
    )

    event_ids = np.zeros(

        len(features),

        dtype=int,
    )

    event_number = 0

    last_abnormal_index = None

    for (
        i,
        is_abnormal,
    ) in enumerate(
        abnormal.values
    ):

        if is_abnormal:

            if (

                last_abnormal_index
                is None

                or

                i -
                last_abnormal_index
                >
                gap_frames
            ):

                event_number += 1

            event_ids[i] = (
                event_number
            )

            last_abnormal_index = i

    features[
        "event_id"
    ] = event_ids

    # ========================================================
    # REMOVE VERY SHORT EVENTS
    # ========================================================

    minimum_event_frames = max(

        1,

        int(
            config.minimum_abnormal_duration
            * fps
        ),
    )

    for event_id in sorted(
        set(event_ids)
    ):

        if event_id == 0:

            continue

        count = int(

            (
                features[
                    "event_id"
                ]

                ==

                event_id
            )

            .sum()
        )

        if (
            count
            <
            minimum_event_frames
        ):

            features.loc[

                features[
                    "event_id"
                ]

                ==

                event_id,

                "event_id",
            ] = 0

    # ========================================================
    # RE-NUMBER EVENTS
    # ========================================================

    mapping = {}

    current = 0

    for event_id in sorted(
        set(
            features[
                "event_id"
            ]
        )
    ):

        if event_id == 0:

            continue

        current += 1

        mapping[
            event_id
        ] = current

    features[
        "event_id"
    ] = (

        features[
            "event_id"
        ]

        .map(
            lambda x:
            mapping.get(
                x,
                0,
            )
        )
    )

    # ========================================================
    # BUILD TEAM JSON EVENTS
    # ========================================================

    events = build_team_events(

        features,

        video_id,

        camera_id,

        zone_id,
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        output_dir /
        "stampede_features.csv"
    )

    features.to_csv(
        csv_path,
        index=False,
    )

    # ========================================================
    # SAVE TEAM EVENT JSON
    # ========================================================

    json_path = (
        output_dir /
        "stampede_events.json"
    )

    team_json = {

        "video_id":
            video_id,

        "detections":
            events,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            team_json,
            file,
            indent=4,
        )

    # ========================================================
    # SAVE INTERNAL ALERT JSON
    # ========================================================

    alerts_path = (
        output_dir /
        "stampede_alerts.json"
    )

    internal_json = {

        "video":
            video_path,

        "video_id":
            video_id,

        "camera_id":
            camera_id,

        "zone_id":
            zone_id,

        "fps":
            fps,

        "detector_version":
            "V3.3",

        "states":
            [
                "NORMAL",
                "ABNORMAL",
                "HIGH_RISK",
            ],

        "total_events":
            len(events),

        "events":
            events,
    }

    with open(
        alerts_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            internal_json,
            file,
            indent=4,
        )

    # ========================================================
    # RISK PLOT
    # ========================================================

    import matplotlib

    matplotlib.use(
        "Agg"
    )

    import matplotlib.pyplot as plt

    plots_dir = (
        output_dir /
        "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(

        features[
            "timestamp"
        ],

        features[
            "risk_score_smoothed"
        ],

        label="Risk Score",
    )

    plt.axhline(

        config.abnormal_threshold,

        linestyle="--",

        label="Abnormal Threshold",
    )

    plt.axhline(

        config.high_risk_threshold,

        linestyle="--",

        label="High-Risk Threshold",
    )

    for event in events:

        if (
            event[
                "severity"
            ]
            ==
            "CRITICAL"
        ):

            alpha = 0.25

        else:

            alpha = 0.12

        plt.axvspan(

            event[
                "event_start"
            ],

            event[
                "event_end"
            ],

            alpha=alpha,
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Risk Score"
    )

    plt.title(
        "Stampede Detection - "
        "Normal / Abnormal / High Risk"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plot_path = (
        plots_dir /
        "stampede_risk.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    normal_frames = int(

        (
            features[
                "state"
            ]
            ==
            "NORMAL"
        )

        .sum()
    )

    abnormal_frames = int(

        (
            features[
                "state"
            ]
            ==
            "ABNORMAL"
        )

        .sum()
    )

    high_frames = int(

        (
            features[
                "state"
            ]
            ==
            "HIGH_RISK"
        )

        .sum()
    )

    persistent_frames = int(

        features[
            "persistent_high_risk"
        ]

        .astype(bool)

        .sum()
    )

    critical_events = sum(

        event[
            "stampede_detected"
        ]

        for event in events
    )

    warning_events = (
        len(events)
        -
        critical_events
    )

    max_risk = float(

        features[
            "risk_score_smoothed"
        ]

        .max()
    )

    mean_risk = float(

        features[
            "risk_score_smoothed"
        ]

        .mean()
    )

    print()
    print("=" * 70)
    print(
        "STAMPEDE DETECTION V3.3 SUMMARY"
    )
    print("=" * 70)

    print(
        f"Maximum risk score : "
        f"{max_risk:.3f}"
    )

    print(
        f"Mean risk score    : "
        f"{mean_risk:.3f}"
    )

    print(
        f"Normal frames      : "
        f"{normal_frames}"
    )

    print(
        f"Abnormal frames    : "
        f"{abnormal_frames}"
    )

    print(
        f"High-risk frames   : "
        f"{high_frames}"
    )

    print(
        f"Persistent frames  : "
        f"{persistent_frames}"
    )

    print(
        f"Detected events    : "
        f"{len(events)}"
    )

    print(
        f"Critical events    : "
        f"{critical_events}"
    )

    print(
        f"Warning events     : "
        f"{warning_events}"
    )

    if events:

        print()
        print(
            "Detected events:"
        )

        for event in events:

            start = float(
                event[
                    "event_start"
                ]
            )

            end = float(
                event[
                    "event_end"
                ]
            )

            start_min = int(
                start // 60
            )

            start_sec = (
                start % 60
            )

            end_min = int(
                end // 60
            )

            end_sec = (
                end % 60
            )

            print(

                f"  Event "
                f"{event['event_id']}: "

                f"{start_min}:"
                f"{start_sec:05.2f}"

                f" - "

                f"{end_min}:"
                f"{end_sec:05.2f}"

                f" | "

                f"{event['severity']}"

                f" | Peak "

                f"{event['confidence']:.3f}"
            )

    else:

        print()
        print(
            "No abnormal events detected."
        )

    # ========================================================
    # OUTPUT INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print(
        "OUTPUTS"
    )
    print("=" * 70)

    print(
        f"Run folder : "
        f"{output_dir}"
    )

    print(
        f"CSV        : "
        f"{csv_path}"
    )

    print(
        f"Team JSON   : "
        f"{json_path}"
    )

    print(
        f"Alerts JSON : "
        f"{alerts_path}"
    )

    print(
        f"Plot        : "
        f"{plot_path}"
    )

    print()
    print(
        "STAMPEDE DETECTION COMPLETE"
    )

    print("=" * 70)

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Single-file YOLOv8 + "
            "BoT-SORT + optical flow "
            "stampede detection."
        )
    )

    parser.add_argument(

        "--video",

        required=True,

        help=(
            "Path to input video."
        ),
    )

    parser.add_argument(

        "--video-id",

        default="VID_001",

        help=(
            "Video ID for team JSON."
        ),
    )

    parser.add_argument(

        "--camera-id",

        default="CAM_03",

        help=(
            "Camera ID for team JSON."
        ),
    )

    parser.add_argument(

        "--zone-id",

        default="ZONE_D",

        help=(
            "Zone ID for team JSON."
        ),
    )

    parser.add_argument(

        "--output-root",

        default=(

            "Crowd_Monitoring/"
            "Stampede_Detection/"
            "data/output/"
            "stampede_runs"
        ),

        help=(
            "Root directory for "
            "timestamped run outputs."
        ),
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    ).expanduser()

    if not video_path.exists():

        raise FileNotFoundError(

            "Video not found:\n"
            +
            str(video_path)
        )

    # --------------------------------------------------------
    # CREATE UNIQUE RUN DIRECTORY
    # --------------------------------------------------------

    run_dir = (
        create_run_directory(
            args.output_root
        )
    )

    print()
    print("=" * 70)
    print(
        "NEW STAMPEDE DETECTION RUN"
    )
    print("=" * 70)

    print(
        f"Video ID  : "
        f"{args.video_id}"
    )

    print(
        f"Camera ID : "
        f"{args.camera_id}"
    )

    print(
        f"Zone ID   : "
        f"{args.zone_id}"
    )

    print(
        f"Output    : "
        f"{run_dir}"
    )

    analyse_video(

        video_path=str(
            video_path
        ),

        output_dir=run_dir,

        video_id=args.video_id,

        camera_id=args.camera_id,

        zone_id=args.zone_id,
    )


if __name__ == "__main__":

    main()