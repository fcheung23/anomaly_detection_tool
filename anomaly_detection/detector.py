import json
import os
import pandas as pd
import copy
from anomaly_detection.pipeline import load_data, compute_sessions


def format_duration(seconds: float) -> str:
    seconds = int(seconds)

    days = seconds // 86400
    seconds %= 86400

    hours = seconds // 3600
    seconds %= 3600

    minutes = seconds // 60
    seconds %= 60

    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


DEFAULT_CONFIG = {
    "duration_bounds": {
        "default":     {"min": 2,  "max": 3 * 24 * 60 * 60},
        "Bathroom":    {"min": 30, "max": 2 * 60 * 60},
        "Bedroom":     {"min": 30, "max": 12 * 60 * 60},
        "LivingRoom":  {"min": 5,  "max": 4 * 60 * 60},
        "Kitchen":     {"min": 5,  "max": 3 * 60 * 60},
        "Chair":       {"min": 2,  "max": 4 * 60 * 60},
    },

    "idle_gap": {
        "threshold_seconds": 7 * 24 * 60 * 60,
        "housewide_window_minutes": 120,
        "housewide_sensor_ratio": 0.5
    },

    "severity": {
        "noise_ratio_threshold": 0.3,
    }
}


def deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(user_config=None):
    config = copy.deepcopy(DEFAULT_CONFIG)

    if os.path.exists("config.json"):
        with open("config.json") as f:
            file_config = json.load(f)
            config = deep_update(config, file_config)

    if user_config:
        config = deep_update(config, user_config)

    return config


class AnomalyDetector:
    def __init__(self, filepath, config=None):
        print("Loading data...")
        self.df = load_data(filepath)
        print("✔ Data loaded")

        print("Computing sessions...")
        self.sessions = compute_sessions(self.df)
        print("✔ Sessions computed")

        self.config = load_config(config)

    def analyze_duration(self):
        cfg = self.config["duration_bounds"]
        results = []

        print("Running duration analysis...")

        for sensor, group in self.sessions.groupby("sensor"):
            rules = cfg.get(sensor, cfg["default"])
            total = len(group)

            noise_sessions = group[group["duration_seconds"] < rules["min"]]
            long_sessions = group[group["duration_seconds"] > rules["max"]]

            noise_ratio = round(len(noise_sessions) / total, 3) if total > 0 else 0
            long_ratio = round(len(long_sessions) / total, 3) if total > 0 else 0

            results.append({
                "sensor": sensor,
                "total_sessions": total,
                "noise_sessions": len(noise_sessions),
                "noise_ratio": noise_ratio,
                "long_sessions": len(long_sessions),
                "long_ratio": long_ratio,
            })

        print("\n✔ Duration analysis complete")
        return pd.DataFrame(results)

    def detect_housewide_silence(self, idle_gaps):
        cfg = self.config["idle_gap"]
        window_minutes = cfg["housewide_window_minutes"]
        sensor_ratio = cfg["housewide_sensor_ratio"]
        total_sensors = self.sessions["sensor"].nunique()

        idle_gaps = idle_gaps.sort_values("start").copy()
        idle_gaps["gap_start"] = pd.to_datetime(idle_gaps["start"])

        housewide_flags = []
        housewide_events = []
        seen_windows = []

        for idx, row in idle_gaps.iterrows():
            window_start = row["gap_start"] - pd.Timedelta(minutes=window_minutes)
            window_end = row["gap_start"] + pd.Timedelta(minutes=window_minutes)

            nearby = idle_gaps[
                (idle_gaps["gap_start"] >= window_start) &
                (idle_gaps["gap_start"] <= window_end)
            ]

            is_housewide = len(nearby["sensor"].unique()) >= (total_sensors * sensor_ratio)
            housewide_flags.append(is_housewide)

            if is_housewide:
                already_seen = any(
                    abs((row["gap_start"] - seen).total_seconds()) < window_minutes * 60
                    for seen in seen_windows
                )
                if not already_seen:
                    seen_windows.append(row["gap_start"])
                    housewide_events.append({
                        "date": row["gap_start"].date(),
                        "started_at": row["gap_start"],
                        "sensors_affected": len(nearby["sensor"].unique()),
                        "idle_time": format_duration(nearby["idle_seconds"].max())
                    })

        idle_gaps["housewide"] = housewide_flags
        return idle_gaps, pd.DataFrame(housewide_events)

    def analyze_idle_gaps(self):
        cfg = self.config["idle_gap"]
        results = []

        print("Running idle gap analysis...")

        for sensor, group in self.sessions.groupby("sensor"):
            group = group.sort_values("start")

            gaps = group["start"].diff().dt.total_seconds()
            long_gaps = gaps[gaps > cfg["threshold_seconds"]]

            for idx, gap in long_gaps.items():
                results.append({
                    "sensor": sensor,
                    "start": group.loc[idx, "start"],
                    "idle_seconds": float(gap),
                    "idle_time": format_duration(gap)
                })

        if results:
            idle_gaps = pd.DataFrame(results)
            idle_gaps, housewide_events = self.detect_housewide_silence(idle_gaps)
            sensor_gaps = idle_gaps[~idle_gaps["housewide"]].drop(columns=["housewide", "gap_start"])
            print(f"  {len(housewide_events)} house-wide silence events detected")
            print(f"  {len(sensor_gaps)} individual sensor gaps remaining")
        else:
            sensor_gaps = pd.DataFrame()
            housewide_events = pd.DataFrame()

        print("✔ Idle gap analysis complete")
        return sensor_gaps, housewide_events

    def compute_severity_score(self, duration, idle_gaps):
        cfg = self.config["severity"]
        sensors = self.sessions["sensor"].unique()
        results = []

        max_idle = idle_gaps["idle_seconds"].max() if len(idle_gaps) > 0 else 1
        sensor_max_idle = idle_gaps.groupby("sensor")["idle_seconds"].max() if len(idle_gaps) > 0 else pd.Series(dtype=float)

        for sensor in sensors:
            flags = []
            score = 0.0

            sensor_rows = duration[duration["sensor"] == sensor]
            if len(sensor_rows) == 0:
                continue  # skip safely (no change in scoring logic for valid rows)

            row = sensor_rows.iloc[0]
            noise_ratio = row["noise_ratio"]

            if noise_ratio > cfg["noise_ratio_threshold"]:
                flags.append("high_noise_ratio")
            score += noise_ratio * 0.3

            if sensor in sensor_max_idle.index:
                flags.append("idle_gaps")
                score += sensor_max_idle[sensor] / max_idle

            results.append({
                "sensor": sensor,
                "severity_score": round(score, 3),
                "flags": ", ".join(flags) if flags else "none"
            })

        return pd.DataFrame(results).sort_values("severity_score", ascending=False).reset_index(drop=True)

    def analyze(self):
        print("\n=== Starting anomaly analysis ===")

        duration = self.analyze_duration()
        idle_gaps, housewide_silences = self.analyze_idle_gaps()
        severity = self.compute_severity_score(duration, idle_gaps)

        results = {
            "duration_summary": duration,
            "idle_gaps": idle_gaps,
            "housewide_silences": housewide_silences,
            "severity_scores": severity
        }

        print("✔ All analysis complete\n")
        return results