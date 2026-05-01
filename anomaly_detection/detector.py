import json
import os
import pandas as pd
import copy
from anomaly_detection.pipeline import load_data


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
    "idle_gap": {
        "threshold_seconds": 3 * 24 * 60 * 60,
        "housewide_window_minutes": 1440,
        "housewide_sensor_ratio": 0.8,
        "return_sensor_ratio": 0.4,
        "return_window_minutes": 60,
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
            config = deep_update(config, json.load(f))
    if user_config:
        config = deep_update(config, user_config)
    return config


class AnomalyDetector:
    def __init__(self, filepath, config=None):
        print("=== Loading data ===")
        self.df = load_data(filepath)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        self.config = load_config(config)
        self.total_sensors = self.df["sensor"].nunique()

    def find_housewide_silence_end(self, housewide_silence_confirmed_at):
        cfg = self.config["idle_gap"]
        return_window = pd.Timedelta(minutes=cfg["return_window_minutes"])
        return_ratio = cfg["return_sensor_ratio"]
        scan_from = housewide_silence_confirmed_at + pd.Timedelta(seconds=cfg["threshold_seconds"])

        after = self.df[self.df["timestamp"] > scan_from].reset_index(drop=True)

        for idx, row in after.iterrows():
            window_events = after[
                (after["timestamp"] >= row["timestamp"]) &
                (after["timestamp"] <= row["timestamp"] + return_window)
            ]
            if window_events["sensor"].nunique() >= (self.total_sensors * return_ratio):
                return row["timestamp"]

        return self.df["timestamp"].max()

    def detect_housewide_silence(self, gaps):
        cfg = self.config["idle_gap"]
        window_minutes = cfg["housewide_window_minutes"]
        sensor_ratio = cfg["housewide_sensor_ratio"]
        window_delta = pd.Timedelta(minutes=window_minutes)

        gaps = gaps.sort_values("silence_start").copy()

        # Pass 1: sweep forward to find clusters of sensors going silent together
        housewide_silence_anchors = []
        for idx, row in gaps.iterrows():
            if any(
                row["silence_start"] >= anchor and row["silence_start"] <= anchor + window_delta
                for anchor in housewide_silence_anchors
            ):
                continue

            cluster = gaps[
                (gaps["silence_start"] >= row["silence_start"]) &
                (gaps["silence_start"] <= row["silence_start"] + window_delta)
            ]

            if len(cluster["sensor"].unique()) >= (self.total_sensors * sensor_ratio):
                housewide_silence_anchors.append(row["silence_start"])

        # Pass 2: mark any gap within ±window of any anchor as housewide
        def in_housewide_window(silence_start):
            return any(
                abs((silence_start - anchor).total_seconds()) < window_minutes * 60
                for anchor in housewide_silence_anchors
            )

        gaps["housewide"] = gaps["silence_start"].apply(in_housewide_window)

        # Build housewide events
        housewide_events = []
        for anchor in housewide_silence_anchors:
            cluster = gaps[
                (gaps["silence_start"] >= anchor - window_delta) &
                (gaps["silence_start"] <= anchor + window_delta)
            ]
            housewide_silence_start = cluster["silence_start"].min()
            housewide_silence_confirmed_at = cluster["silence_start"].max()
            housewide_silence_end = self.find_housewide_silence_end(housewide_silence_confirmed_at)
            duration = (housewide_silence_end - housewide_silence_start).total_seconds()

            housewide_events.append({
                "started_at": housewide_silence_start,
                "ended_at": housewide_silence_end,
                "sensors_affected": len(cluster["sensor"].unique()),
                "idle_time": format_duration(duration),
                "cluster_sensors": set(cluster["sensor"].unique()),
            })

        return gaps, pd.DataFrame(housewide_events)

    def analyze_idle_gaps(self):
        cfg = self.config["idle_gap"]
        results = []

        for sensor, group in self.df.groupby("sensor"):
            group = group.reset_index(drop=True)
            gaps = group["timestamp"].diff().dt.total_seconds()
            long_gaps = gaps[gaps > cfg["threshold_seconds"]]

            for idx, gap in long_gaps.items():
                results.append({
                    "sensor": sensor,
                    "silence_start": group.loc[idx - 1, "timestamp"],
                    "silence_end": group.loc[idx, "timestamp"],
                    "idle_seconds": float(gap),
                    "idle_time": format_duration(gap)
                })

        if results:
            gaps = pd.DataFrame(results)
            gaps, housewide_events = self.detect_housewide_silence(gaps)
            sensor_gaps = gaps[~gaps["housewide"]].drop(columns=["housewide"])
            housewide_gaps = gaps[gaps["housewide"]].drop(columns=["housewide"])
        else:
            sensor_gaps = pd.DataFrame()
            housewide_gaps = pd.DataFrame()
            housewide_events = pd.DataFrame()

        return sensor_gaps, housewide_gaps, housewide_events

    def detect_absent_firing(self, housewide_silences):
        results = []

        for _, silence in housewide_silences.iterrows():
            during = self.df[
                (self.df["timestamp"] > silence["started_at"]) &
                (self.df["timestamp"] < silence["ended_at"]) &
                (~self.df["sensor"].isin(silence["cluster_sensors"]))
            ]

            if len(during) == 0:
                continue

            counts = during.groupby("sensor").size().reset_index(name="fires")
            counts["silence_start"] = silence["started_at"]
            results.append(counts)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def analyze(self):
        print("\n=== Starting anomaly analysis ===")
        sensor_gaps, housewide_gaps, housewide_silences = self.analyze_idle_gaps()
        absent_firing = self.detect_absent_firing(housewide_silences)
        print("✔ Analysis complete\n")
        return {
            "idle_gaps": sensor_gaps,
            "housewide_gaps": housewide_gaps,
            "housewide_silences": housewide_silences,
            "absent_firing": absent_firing,
        }