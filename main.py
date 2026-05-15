from anomaly_detection.detector import AnomalyDetector
from anomaly_detection.report import print_report

detector = AnomalyDetector("weekendaway_fixed.csv")
detector.configure({
    "idle_gap": {
        "threshold_seconds": 12 * 60 * 60,
        "housewide_sensor_ratio": 0.8,
        "return_window_minutes": 60,
    }
})

results = detector.analyze()
print_report(results)

# --- debug ---
# from anomaly_detection.report import fmt_timestamps
# print(fmt_timestamps(results["housewide_gaps"], ["silence_start", "silence_end"]).to_string(index=False))