from anomaly_detection.detector import AnomalyDetector

detector = AnomalyDetector("hh103.csv")
results = detector.analyze()

print("\n--- Duration summary ---")
print(results["duration_summary"].to_string(index=False))

print("\n--- House-wide silences (excluded from scoring) ---")
print(results["housewide_silences"].to_string(index=False))

print("\n--- Idle gaps (individual sensors) ---")
print(results["idle_gaps"].sort_values("idle_seconds", ascending=False).to_string(index=False))

print("\n--- Severity scores ---")
print(results["severity_scores"].to_string(index=False))