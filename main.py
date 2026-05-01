from anomaly_detection.detector import AnomalyDetector

detector = AnomalyDetector("hh103.csv")
results = detector.analyze()

def fmt_timestamps(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M")
    return df

n_absences = len(results["housewide_silences"])
n_gaps = len(results["idle_gaps"])
n_absent = len(results["absent_firing"])

print(f"\n--- Likely home absences ({n_absences}) ---")
df = fmt_timestamps(results["housewide_silences"], ["started_at", "ended_at"])
print(df.to_string(index=False))

# print("\n--- Home absence breakdown (individual sensors) ---")
# print(results["housewide_gaps"].sort_values("silence_start").to_string(index=False))

print(f"\n--- Idle gaps, not during home absence ({n_gaps}) ---")
df = fmt_timestamps(results["idle_gaps"], ["silence_start", "silence_end"]).drop(columns=["idle_seconds"])
print(df.sort_values("silence_start").to_string(index=False))

print(f"\n--- Sensors firing during home absences ({n_absent}) ---")
for started_at, group in results["absent_firing"].groupby("silence_start"):
    print(f"\nabsence: {started_at.strftime('%Y-%m-%d %H:%M')}")
    print(group.drop(columns=["silence_start"]).sort_values("fires", ascending=False).to_string(index=False))