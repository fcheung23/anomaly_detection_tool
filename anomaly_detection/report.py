# anomaly_detection/report.py

def fmt_timestamps(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M")
    return df


def print_report(results):
    housewide_silences = results["housewide_silences"]
    idle_gaps = results["idle_gaps"]
    absent_firing = results["absent_firing"]

    n_absences = len(housewide_silences)
    n_gaps = len(idle_gaps)
    n_absent = len(absent_firing)

    print(f"\n--- Likely home absences ({n_absences}) ---")
    if not housewide_silences.empty:
        df = fmt_timestamps(housewide_silences, ["started_at", "ended_at"])
        print(df.to_string(index=False))
    else:
        print("  None detected.")

    print(f"\n--- Idle gaps, not during home absence ({n_gaps}) ---")
    if not idle_gaps.empty:
        df = fmt_timestamps(idle_gaps, ["silence_start", "silence_end"]).drop(columns=["idle_seconds"])
        print(df.sort_values("silence_start").to_string(index=False))
    else:
        print("  None detected.")

    print(f"\n--- Absent firing during home absences ({n_absent}) ---")
    if not absent_firing.empty:
        for started_at, group in absent_firing.groupby("silence_start"):
            print(f"\n  absence: {started_at.strftime('%Y-%m-%d %H:%M')}")
            print(group.drop(columns=["silence_start"]).sort_values("fires", ascending=False).to_string(index=False))
    else:
        print("  None detected.")