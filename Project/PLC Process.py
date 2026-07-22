"""
Manufacturing Process Data Analysis
------------------------------------
Single-file version: generates synthetic plant DCS data (pressure, temperature,
flow) for 4 equipment units, detects downtime & fault patterns, and computes
KPIs (Availability, MTBF, MTTR) ready for a Power BI dashboard.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ----------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------
START = datetime(2026, 1, 1)
DAYS = 90
FREQ_MIN = 15
N_STEPS = int(DAYS * 24 * 60 / FREQ_MIN)
timestamps = [START + timedelta(minutes=FREQ_MIN * i) for i in range(N_STEPS)]

equipment = {
    "COMP-101": {"pressure": 8.2, "temp": 68, "flow": 320},
    "COMP-102": {"pressure": 8.0, "temp": 70, "flow": 315},
    "PUMP-201": {"pressure": 5.5, "temp": 55, "flow": 210},
    "PUMP-202": {"pressure": 5.6, "temp": 54, "flow": 205},
}

# ----------------------------------------------------------------------
# 2. GENERATE SYNTHETIC DCS LOG DATA (with realistic fault patterns)
# ----------------------------------------------------------------------
all_readings = []

for eq_id, base in equipment.items():
    pressure = base["pressure"] + np.random.normal(0, 0.05, N_STEPS)
    temperature = base["temp"] + np.random.normal(0, 0.6, N_STEPS)
    flow = base["flow"] + np.random.normal(0, 3.0, N_STEPS)
    status = np.array(["RUN"] * N_STEPS, dtype=object)

    # inject a handful of fault episodes per equipment
    for _ in range(4):
        start = np.random.randint(300, N_STEPS - 300)
        ramp = np.random.randint(20, 80)     # steps of gradual drift before trip
        down = np.random.randint(8, 24)      # steps of downtime (repair)
        fault = np.random.choice(["Bearing Overheat", "Seal Leak", "Line Blockage"])

        if fault == "Bearing Overheat":
            temperature[start:start + ramp] += np.linspace(0, 25, ramp)
            flow[start:start + ramp] -= np.linspace(0, 40, ramp)
        elif fault == "Seal Leak":
            pressure[start:start + ramp] -= np.linspace(0, 2.0, ramp)
        else:  # Line Blockage
            flow[start:start + ramp] -= np.linspace(0, 70, ramp)

        trip = start + ramp
        status[trip:trip + down] = "DOWN"
        pressure[trip:trip + down] = 0
        temperature[trip:trip + down] = np.nan
        flow[trip:trip + down] = 0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "equipment_id": eq_id,
        "pressure_bar": np.clip(pressure, 0, None).round(2),
        "temperature_c": temperature.round(2),
        "flow_m3h": np.clip(flow, 0, None).round(1),
        "status": status,
    })
    all_readings.append(df)

readings = pd.concat(all_readings, ignore_index=True)
readings.to_csv("fact_readings.csv", index=False)

# ----------------------------------------------------------------------
# 3. DETECT DOWNTIME EVENTS FROM THE STATUS TAG
# ----------------------------------------------------------------------
downtime_events = []

for eq_id, g in readings.groupby("equipment_id"):
    g = g.reset_index(drop=True)
    is_down = (g["status"] == "DOWN").astype(int)
    change = is_down.diff().fillna(0)
    starts = g.index[change == 1].tolist()
    ends = g.index[change == -1].tolist()

    for s, e in zip(starts, ends):
        down_start = g.loc[s, "timestamp"]
        down_end = g.loc[min(e, len(g) - 1), "timestamp"]
        duration_min = (down_end - down_start).total_seconds() / 60

        # simple rule-based fault classification from the pre-trip trend
        pre = g.loc[max(0, s - 8):s - 1]
        if len(pre) >= 4:
            x = np.arange(len(pre))
            temp_slope = np.polyfit(x, pre["temperature_c"].ffill(), 1)[0]
            flow_slope = np.polyfit(x, pre["flow_m3h"], 1)[0]
            pressure_slope = np.polyfit(x, pre["pressure_bar"], 1)[0]

            if temp_slope > 0.1 and flow_slope < -0.2:
                fault_type = "Bearing Overheat"
            elif pressure_slope < -0.03:
                fault_type = "Seal Leak"
            elif flow_slope < -1.5:
                fault_type = "Line Blockage"
            else:
                fault_type = "Unclassified"
        else:
            fault_type = "Unclassified"

        downtime_events.append({
            "equipment_id": eq_id,
            "fault_type": fault_type,
            "start_time": down_start,
            "end_time": down_end,
            "duration_min": round(duration_min, 1),
        })

downtime = pd.DataFrame(downtime_events).sort_values(["equipment_id", "start_time"])
downtime.to_csv("fault_events.csv", index=False)

# ----------------------------------------------------------------------
# 4. CALCULATE KPIs: AVAILABILITY, MTBF, MTTR
# ----------------------------------------------------------------------
total_minutes = (readings["timestamp"].max() - readings["timestamp"].min()).total_seconds() / 60

kpi_rows = []
for eq_id, dt in downtime.groupby("equipment_id"):
    n_failures = len(dt)
    total_down = dt["duration_min"].sum()
    uptime = total_minutes - total_down
    kpi_rows.append({
        "equipment_id": eq_id,
        "availability_pct": round(uptime / total_minutes * 100, 2),
        "mtbf_hr": round((uptime / 60) / n_failures, 1),
        "mttr_hr": round((total_down / 60) / n_failures, 2),
        "failure_count": n_failures,
        "total_downtime_hr": round(total_down / 60, 1),
    })

kpi_summary = pd.DataFrame(kpi_rows)
kpi_summary.to_csv("kpi_summary.csv", index=False)

# ----------------------------------------------------------------------
# 5. PRINT RESULTS
# ----------------------------------------------------------------------
print("Rows generated:", len(readings))
print("\nDowntime events detected:", len(downtime))
print(downtime["fault_type"].value_counts())
print("\nKPI Summary:")
print(kpi_summary.to_string(index=False))
print("\nFiles saved: fact_readings.csv, fault_events.csv, kpi_summary.csv")
