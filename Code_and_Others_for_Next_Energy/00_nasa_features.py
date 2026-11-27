"""Step 00 (optional). Build data/nasa_cycle_features.csv from the NASA PCoE Li-ion battery aging files
B0005.mat, B0006.mat, B0007.mat and B0018.mat (Saha and Goebel, NASA Ames Prognostics Center of Excellence, Battery Data
Set, folder 1. BatteryAgingARC-FY08Q4). Put the four .mat files in data/nasa_raw/ and run: python 00_nasa_features.py
One row per discharge cycle (168, 168, 168 and 132 cycles):
  capacity  discharge capacity (Ah), field Capacity
  t_dis     time until the measured voltage first falls below 2.7 V (s)          [proxy of capacity; not used by 06]
  t_Tmax    time of the maximum measured temperature (s)                         [proxy of capacity; not used by 06]
  V_mean    mean measured voltage from the start of the record until 2.7 V (V)
  V_600s    measured voltage at 600 s, linear interpolation (V)
  T_max     maximum measured temperature (deg C)
  T_start   measured temperature at the first sample (deg C)
  I_mean    mean measured current from the start of the record until 2.7 V (A)
  T_amb     ambient temperature of the cycle (deg C)"""
import os
import numpy as np, pandas as pd
from scipy.io import loadmat

RAW = os.path.join('data', 'nasa_raw'); rows = []
for b in ['B0005', 'B0006', 'B0007', 'B0018']:
    cycles = loadmat(os.path.join(RAW, f'{b}.mat'), simplify_cells=True)[b]['cycle']; k = 0
    for c in cycles:
        if c['type'] != 'discharge':
            continue
        k += 1; d = c['data']
        t = np.asarray(d['Time'], float); v = np.asarray(d['Voltage_measured'], float)
        T = np.asarray(d['Temperature_measured'], float); I = np.asarray(d['Current_measured'], float)
        below = np.flatnonzero(v < 2.7); end = below[0] if len(below) else len(v) - 1   # first sample below 2.7 V
        rows.append(dict(battery=int(b[1:]), cycle=k, capacity=float(np.atleast_1d(d['Capacity'])[0]), t_dis=float(t[end]),
                         t_Tmax=float(t[int(np.argmax(T))]), V_mean=float(v[:end + 1].mean()), V_600s=float(np.interp(600.0, t, v)),
                         T_max=float(T.max()), T_start=float(T[0]), I_mean=float(I[:end + 1].mean()),
                         T_amb=float(c['ambient_temperature'])))
N = pd.DataFrame(rows)
N.to_csv(os.path.join('data', 'nasa_cycle_features.csv'), index=False, float_format='%.10g')
print('data/nasa_cycle_features.csv:', len(N), 'discharge cycles;', N.groupby('battery').size().to_dict())
