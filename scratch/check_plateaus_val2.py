import os
import glob
import numpy as np
import pandas as pd

base_dir = r"c:\Users\vicio\Documents\AEROPENDULO"
exp2_dir = os.path.join(base_dir, "data", "experimentos_novos", "EXP001_reteste_2_narx-mpc")
csvs = sorted(glob.glob(os.path.join(exp2_dir, "4_aeropendulo", "val2", "*.csv")))

plateaus = [
    ("Degrau 1 (71.4°)", 10.0, 14.0),
    ("Degrau 2 (95.0°)", 25.0, 31.0),
    ("Degrau 3 (37.1°)", 45.0, 51.0),
    ("Degrau 4 (51.7°)", 55.0, 62.0),
]

print("=== ERRO DE REGIME PERMANENTE NOS DEGRAUS (Val 2 - EXP001_reteste_2) ===")
for name, t0, t1 in plateaus:
    plateau_biases = []
    plateau_stds = []
    for f in csvs:
        df = pd.read_csv(f)
        t = df["tempo_ms"] / 1000.0
        m = (t >= t0) & (t <= t1)
        e = df["angulo_deg"].values[m] - df["referencia"].values[m]
        plateau_biases.append(np.mean(e))
        plateau_stds.append(np.std(e))
    
    print(f"{name:18s} | Erro Médio: {np.mean(plateau_biases):+.3f}° ± {np.std(plateau_biases):.3f}° | Ruído/Ripple: {np.mean(plateau_stds):.3f}°")

