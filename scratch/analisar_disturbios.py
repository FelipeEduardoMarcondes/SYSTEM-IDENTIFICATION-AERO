import os
import numpy as np
import pandas as pd

f1 = r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos_novos\EXP001_reteste_2_narx-mpc\4_aeropendulo\20261103_152548_degraus_hybrid.csv"
f2 = r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos_novos\EXP001_reteste_2_narx-mpc\4_aeropendulo\20261103_152809_degraus_hybrid.csv"
f3 = r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos_novos\EXP001_reteste_2_narx-mpc\4_aeropendulo\20261103_153004_degraus_hybrid.csv"

for f in [f1, f2, f3]:
    df = pd.read_csv(f)
    name = os.path.basename(f)
    t = df["tempo_ms"] / 1000.0
    # Active range 5 to 70s
    mask = (t >= 5.0) & (t <= 70.0)
    y = df["angulo_deg"].values[mask]
    r = df["referencia"].values[mask]
    u = df["u_pct"].values[mask]
    t_w = t.values[mask]
    
    e = y - r
    print(f"=== {name} ===")
    print(f"Ref: {np.median(r):.1f}° | Duração ativa: {t_w[-1]-t_w[0]:.1f}s")
    print(f"y: min={np.min(y):.1f}°, max={np.max(y):.1f}°")
    print(f"u: min={np.min(u):.1f}%, max={np.max(u):.1f}%, mean={np.mean(u):.1f}%")
    print(f"RMSE (com distúrbios na janela): {np.sqrt(np.mean(e**2)):.2f}°")
    print()

