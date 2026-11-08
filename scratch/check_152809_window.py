import pandas as pd
import numpy as np

f_path = r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos_novos\EXP001_reteste_2_narx-mpc\4_aeropendulo\20261103_152809_degraus_hybrid.csv"
df = pd.read_csv(f_path)
t = df["tempo_ms"] / 1000.0

print("Tempo total:", t.min(), "a", t.max())

# Let's inspect around t=5s
for ti in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
    idx = np.argmin(np.abs(t - ti))
    print(f"t={t[idx]:.2f}s | theta={df['angulo_deg'][idx]:.2f} | ref={df['referencia'][idx]:.2f} | u={df['u_pct'][idx]:.2f}")

# Let's inspect around t=70s
for ti in [65.0, 68.0, 69.0, 70.0, 71.0]:
    idx = np.argmin(np.abs(t - ti))
    print(f"t={t[idx]:.2f}s | theta={df['angulo_deg'][idx]:.2f} | ref={df['referencia'][idx]:.2f} | u={df['u_pct'][idx]:.2f}")

