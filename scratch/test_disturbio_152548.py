import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = r"c:\Users\vicio\Documents\AEROPENDULO\data\experimentos_novos\EXP001_reteste_2_narx-mpc\4_aeropendulo\20261103_152548_degraus_hybrid.csv"
df = pd.read_csv(f)
t = df["tempo_ms"] / 1000.0

# Active window 5 to 70s
mask = (t >= 5.0) & (t <= 70.0)
t_w = t[mask].values
y_w = df["angulo_deg"][mask].values
r_w = df["referencia"][mask].values
u_w = df["u_pct"][mask].values

print("Duração:", t_w[0], "a", t_w[-1])
print("Max desvio positivo:", np.max(y_w - 60))
print("Max desvio negativo:", np.min(y_w - 60))
print("Erro estático fora dos distúrbios (ex: 45 a 65s):", np.mean(y_w[(t_w >= 45) & (t_w <= 65)] - 60))

