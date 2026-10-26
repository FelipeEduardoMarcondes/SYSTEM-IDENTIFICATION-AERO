import pandas as pd
import matplotlib.pyplot as plt
import os

p = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_reteste_narx-mpc\4_aeropendulo'

f_sem1 = '20261023_180508_referencia_legado_val1_mpc_hybrid.csv'
f_com1 = '20261023_181157_referencia_legado_val1_mpc_hybrid.csv'

df_s = pd.read_csv(os.path.join(p, f_sem1))
df_c = pd.read_csv(os.path.join(p, f_com1))

print("Total points:", len(df_s), len(df_c))
print("Max time (s):", df_s['tempo_ms'].iloc[-1]/1000, df_c['tempo_ms'].iloc[-1]/1000)
print("Reference at ends:")
print(df_s[['tempo_ms', 'referencia', 'angulo_deg', 'u_pct']].iloc[::500])
