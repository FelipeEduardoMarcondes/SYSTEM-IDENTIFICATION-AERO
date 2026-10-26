import pandas as pd
import numpy as np
import os

p_old = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_legado_narx-mpc\4_aeropendulo\20261006_181737_referencia_mpc.csv'
df_old = pd.read_csv(p_old)

print("=== 20261006_181737 ===")
print("Columns:", df_old.columns)
print("Points:", len(df_old))
print("First 15 rows:")
print(df_old.head(15))
print("At t=66 to 70s:")
print(df_old[(df_old['tempo_ms'] >= 65000) & (df_old['tempo_ms'] <= 71000)].iloc[::50])
