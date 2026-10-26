import pandas as pd
import numpy as np
import os

p = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_reteste_narx-mpc\4_aeropendulo'

pairs = [
    ('Val1', '20261023_180508_referencia_legado_val1_mpc_hybrid.csv', '20261023_181157_referencia_legado_val1_mpc_hybrid.csv'),
    ('Val2', '20261023_180711_referencia_legado_val2_degraus_hybrid.csv', '20261023_181401_referencia_legado_val2_degraus_hybrid.csv'),
    ('Val3', '20261023_180920_referencia_legado_val3_senoidal_hybrid.csv', '20261023_181719_referencia_legado_val3_senoidal_hybrid.csv')
]

for name, f1, f2 in pairs:
    df1 = pd.read_csv(os.path.join(p, f1))
    df2 = pd.read_csv(os.path.join(p, f2))
    min_len = min(len(df1), len(df2))
    ref_diff = np.max(np.abs(df1['referencia'].iloc[:min_len] - df2['referencia'].iloc[:min_len]))
    dt1 = np.diff(df1['tempo_ms']).mean()
    dt2 = np.diff(df2['tempo_ms']).mean()
    t1_end = df1['tempo_ms'].iloc[-1]/1000
    t2_end = df2['tempo_ms'].iloc[-1]/1000
    print(f'{name}: ref max abs diff = {ref_diff:.6f}, dt1 = {dt1:.2f}ms, dt2 = {dt2:.2f}ms, duration = {t1_end:.2f}s vs {t2_end:.2f}s')
