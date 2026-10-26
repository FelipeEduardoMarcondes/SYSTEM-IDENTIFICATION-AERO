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
    
    mask1 = (df1['tempo_ms'] >= 5000) & (df1['tempo_ms'] <= 75000)
    mask2 = (df2['tempo_ms'] >= 5000) & (df2['tempo_ms'] <= 75000)
    
    err1 = df1.loc[mask1, 'angulo_deg'] - df1.loc[mask1, 'referencia']
    err2 = df2.loc[mask2, 'angulo_deg'] - df2.loc[mask2, 'referencia']
    
    print(f'=== {name} ===')
    print(f'Set 1 (18:05-09): {f1}')
    print(f'  RMSE: {np.sqrt(np.mean(err1**2)):.3f} deg | Mean Error (Bias): {np.mean(err1):.3f} deg | Max Abs Error: {np.max(np.abs(err1)):.3f} deg')
    print(f'  Mean u: {df1.loc[mask1, "u_pct"].mean():.2f}% | Std u: {df1.loc[mask1, "u_pct"].std():.2f}%')
    
    print(f'Set 2 (18:11-17): {f2}')
    print(f'  RMSE: {np.sqrt(np.mean(err2**2)):.3f} deg | Mean Error (Bias): {np.mean(err2):.3f} deg | Max Abs Error: {np.max(np.abs(err2)):.3f} deg')
    print(f'  Mean u: {df2.loc[mask2, "u_pct"].mean():.2f}% | Std u: {df2.loc[mask2, "u_pct"].std():.2f}%')
    print()
