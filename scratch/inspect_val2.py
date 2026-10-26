import pandas as pd
import os

p = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_reteste_narx-mpc\4_aeropendulo'

files = [
    '20261022_205643_referencia_val2_degraus_hybrid.csv',
    '20261023_175425_referencia_val2_degraus_hybrid.csv',
    '20261023_180711_referencia_legado_val2_degraus_hybrid.csv',
    '20261023_181401_referencia_legado_val2_degraus_hybrid.csv',
]

for f in files:
    if os.path.exists(os.path.join(p, f)):
        df = pd.read_csv(os.path.join(p, f))
        print(f"=== {f} ===")
        print("Length:", len(df), "Max time:", df['tempo_ms'].iloc[-1]/1000)
        # sample some points: at t = 2s, 10s, 16s, 25s, 38s, 48s, 58s, 65s
        for t_target in [2, 10, 16, 25, 38, 48, 58, 65]:
            idx = (df['tempo_ms'] - t_target*1000).abs().idxmin()
            print(f"  t={df.loc[idx, 'tempo_ms']/1000:.1f}s: ref={df.loc[idx, 'referencia']:.1f}, theta={df.loc[idx, 'angulo_deg']:.1f}, u={df.loc[idx, 'u_pct']:.1f}")
