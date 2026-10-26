import os
import pandas as pd
import numpy as np

# Find CSV where:
# 1. At t between 66s and 70s (66000 - 70000 ms), angulo_deg stays around 40-45 deg!
# (while the reference jumps to ~68 deg)
# 2. At t near 1s, angulo_deg peaks to > 80 deg.

root = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main'
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if f.endswith('.csv'):
            p = os.path.join(dirpath, f)
            try:
                df = pd.read_csv(p)
                if 'angulo_deg' in df and 'tempo_ms' in df:
                    # check first 2s peak
                    m_start = (df['tempo_ms'] <= 2000)
                    if df.loc[m_start, 'angulo_deg'].max() > 75:
                        m_late = (df['tempo_ms'] >= 66000) & (df['tempo_ms'] <= 70000)
                        if m_late.sum() > 0:
                            mean_late = df.loc[m_late, 'angulo_deg'].mean()
                            max_late = df.loc[m_late, 'angulo_deg'].max()
                            print(f"Match: {p} -> max first 2s: {df.loc[m_start, 'angulo_deg'].max():.1f}, late mean: {mean_late:.1f}, late max: {max_late:.1f}")
            except Exception:
                pass
