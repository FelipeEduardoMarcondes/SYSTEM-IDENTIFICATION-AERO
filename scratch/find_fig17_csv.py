import os
import pandas as pd
import numpy as np

# Let's find which CSV matches the red curve in Figura 17:
# Characteristics of that curve:
# 1. At t near 1.0s, angulo_deg reaches ~88 deg
# 2. At t near 4s, angulo_deg settles near 45 deg
# 3. At t near 50s, angulo_deg drops to ~39 deg
# 4. At t near 66s, angulo_deg stays around 42 deg (does NOT follow step to 68 deg)

root = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main'
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if f.endswith('.csv'):
            p = os.path.join(dirpath, f)
            try:
                df = pd.read_csv(p)
                if 'angulo_deg' in df and 'tempo_ms' in df:
                    # check if max in first 2s is between 80 and 95
                    m1 = (df['tempo_ms'] <= 2000)
                    if df.loc[m1, 'angulo_deg'].max() > 80:
                        print("Candidate with >80 deg in first 2s:", p)
            except Exception:
                pass
