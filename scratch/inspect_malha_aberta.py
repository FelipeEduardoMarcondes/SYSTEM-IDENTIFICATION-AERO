import os
import pandas as pd

p = r'data/experimentos/coleta_dados/MALHA-ABERTA'
for f in sorted(os.listdir(p)):
    if f.endswith('.csv') and not f.startswith('malha_aberta_-'):
        fp = os.path.join(p, f)
        df = pd.read_csv(fp)
        u_col = 'u_pct' if 'u_pct' in df.columns else ('u' if 'u' in df.columns else None)
        y_col = 'angulo_deg' if 'angulo_deg' in df.columns else ('angulo' if 'angulo' in df.columns else None)
        t_col = 'tempo_ms' if 'tempo_ms' in df.columns else ('tempo_s' if 'tempo_s' in df.columns else ('t' if 't' in df.columns else None))
        
        u_min = df[u_col].min() if u_col else 'N/A'
        u_max = df[u_col].max() if u_col else 'N/A'
        y_min = df[y_col].min() if y_col else 'N/A'
        y_max = df[y_col].max() if y_col else 'N/A'
        dt = (df[t_col].iloc[1] - df[t_col].iloc[0]) if (t_col and len(df) > 1) else 'N/A'
        
        print(f"{f}: N={len(df)}, dt={dt}, cols={list(df.columns)}, u=[{u_min}, {u_max}], y=[{y_min}, {y_max}]")
