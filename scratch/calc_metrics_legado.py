import os
import pandas as pd
import numpy as np

p = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_reteste_narx-mpc\4_aeropendulo'

pairs = [
    ("Validação 1 (Multisseno)", 
     "20261023_180508_referencia_legado_val1_mpc_hybrid.csv",
     "20261023_181157_referencia_legado_val1_mpc_hybrid.csv"),
    ("Validação 2 (Degraus)", 
     "20261023_180711_referencia_legado_val2_degraus_hybrid.csv",
     "20261023_181401_referencia_legado_val2_degraus_hybrid.csv"),
    ("Validação 3 (Senoidal)", 
     "20261023_180920_referencia_legado_val3_senoidal_hybrid.csv",
     "20261023_181719_referencia_legado_val3_senoidal_hybrid.csv")
]

windows = [
    ("Ativa (5s a 70s)", 5000, 70000),
    ("Ativa estendida (5s a 75s)", 5000, 75000),
    ("Ativa total (5s ao fim)", 5000, 83000),
    ("Completa (0s ao fim)", 0, 85000)
]

for wname, t_start, t_end in windows:
    print(f"==================================================")
    print(f"JANELA: {wname}")
    print(f"==================================================")
    for name, f_sem, f_com in pairs:
        df_s = pd.read_csv(os.path.join(p, f_sem))
        df_c = pd.read_csv(os.path.join(p, f_com))
        
        m_s = (df_s['tempo_ms'] >= t_start) & (df_s['tempo_ms'] <= t_end)
        m_c = (df_c['tempo_ms'] >= t_start) & (df_c['tempo_ms'] <= t_end)
        
        e_s = df_s.loc[m_s, 'angulo_deg'] - df_s.loc[m_s, 'referencia']
        e_c = df_c.loc[m_c, 'angulo_deg'] - df_c.loc[m_c, 'referencia']
        
        rmse_s = np.sqrt(np.mean(e_s**2))
        rmse_c = np.sqrt(np.mean(e_c**2))
        bias_s = np.mean(e_s)
        bias_c = np.mean(e_c)
        reduc_rmse = (1.0 - rmse_c / rmse_s) * 100
        reduc_bias = (1.0 - abs(bias_c) / abs(bias_s)) * 100
        
        du_s = np.mean(np.abs(np.diff(df_s.loc[m_s, 'u_pct'])))
        du_c = np.mean(np.abs(np.diff(df_c.loc[m_c, 'u_pct'])))
        
        print(f"--- {name} ---")
        print(f"  SEM Integrador : RMSE = {rmse_s:.2f}° | Bias = {bias_s:+.2f}° | delta_u = {du_s:.3f}%")
        print(f"  COM Integrador : RMSE = {rmse_c:.2f}° | Bias = {bias_c:+.2f}° | delta_u = {du_c:.3f}%")
        print(f"  Melhoria       : RMSE {reduc_rmse:+.1f}% | Bias {reduc_bias:+.1f}%")
    print()
