import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

base_dir = r"c:\Users\vicio\Documents\AEROPENDULO"
exp1_dir = os.path.join(base_dir, "data", "experimentos_novos", "EXP001_reteste_narx-mpc")
exp2_dir = os.path.join(base_dir, "data", "experimentos_novos", "EXP001_reteste_2_narx-mpc")

T_START = 5.0
T_END = 70.0

vals = [
    ("Val 1 (Multisseno+Degraus)", "val1", 
     os.path.join(exp1_dir, "4_aeropendulo", "20261023_180508_referencia_legado_val1_mpc_hybrid.csv"),
     os.path.join(exp1_dir, "4_aeropendulo", "20261023_181157_referencia_legado_val1_mpc_hybrid.csv")),
    ("Val 2 (Degraus)", "val2", 
     os.path.join(exp1_dir, "4_aeropendulo", "20261023_180711_referencia_legado_val2_degraus_hybrid.csv"),
     os.path.join(exp1_dir, "4_aeropendulo", "20261023_181401_referencia_legado_val2_degraus_hybrid.csv")),
    ("Val 3 (Senoidal)", "val3", 
     os.path.join(exp1_dir, "4_aeropendulo", "20261023_180920_referencia_legado_val3_senoidal_hybrid.csv"),
     os.path.join(exp1_dir, "4_aeropendulo", "20261023_181719_referencia_legado_val3_senoidal_hybrid.csv")),
]

for title, fld, f_sem, f_com in vals:
    csvs = sorted(glob.glob(os.path.join(exp2_dir, "4_aeropendulo", fld, "*.csv")))
    dfs = [pd.read_csv(f) for f in csvs]
    
    # Trim to 5-70s
    t0 = 5.0
    t1 = 70.0
    
    y_list = []
    u_list = []
    rmse_list = []
    mae_list = []
    bias_list = []
    
    for df in dfs:
        t = df["tempo_ms"] / 1000.0
        mask = (t >= t0) & (t <= t1)
        y = df["angulo_deg"].values[mask]
        r = df["referencia"].values[mask]
        u = df["u_pct"].values[mask]
        y_list.append(y)
        u_list.append(u)
        
        e = y - r
        rmse_list.append(np.sqrt(np.mean(e**2)))
        mae_list.append(np.mean(np.abs(e)))
        bias_list.append(np.mean(e))
        
    min_len = min(len(y) for y in y_list)
    y_mat = np.array([y[:min_len] for y in y_list])
    u_mat = np.array([u[:min_len] for u in y_list])
    
    y_mean = np.mean(y_mat, axis=0)
    y_std = np.std(y_mat, axis=0, ddof=1)
    
    # Correlation of each run with the mean curve
    corrs = [np.corrcoef(y_mat[i], y_mean)[0, 1] for i in range(len(dfs))]
    # R2 with mean curve
    ss_tot = np.sum((y_mean - np.mean(y_mean))**2)
    r2_to_mean = [1.0 - (np.sum((y_mat[i] - y_mean)**2) / ss_tot) for i in range(len(dfs))]
    
    # 95% Confidence Interval for RMSE
    n = len(rmse_list)
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list, ddof=1)
    ci95_rmse = stats.t.ppf(0.975, df=n-1) * (std_rmse / np.sqrt(n))
    
    # Old com integrador
    df_old_com = pd.read_csv(f_com)
    t_old = df_old_com["tempo_ms"] / 1000.0
    m_old = (t_old >= t0) & (t_old <= t1)
    e_old_com = df_old_com["angulo_deg"].values[m_old] - df_old_com["referencia"].values[m_old]
    rmse_old_com = np.sqrt(np.mean(e_old_com**2))
    
    # Old sem integrador
    df_old_sem = pd.read_csv(f_sem)
    t_old_s = df_old_sem["tempo_ms"] / 1000.0
    m_old_s = (t_old_s >= t0) & (t_old_s <= t1)
    e_old_sem = df_old_sem["angulo_deg"].values[m_old_s] - df_old_sem["referencia"].values[m_old_s]
    rmse_old_sem = np.sqrt(np.mean(e_old_sem**2))
    
    # Z-score of old_com relative to reteste_2 distribution
    z_score_old = (rmse_old_com - mean_rmse) / std_rmse
    
    print(f"=== {title} ===")
    print(f"Número de repetições (N): {n}")
    print(f"RMSE: {mean_rmse:.4f}° ± {std_rmse:.4f}° [IC 95%: {mean_rmse - ci95_rmse:.4f}° a {mean_rmse + ci95_rmse:.4f}°]")
    print(f"CV(RMSE): {(std_rmse / mean_rmse) * 100:.2f}%")
    print(f"MAE: {np.mean(mae_list):.4f}° ± {np.std(mae_list, ddof=1):.4f}°")
    print(f"Viés Estático (Mean e): {np.mean(bias_list):+.4f}° ± {np.std(bias_list, ddof=1):.4f}°")
    print(f"Correlação Pearson com a curva média (min/méd/max): {np.min(corrs):.5f} / {np.mean(corrs):.5f} / {np.max(corrs):.5f}")
    print(f"R² com a curva média (min/méd/max): {np.min(r2_to_mean):.5f} / {np.mean(r2_to_mean):.5f} / {np.max(r2_to_mean):.5f}")
    print(f"Desvio-padrão ponto a ponto médio (sigma_y(t)): {np.mean(y_std):.4f}° (máximo: {np.max(y_std):.4f}°)")
    print(f"EXP001_reteste (23/10) Com Integrador: RMSE = {rmse_old_com:.4f}° (Z-score relativo a hoje = {z_score_old:+.2f})")
    print(f"EXP001_reteste (23/10) Sem Integrador: RMSE = {rmse_old_sem:.4f}° (Redução de erro com o integrador hoje = {(1 - mean_rmse/rmse_old_sem)*100:.1f}%)")
    print()

