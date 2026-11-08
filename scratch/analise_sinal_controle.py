import os
import glob
import numpy as np
import pandas as pd

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
    
    u_list = []
    u_mean_list = []
    u_std_list = []
    u_min_list = []
    u_max_list = []
    tv_list = []
    delta_u_mean_list = []
    delta_u_max_list = []
    
    for df in dfs:
        t = df["tempo_ms"] / 1000.0
        mask = (t >= T_START) & (t <= T_END)
        u = df["u_pct"].values[mask]
        u_list.append(u)
        
        u_mean_list.append(np.mean(u))
        u_std_list.append(np.std(u))
        u_min_list.append(np.min(u))
        u_max_list.append(np.max(u))
        
        du = np.abs(np.diff(u))
        tv_list.append(np.sum(du))
        delta_u_mean_list.append(np.mean(du))
        delta_u_max_list.append(np.max(du))
        
    min_len = min(len(u) for u in u_list)
    u_mat = np.array([u[:min_len] for u in u_list])
    
    # Pointwise standard deviation across runs at each time step
    sigma_u_t = np.std(u_mat, axis=0, ddof=1)
    
    # Repeatability correlation between u curves
    u_mean_curve = np.mean(u_mat, axis=0)
    u_corrs = [np.corrcoef(u_mat[i], u_mean_curve)[0, 1] for i in range(len(dfs))]
    
    # Sem integrador vs Com integrador (EXP001_reteste)
    df_sem = pd.read_csv(f_sem)
    t_sem = df_sem["tempo_ms"] / 1000.0
    u_sem = df_sem["u_pct"].values[(t_sem >= T_START) & (t_sem <= T_END)]
    tv_sem = np.sum(np.abs(np.diff(u_sem)))
    
    df_com = pd.read_csv(f_com)
    t_com = df_com["tempo_ms"] / 1000.0
    u_com = df_com["u_pct"].values[(t_com >= T_START) & (t_com <= T_END)]
    tv_com = np.sum(np.abs(np.diff(u_com)))
    
    print(f"==================================================")
    print(f"ANÁLISE DO SINAL DE CONTROLE: {title}")
    print(f"==================================================")
    print(f"1. Faixa e Excursão Operacional (Média ± Std):")
    print(f"   - Valor Médio u: {np.mean(u_mean_list):.2f}% ± {np.std(u_mean_list):.2f}%")
    print(f"   - Mínimo atingido: {np.mean(u_min_list):.2f}% (extremo: {np.min(u_min_list):.2f}%)")
    print(f"   - Máximo atingido: {np.mean(u_max_list):.2f}% (extremo: {np.max(u_max_list):.2f}%)")
    print(f"   - Desvio-padrão temporal médio (dinâmica): {np.mean(u_std_list):.2f}%")
    print()
    print(f"2. Variabilidade Inter-Repetições (Dispersão entre os ensaios):")
    print(f"   - Desvio-padrão ponto a ponto sigma_u(t): Média = {np.mean(sigma_u_t):.3f}%, Máx = {np.max(sigma_u_t):.3f}%")
    print(f"   - Correlação Pearson de u(t) com a média: {np.mean(u_corrs):.5f} (min={np.min(u_corrs):.5f})")
    print()
    print(f"3. Suavidade e Chattering (|Deltau| a cada Ts = 10 ms):")
    print(f"   - Variação Total TV(u): {np.mean(tv_list):.1f} ± {np.std(tv_list):.1f}")
    print(f"   - Degrau médio por ciclo de 10ms (|Deltau|): {np.mean(delta_u_mean_list):.4f}% por ciclo")
    print(f"   - Maior degrau instantâneo max(|Deltau|): {np.mean(delta_u_max_list):.2f}%")
    print()
    print(f"4. Comparativo de Agressividade (TV):")
    print(f"   - TV(u) Sem Integrador (23/10): {tv_sem:.1f}")
    print(f"   - TV(u) Com Integrador (23/10): {tv_com:.1f}")
    print(f"   - TV(u) Reteste 2 (Média hoje): {np.mean(tv_list):.1f}")
    print()
