import os
import glob
import numpy as np
import pandas as pd

base_dir = r"c:\Users\vicio\Documents\AEROPENDULO"
exp1_dir = os.path.join(base_dir, "data", "experimentos_novos", "EXP001_reteste_narx-mpc")
exp2_dir = os.path.join(base_dir, "data", "experimentos_novos", "EXP001_reteste_2_narx-mpc")

def load_csv(path):
    df = pd.read_csv(path)
    # Ensure columns exist
    # time in seconds
    t = df["tempo_ms"] / 1000.0
    return df, t

# Active window: 5.0s to 70.0s (outside PID)
T_START = 5.0
T_END = 70.0

def calc_metrics(df, t_start=T_START, t_end=T_END):
    t = df["tempo_ms"] / 1000.0
    mask = (t >= t_start) & (t <= t_end)
    
    y = df["angulo_deg"].values[mask]
    r = df["referencia"].values[mask]
    u = df["u_pct"].values[mask]
    time_pts = t.values[mask]
    
    e = y - r
    rmse = np.sqrt(np.mean(e**2))
    mae = np.mean(np.abs(e))
    max_e = np.max(np.abs(e))
    std_e = np.std(e)
    mean_e = np.mean(e)
    
    # Control metrics
    u_mean = np.mean(u)
    u_std = np.std(u)
    du = np.abs(np.diff(u))
    tv_u = np.sum(du)
    sat_pct = np.mean((u <= -10.0) | (u >= 80.0)) * 100.0
    
    return {
        "rmse": rmse,
        "mae": mae,
        "max_e": max_e,
        "mean_e": mean_e,
        "std_e": std_e,
        "u_mean": u_mean,
        "u_std": u_std,
        "tv_u": tv_u,
        "sat_pct": sat_pct,
        "n_samples": len(y)
    }

print("=================================================================")
print("ANÁLISE ESTATÍSTICA: EXP001_reteste_2 vs EXP001_reteste (Janela 5 a 70s)")
print("=================================================================\n")

# Reteste 2 files
for val_name, folder_name in [("Val 1 (Multisseno+Degraus)", "val1"), 
                             ("Val 2 (Degraus)", "val2"), 
                             ("Val 3 (Senoidal)", "val3")]:
    val_path = os.path.join(exp2_dir, "4_aeropendulo", folder_name)
    csv_files = sorted(glob.glob(os.path.join(val_path, "*.csv")))
    
    metrics_list = []
    y_curves = []
    u_curves = []
    
    for f in csv_files:
        df, t = load_csv(f)
        m = calc_metrics(df)
        metrics_list.append(m)
        
        mask = (t >= T_START) & (t <= T_END)
        y_curves.append(df["angulo_deg"].values[mask])
        u_curves.append(df["u_pct"].values[mask])
    
    # Aggregate stats across runs
    df_metrics = pd.DataFrame(metrics_list)
    
    print(f"--- {val_name} (N = {len(csv_files)} repetições em reteste_2) ---")
    print(f"RMSE: {df_metrics['rmse'].mean():.4f}° ± {df_metrics['rmse'].std():.4f}° (min={df_metrics['rmse'].min():.4f}°, max={df_metrics['rmse'].max():.4f}°, CV={df_metrics['rmse'].std()/df_metrics['rmse'].mean()*100:.2f}%)")
    print(f"MAE:  {df_metrics['mae'].mean():.4f}° ± {df_metrics['mae'].std():.4f}°")
    print(f"Erro Médio (Viés): {df_metrics['mean_e'].mean():.4f}° ± {df_metrics['mean_e'].std():.4f}°")
    print(f"Erro Máx: {df_metrics['max_e'].mean():.4f}° ± {df_metrics['max_e'].std():.4f}°")
    print(f"TV(u): {df_metrics['tv_u'].mean():.2f} ± {df_metrics['tv_u'].std():.2f}")
    
    # Pointwise standard deviation
    min_len = min(len(c) for c in y_curves)
    y_arr = np.array([c[:min_len] for c in y_curves])
    u_arr = np.array([c[:min_len] for c in u_curves])
    
    sigma_y_pointwise = np.std(y_arr, axis=0)
    sigma_u_pointwise = np.std(u_arr, axis=0)
    
    print(f"Dispersão ponto a ponto sigma_y(t): média={np.mean(sigma_y_pointwise):.4f}°, máx={np.max(sigma_y_pointwise):.4f}°")
    print(f"Dispersão ponto a ponto sigma_u(t): média={np.mean(sigma_u_pointwise):.4f}%, máx={np.max(sigma_u_pointwise):.4f}%")
    print()

print("=================================================================")
print("COMPARAÇÃO COM EXP001_reteste (23/10/2026)")
print("=================================================================")

exp1_files = {
    "Val 1 Sem Integrador": os.path.join(exp1_dir, "4_aeropendulo", "20261023_180508_referencia_legado_val1_mpc_hybrid.csv"),
    "Val 1 Com Integrador": os.path.join(exp1_dir, "4_aeropendulo", "20261023_181157_referencia_legado_val1_mpc_hybrid.csv"),
    "Val 2 Sem Integrador": os.path.join(exp1_dir, "4_aeropendulo", "20261023_180711_referencia_legado_val2_degraus_hybrid.csv"),
    "Val 2 Com Integrador": os.path.join(exp1_dir, "4_aeropendulo", "20261023_181401_referencia_legado_val2_degraus_hybrid.csv"),
    "Val 3 Sem Integrador": os.path.join(exp1_dir, "4_aeropendulo", "20261023_180920_referencia_legado_val3_senoidal_hybrid.csv"),
    "Val 3 Com Integrador": os.path.join(exp1_dir, "4_aeropendulo", "20261023_181719_referencia_legado_val3_senoidal_hybrid.csv"),
}

for name, fpath in exp1_files.items():
    if os.path.exists(fpath):
        df, t = load_csv(fpath)
        m = calc_metrics(df)
        print(f"{name:25s} | RMSE: {m['rmse']:.4f}° | MAE: {m['mae']:.4f}° | Mean(e): {m['mean_e']:+.4f}° | Max|e|: {m['max_e']:.4f}°")

print("\n=================================================================")
print("SIL vs PYTHON vs REALIDADE (Val 1, 2, 3)")
print("=================================================================")
sil_files = {
    "Val 1 SIL": os.path.join(exp1_dir, "3_sil_stm32", "20261023_214321_referencia_legado_val1_mpc_hybrid.csv"),
    "Val 2 SIL": os.path.join(exp1_dir, "3_sil_stm32", "20261023_214546_referencia_legado_val2_degraus_hybrid.csv"),
    "Val 3 SIL": os.path.join(exp1_dir, "3_sil_stm32", "20261023_214749_referencia_legado_val3_senoidal_hybrid.csv"),
}
for name, fpath in sil_files.items():
    if os.path.exists(fpath):
        df, t = load_csv(fpath)
        m = calc_metrics(df)
        print(f"{name:25s} | RMSE: {m['rmse']:.4f}° | MAE: {m['mae']:.4f}° | Mean(e): {m['mean_e']:+.4f}° | Max|e|: {m['max_e']:.4f}°")

