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

# 1 from exp1 (com integrador) + 9 from exp2
val_configs = [
    {
        "name": "Val 1 (Multisseno + Degraus)",
        "label": "val1",
        "nominal_file": os.path.join(exp1_dir, "4_aeropendulo", "20261023_180508_referencia_legado_val1_mpc_hybrid.csv"),
        "exp1_hybrid": os.path.join(exp1_dir, "4_aeropendulo", "20261023_181157_referencia_legado_val1_mpc_hybrid.csv"),
        "exp2_folder": os.path.join(exp2_dir, "4_aeropendulo", "val1"),
    },
    {
        "name": "Val 2 (Degraus Aleatórios)",
        "label": "val2",
        "nominal_file": os.path.join(exp1_dir, "4_aeropendulo", "20261023_180711_referencia_legado_val2_degraus_hybrid.csv"),
        "exp1_hybrid": os.path.join(exp1_dir, "4_aeropendulo", "20261023_181401_referencia_legado_val2_degraus_hybrid.csv"),
        "exp2_folder": os.path.join(exp2_dir, "4_aeropendulo", "val2"),
    },
    {
        "name": "Val 3 (Senoidal Suave)",
        "label": "val3",
        "nominal_file": os.path.join(exp1_dir, "4_aeropendulo", "20261023_180920_referencia_legado_val3_senoidal_hybrid.csv"),
        "exp1_hybrid": os.path.join(exp1_dir, "4_aeropendulo", "20261023_181719_referencia_legado_val3_senoidal_hybrid.csv"),
        "exp2_folder": os.path.join(exp2_dir, "4_aeropendulo", "val3"),
    }
]

def analyze_file(filepath):
    df = pd.read_csv(filepath)
    t = df["tempo_ms"] / 1000.0
    mask = (t >= T_START) & (t <= T_END)
    y = df["angulo_deg"].values[mask]
    r = df["referencia"].values[mask]
    u = df["u_pct"].values[mask]
    
    e = y - r
    rmse = np.sqrt(np.mean(e**2))
    mae = np.mean(np.abs(e))
    bias = np.mean(e)
    max_e = np.max(np.abs(e))
    
    du = np.abs(np.diff(u))
    tv_u = np.sum(du)
    delta_u_mean = np.mean(du)
    u_mean = np.mean(u)
    u_std = np.std(u)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "max_e": max_e,
        "tv_u": tv_u,
        "delta_u_mean": delta_u_mean,
        "u_mean": u_mean,
        "u_std": u_std,
        "y": y,
        "u": u,
        "r": r
    }

results = {}

for vc in val_configs:
    # Nominal
    nom_res = analyze_file(vc["nominal_file"])
    
    # 1 from exp1 + 9 from exp2
    exp2_files = sorted(glob.glob(os.path.join(vc["exp2_folder"], "*.csv")))[:9]
    combined_files = [vc["exp1_hybrid"]] + exp2_files
    assert len(combined_files) == 10, f"Expected 10 files, got {len(combined_files)}"
    
    hybrid_res_list = [analyze_file(f) for f in combined_files]
    
    rmses = [h["rmse"] for h in hybrid_res_list]
    maes = [h["mae"] for h in hybrid_res_list]
    biases = [h["bias"] for h in hybrid_res_list]
    max_es = [h["max_e"] for h in hybrid_res_list]
    tv_us = [h["tv_u"] for h in hybrid_res_list]
    delta_us = [h["delta_u_mean"] for h in hybrid_res_list]
    u_means = [h["u_mean"] for h in hybrid_res_list]
    u_stds = [h["u_std"] for h in hybrid_res_list]
    
    min_len = min(len(h["y"]) for h in hybrid_res_list)
    y_mat = np.array([h["y"][:min_len] for h in hybrid_res_list])
    y_mean = np.mean(y_mat, axis=0)
    sigma_y_t = np.std(y_mat, axis=0, ddof=1)
    
    corrs = [np.corrcoef(y_mat[i], y_mean)[0, 1] for i in range(10)]
    ss_tot = np.sum((y_mean - np.mean(y_mean))**2)
    r2s = [1.0 - (np.sum((y_mat[i] - y_mean)**2) / ss_tot) for i in range(10)]
    
    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses, ddof=1)
    ci95_rmse = stats.t.ppf(0.975, df=9) * (std_rmse / np.sqrt(10))
    
    # Reductions
    red_rmse = (1.0 - mean_rmse / nom_res["rmse"]) * 100.0
    red_bias = (1.0 - np.abs(np.mean(biases)) / np.abs(nom_res["bias"])) * 100.0
    
    results[vc["label"]] = {
        "name": vc["name"],
        "nom_rmse": nom_res["rmse"],
        "nom_mae": nom_res["mae"],
        "nom_bias": nom_res["bias"],
        "nom_tv": nom_res["tv_u"],
        "nom_delta_u": nom_res["delta_u_mean"],
        "rmses": rmses,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "ci95_rmse": ci95_rmse,
        "cv_rmse": (std_rmse / mean_rmse) * 100.0,
        "mean_mae": np.mean(maes),
        "std_mae": np.std(maes, ddof=1),
        "mean_bias": np.mean(biases),
        "std_bias": np.std(biases, ddof=1),
        "mean_max_e": np.mean(max_es),
        "mean_tv": np.mean(tv_us),
        "std_tv": np.std(tv_us, ddof=1),
        "mean_delta_u": np.mean(delta_us),
        "mean_u": np.mean(u_means),
        "mean_u_std": np.mean(u_stds),
        "sigma_y_mean": np.mean(sigma_y_t),
        "sigma_y_max": np.max(sigma_y_t),
        "r2_mean": np.mean(r2s),
        "corr_mean": np.mean(corrs),
        "red_rmse": red_rmse,
        "red_bias": red_bias,
    }

for k, v in results.items():
    print(f"=== {v['name']} (N = 10) ===")
    print(f"RMSE Nominal: {v['nom_rmse']:.2f}° | RMSE Híbrido: {v['mean_rmse']:.2f}° ± {v['std_rmse']:.2f}° [IC 95%: {v['mean_rmse']-v['ci95_rmse']:.2f}° a {v['mean_rmse']+v['ci95_rmse']:.2f}°]")
    print(f"CV(RMSE): {v['cv_rmse']:.2f}% | R² médio com curva média: {v['r2_mean']:.5f}")
    print(f"MAE: {v['mean_mae']:.2f}° ± {v['std_mae']:.2f}°")
    print(f"Viés: Nominal = {v['nom_bias']:+.2f}° -> Híbrido = {v['mean_bias']:+.2f}° ± {v['std_bias']:.2f}° (Redução: {v['red_bias']:.1f}%)")
    print(f"Redução RMSE: {v['red_rmse']:.1f}%")
    print(f"Dispersão ponto a ponto sigma_y: média = {v['sigma_y_mean']:.3f}°, máx = {v['sigma_y_max']:.3f}°")
    print(f"TV(u): Nominal = {v['nom_tv']:.1f} -> Híbrido = {v['mean_tv']:.1f} ± {v['std_tv']:.1f}")
    print(f"Delta_u por ciclo de 10ms: Nominal = {v['nom_delta_u']:.4f}% -> Híbrido = {v['mean_delta_u']:.4f}%")
    print(f"Valores individuais RMSE ({k}): {[round(x, 4) for x in v['rmses']]}")
    print()

