import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.85",
    "figure.autolayout": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

base_dir = r"c:\Users\vicio\Documents\AEROPENDULO"
exp1_dir = os.path.join(base_dir, "data", "experimentos_novos", "EXP001_reteste_narx-mpc")
exp2_dir = os.path.join(base_dir, "data", "experimentos_novos", "EXP001_reteste_2_narx-mpc")

dest_dirs = [
    os.path.join(base_dir, "tcc", "img", "resultados"),
    os.path.join(exp2_dir, "plots_academicos")
]
for d in dest_dirs:
    os.makedirs(d, exist_ok=True)

T_START = 5.0
T_END = 70.0

configs = [
    {
        "id": 1,
        "title": "Validação 1 (Multisseno + Degraus)",
        "label": "val1",
        "file_nom": os.path.join(exp1_dir, "4_aeropendulo", "20261023_180508_referencia_legado_val1_mpc_hybrid.csv"),
        "file_exp1": os.path.join(exp1_dir, "4_aeropendulo", "20261023_181157_referencia_legado_val1_mpc_hybrid.csv"),
        "folder_exp2": os.path.join(exp2_dir, "4_aeropendulo", "val1"),
        "y_lim": [15, 75],
    },
    {
        "id": 2,
        "title": "Validação 2 (Degraus Aleatórios)",
        "label": "val2",
        "file_nom": os.path.join(exp1_dir, "4_aeropendulo", "20261023_180711_referencia_legado_val2_degraus_hybrid.csv"),
        "file_exp1": os.path.join(exp1_dir, "4_aeropendulo", "20261023_181401_referencia_legado_val2_degraus_hybrid.csv"),
        "folder_exp2": os.path.join(exp2_dir, "4_aeropendulo", "val2"),
        "y_lim": [10, 115],
    },
    {
        "id": 3,
        "title": "Validação 3 (Senoidal Suave)",
        "label": "val3",
        "file_nom": os.path.join(exp1_dir, "4_aeropendulo", "20261023_180920_referencia_legado_val3_senoidal_hybrid.csv"),
        "file_exp1": os.path.join(exp1_dir, "4_aeropendulo", "20261023_181719_referencia_legado_val3_senoidal_hybrid.csv"),
        "folder_exp2": os.path.join(exp2_dir, "4_aeropendulo", "val3"),
        "y_lim": [20, 110],
    }
]

boxplot_data = []
nom_rmse_vals = []

for cfg in configs:
    # Nominal
    df_nom = pd.read_csv(cfg["file_nom"])
    t_nom = df_nom["tempo_ms"] / 1000.0
    m_nom = (t_nom >= T_START) & (t_nom <= T_END)
    e_nom = df_nom["angulo_deg"].values[m_nom] - df_nom["referencia"].values[m_nom]
    rmse_nom = np.sqrt(np.mean(e_nom**2))
    nom_rmse_vals.append(rmse_nom)
    
    # 10 combined files (1 from exp1 + 9 from exp2)
    files_exp2 = sorted(glob.glob(os.path.join(cfg["folder_exp2"], "*.csv")))[:9]
    combined_files = [cfg["file_exp1"]] + files_exp2
    
    y_runs = []
    u_runs = []
    t_ref = None
    r_ref = None
    rmses = []
    
    for f in combined_files:
        df = pd.read_csv(f)
        t = df["tempo_ms"] / 1000.0
        mask = (t >= T_START) & (t <= T_END)
        y = df["angulo_deg"].values[mask]
        u = df["u_pct"].values[mask]
        r = df["referencia"].values[mask]
        t_pts = t.values[mask]
        
        if t_ref is None or len(t_pts) < len(t_ref):
            t_ref = t_pts
            r_ref = r
            
        y_runs.append(y)
        u_runs.append(u)
        rmses.append(np.sqrt(np.mean((y - r)**2)))
        
    boxplot_data.append(rmses)
    
    min_len = len(t_ref)
    y_mat = np.array([y[:min_len] for y in y_runs])
    u_mat = np.array([u[:min_len] for u in u_runs])
    
    y_mean = np.mean(y_mat, axis=0)
    y_std = np.std(y_mat, axis=0, ddof=1)
    
    u_mean = np.mean(u_mat, axis=0)
    u_std = np.std(u_mat, axis=0, ddof=1)
    
    # Generate shadow plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    # Upper subplot: Angle
    ax1.plot(t_ref, r_ref[:min_len], color="black", linestyle="--", linewidth=1.1, label="Referência $r(t)$")
    ax1.plot(t_ref, y_mean, color="#d62728", linestyle="-", linewidth=1.2, label="Média $\\mu_\\theta(t)$ ($N=10$)")
    ax1.fill_between(t_ref, y_mean - y_std, y_mean + y_std, color="#d62728", alpha=0.25, label="Faixa $\\pm 1\\sigma_\\theta(t)$")
    ax1.set_ylabel("Ângulo $\\theta$ [°]")
    ax1.set_ylim(cfg["y_lim"])
    ax1.grid(True)
    ax1.legend(loc="upper right" if cfg["id"] == 2 else "lower right", framealpha=0.95)
    
    # Lower subplot: Control
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
    ax2.plot(t_ref, u_mean, color="#1f77b4", linestyle="-", linewidth=1.1, label="Controle Médio $\\mu_u(t)$")
    ax2.fill_between(t_ref, u_mean - u_std, u_mean + u_std, color="#1f77b4", alpha=0.25, label="Faixa $\\pm 1\\sigma_u(t)$")
    ax2.set_xlabel("Tempo [s] (Janela Ativa: 5 a 70 s)")
    ax2.set_ylabel("Controle $u$ [%]")
    ax2.set_ylim(-10, 90)
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.95)
    
    ax1.set_xlim(T_START, T_END)
    plt.tight_layout()
    
    plot_name = f"curva_media_sombra_{cfg['label']}"
    for d in dest_dirs:
        fig.savefig(os.path.join(d, f"{plot_name}.png"), dpi=300)
        fig.savefig(os.path.join(d, f"{plot_name}.pdf"), format="pdf")
    plt.close(fig)
    print(f"Salvo: {plot_name}")

# Generate Boxplot
fig, ax = plt.subplots(figsize=(7.2, 4.0))

positions = [1, 2, 3]
labels = ["Val 1\n(Multisseno+Degraus)", "Val 2\n(Degraus Aleat.)", "Val 3\n(Senoidal Suave)"]

# Boxplot of N=10 hybrid runs
bp = ax.boxplot(boxplot_data, positions=positions, widths=0.42, patch_artist=True,
                showmeans=True, meanline=False,
                boxprops=dict(facecolor="#d62728", color="#8b0000", alpha=0.7, linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.5),
                meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white", markersize=5),
                whiskerprops=dict(color="#8b0000", linewidth=1.0),
                capprops=dict(color="#8b0000", linewidth=1.0),
                flierprops=dict(marker="o", color="#8b0000", alpha=0.6, markersize=4))

# Overlay individual jitter points
for i, rmses in enumerate(boxplot_data):
    x = np.random.normal(positions[i], 0.04, size=len(rmses))
    ax.scatter(x, rmses, color="#8b0000", alpha=0.6, s=25, zorder=3)

# Overlay Nominal markers (single run from 23/10)
sc_nom = ax.scatter(positions, nom_rmse_vals, color="#1f77b4", marker="D", s=70, zorder=4,
                    label="Bancada Nominal (Sem Integrador)")

# Proxy legend handles
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    sc_nom,
    Patch(facecolor="#d62728", edgecolor="#8b0000", alpha=0.7, label="Bancada Híbrida ($N=10$ repetições)"),
    Line2D([0], [0], color="black", linewidth=1.5, label="Mediana"),
    Line2D([0], [0], marker="o", color="w", markeredgecolor="black", markerfacecolor="white", markersize=5, label="Média")
]

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel("RMSE Angular [°] (Janela Ativa: 5 a 70 s)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(handles=legend_elements, loc="upper left", framealpha=0.95)

plt.tight_layout()
for d in dest_dirs:
    fig.savefig(os.path.join(d, "boxplot_reteste_estatistico.png"), dpi=300)
    fig.savefig(os.path.join(d, "boxplot_reteste_estatistico.pdf"), format="pdf")
plt.close(fig)
print("Salvo: boxplot_reteste_estatistico")

