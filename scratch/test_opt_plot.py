import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def setup_academic():
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
        "lines.linewidth": 1.2,
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

setup_academic()

p = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_reteste_narx-mpc\4_aeropendulo'
out_dir = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\scratch'

comparacoes = [
    {
        "id": 1,
        "name": "opt_val1",
        "f_sem": os.path.join(p, "20261023_180508_referencia_legado_val1_mpc_hybrid.csv"),
        "f_com": os.path.join(p, "20261023_181157_referencia_legado_val1_mpc_hybrid.csv"),
        "leg1_loc": "lower right",
        "leg1_ncol": 3,
        "leg2_loc": "upper right",
        "leg2_ncol": 2
    },
    {
        "id": 2,
        "name": "opt_val2",
        "f_sem": os.path.join(p, "20261023_180711_referencia_legado_val2_degraus_hybrid.csv"),
        "f_com": os.path.join(p, "20261023_181401_referencia_legado_val2_degraus_hybrid.csv"),
        "leg1_loc": "upper right",
        "leg1_ncol": 1,
        "leg2_loc": "upper right",
        "leg2_ncol": 2
    },
    {
        "id": 3,
        "name": "opt_val3",
        "f_sem": os.path.join(p, "20261023_180920_referencia_legado_val3_senoidal_hybrid.csv"),
        "f_com": os.path.join(p, "20261023_181719_referencia_legado_val3_senoidal_hybrid.csv"),
        "leg1_loc": "lower right",
        "leg1_ncol": 3,
        "leg2_loc": "upper right",
        "leg2_ncol": 2
    }
]

for c in comparacoes:
    df_s = pd.read_csv(c["f_sem"])
    df_c = pd.read_csv(c["f_com"])
    
    t_s = df_s["tempo_ms"] / 1000.0
    t_c = df_c["tempo_ms"] / 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    # Subplot 1: Ângulo theta
    ax1.plot(t_c, df_c["referencia"], color="black", linestyle="--", linewidth=1.1, alpha=0.85, label="Referência")
    ax1.plot(t_s, df_s["angulo_deg"], color="#1f77b4", linestyle="-", linewidth=1.2, alpha=0.85, label="Bancada Sem Integrador (Nominal)")
    ax1.plot(t_c, df_c["angulo_deg"], color="#d62728", linestyle="-", linewidth=1.3, alpha=0.9, label="Bancada Com Integrador (Híbrido)")
    
    ax1.set_ylabel(r"Ângulo $\theta$ [$^\circ$]")
    ax1.grid(True)
    ax1.legend(loc=c["leg1_loc"], ncol=c["leg1_ncol"], framealpha=0.95)
    
    # Subplot 2: Controle u
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
    ax2.plot(t_s, df_s["u_pct"], color="#1f77b4", linestyle="-", linewidth=1.1, alpha=0.8, label="Sem Integrador")
    ax2.plot(t_c, df_c["u_pct"], color="#d62728", linestyle="-", linewidth=1.2, alpha=0.85, label="Com Integrador")
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel(r"Controle $u$ [\%]")
    ax2.set_ylim(-15, 95)
    ax2.grid(True)
    ax2.legend(loc=c["leg2_loc"], ncol=c["leg2_ncol"], framealpha=0.95)
    
    max_t = min(t_s.max(), t_c.max())
    ax1.set_xlim(0, max_t)
    plt.tight_layout()
    
    fig.savefig(os.path.join(out_dir, f"{c['name']}.png"), dpi=300)
    plt.close(fig)
