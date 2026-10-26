"""
Script de Geração de Plots Acadêmicos no Padrão IEEE / PUCPR
Comparação Direta na Bancada Real: Sem Integrador (Nominal) vs Com Integrador (Híbrido)
Projeto: Controle de Aeropêndulo com NMPC Aproximado por Rede Neural (TinyML)
Autor: Felipe Eduardo Marcondes

Gera gráficos padronizados focados estritamente na avaliação do integrador residual:
- Linha preta tracejada: Trajetória de Referência r(t)
- Linha azul contínua: Bancada Real Sem Integrador (Nominal Puro - 23/10)
- Linha vermelha contínua: Bancada Real Com Integrador (Híbrido Residual - 22/10)
- Subplot inferior: Sinal de Controle u(t) de ambos os métodos
"""

import os
import glob
import numpy as np
import pandas as pd
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

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

exp_dir = os.path.join(root_dir, "data", "experimentos_novos", "EXP001_reteste_narx-mpc")
real_dir = os.path.join(exp_dir, "4_aeropendulo")

dest_dirs = [
    os.path.join(root_dir, "tcc", "img", "resultados"),
    os.path.join(root_dir, "tcc", "img"),
    os.path.join(exp_dir, "plots_academicos"),
    os.path.join(root_dir, "fluxograma-tcc", "10_comparativos_finais", "nominal_vs_hibrido")
]
for d in dest_dirs:
    os.makedirs(d, exist_ok=True)

comparacoes = [
    {
        "id": 1,
        "base_name": "comparacao_hibrido_val1",
        "title": "Validação 1 (Multisseno + Degraus / Chirp)",
        "f_sem": os.path.join(real_dir, "20261023_180508_referencia_legado_val1_mpc_hybrid.csv"),
        "f_com": os.path.join(real_dir, "20261023_181157_referencia_legado_val1_mpc_hybrid.csv"),
        "leg1_pos": "lower right",
        "leg1_ncol": 3,
        "leg2_pos": "upper right",
        "leg2_ncol": 2
    },
    {
        "id": 2,
        "base_name": "comparacao_hibrido_val2",
        "title": "Validação 2 (Degraus Aleatórios)",
        "f_sem": os.path.join(real_dir, "20261023_180711_referencia_legado_val2_degraus_hybrid.csv"),
        "f_com": os.path.join(real_dir, "20261023_181401_referencia_legado_val2_degraus_hybrid.csv"),
        "leg1_pos": "upper right",
        "leg1_ncol": 1,
        "leg2_pos": "upper right",
        "leg2_ncol": 2
    },
    {
        "id": 3,
        "base_name": "comparacao_hibrido_val3",
        "title": "Validação 3 (Senoidal Suave)",
        "f_sem": os.path.join(real_dir, "20261023_180920_referencia_legado_val3_senoidal_hybrid.csv"),
        "f_com": os.path.join(real_dir, "20261023_181719_referencia_legado_val3_senoidal_hybrid.csv"),
        "leg1_pos": "lower right",
        "leg1_ncol": 3,
        "leg2_pos": "upper right",
        "leg2_ncol": 2
    }
]

print("=== Gerando Gráficos Comparativos: Bancada Sem vs Com Integrador ===")

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
    ax1.legend(loc=c["leg1_pos"], ncol=c["leg1_ncol"], framealpha=0.95)
    
    # Subplot 2: Controle u
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
    ax2.plot(t_s, df_s["u_pct"], color="#1f77b4", linestyle="-", linewidth=1.1, alpha=0.8, label="Sem Integrador")
    ax2.plot(t_c, df_c["u_pct"], color="#d62728", linestyle="-", linewidth=1.2, alpha=0.85, label="Com Integrador")
    ax2.set_xlabel("Tempo [s]")
    ax2.set_ylabel(r"Controle $u$ [\%]")
    ax2.set_ylim(-15, 95)
    ax2.grid(True)
    ax2.legend(loc=c["leg2_pos"], ncol=c["leg2_ncol"], framealpha=0.95)
    
    max_t = min(t_s.max(), t_c.max())
    ax1.set_xlim(0, max_t)
    plt.tight_layout()
    
    for d in dest_dirs:
        png_p = os.path.join(d, f"{c['base_name']}.png")
        pdf_p = os.path.join(d, f"{c['base_name']}.pdf")
        fig.savefig(png_p, dpi=300)
        fig.savefig(pdf_p, format="pdf")
        
        # Também salva com o nome sem_vs_com
        fig.savefig(os.path.join(d, f"comparacao_bancada_sem_vs_com_{c['id']}.png"), dpi=300)
        fig.savefig(os.path.join(d, f"comparacao_bancada_sem_vs_com_{c['id']}.pdf"), format="pdf")
        fig.savefig(os.path.join(d, f"comparacao_bancada_sem_vs_com_val{c['id']}.png"), dpi=300)
        fig.savefig(os.path.join(d, f"comparacao_bancada_sem_vs_com_val{c['id']}.pdf"), format="pdf")
        
    print(f"-> Sucesso: {c['base_name']} gerado em PNG (300 DPI) e PDF vetorial.")
    plt.close(fig)

print("Todos os gráficos comparativos foram atualizados no padrão IEEE / TCC!")
