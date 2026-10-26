import os
import pandas as pd
import numpy as np
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

p_sil = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_reteste_narx-mpc\3_sil_stm32\20261023_214321_referencia_legado_val1_mpc_hybrid.csv'
p_real = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\data\experimentos_novos\EXP001_reteste_narx-mpc\4_aeropendulo\20261023_180508_referencia_legado_val1_mpc_hybrid.csv'

df_sil = pd.read_csv(p_sil)
df_real = pd.read_csv(p_real)

t_sil = df_sil['tempo_ms'] / 1000.0
t_real = df_real['tempo_ms'] / 1000.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2]})

ax1.plot(t_sil, df_sil['referencia'], color='black', linestyle='--', linewidth=1.1, alpha=0.85, label='Referência')
ax1.plot(t_sil, df_sil['angulo_deg'], color='#1f77b4', linestyle='-', linewidth=1.3, alpha=0.9, label='STM32 em C (SIL)')
ax1.plot(t_real, df_real['angulo_deg'], color='#d62728', linestyle='-', linewidth=1.2, alpha=0.85, label='Bancada Real')

ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
ax1.grid(True)
ax1.legend(loc='lower right', ncol=3, framealpha=0.95)

ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
ax2.axhline(80, color='gray', linestyle='--', linewidth=0.7)
ax2.plot(t_sil, df_sil['u_pct'], color='#1f77b4', linestyle='-', linewidth=1.1, alpha=0.85, label='SIL')
ax2.plot(t_real, df_real['u_pct'], color='#d62728', linestyle='-', linewidth=1.1, alpha=0.85, label='Bancada Real')

ax2.set_xlabel('Tempo [s]')
ax2.set_ylabel(r'Controle $u$ [\%]')
ax2.set_ylim(-15, 95)
ax2.grid(True)
ax2.legend(loc='upper right', ncol=2, framealpha=0.95)

max_t = min(t_sil.max(), t_real.max())
ax1.set_xlim(0, max_t)

plt.tight_layout()
fig.savefig(r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\scratch\test_clean_comparacao_plot_0.png', dpi=300)
plt.close(fig)
print("Generated test_clean_comparacao_plot_0.png!")
