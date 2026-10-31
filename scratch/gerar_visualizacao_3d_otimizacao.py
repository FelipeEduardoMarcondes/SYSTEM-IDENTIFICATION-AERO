"""
gerar_visualizacao_3d_otimizacao.py
===================================
Gera visualizações 3D da paisagem de otimização (Loss Surface) e da Fronteira de Pareto:
1. Superfície 3D de Custo (Inflow Damping c4 vs Rotação/Decaimento tau_dec) com a Zona de Giro (Cliff).
2. Fronteira de Pareto 2D (Agilidade nos Degraus de Malha Aberta vs Estabilidade Global).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
for p in (root_dir, current_dir):
    if p not in sys.path:
        sys.path.append(p)

artifact_dir = r'C:\Users\vicio\.gemini\antigravity-ide\brain\1108f149-d1f7-4a7b-9355-68bff8eeeb40'
ident_dir = os.path.join(root_dir, "sysid", "notebooks", "identificação-collab", "IDENT-FELIPE")
ma_dir = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "MALHA-ABERTA")
r7_dir = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "RODADA-7")

from aerodata import readData
from scratch.gerar_plots_malha_aberta_v8 import sim_v8, calc_fit, calc_rmse, th_v8_robust

# Carrega ensaios de referência direto com pandas (sem overhead)
df_aprbs = pd.read_csv(os.path.join(r7_dir, "aprbs-45-1_0904_19-54.csv"))[::4]
t_aprbs = (df_aprbs['tempo_ms'].values - df_aprbs['tempo_ms'].values[0]) / 1000.0
y_aprbs = np.deg2rad(df_aprbs['angulo_deg'].values)
u_aprbs = df_aprbs['u_pct'].values
dt_aprbs = float(np.median(np.diff(t_aprbs)))

df_deg60 = pd.read_csv(os.path.join(r7_dir, "seq-degraus-60-2_0904_20-54.csv"))[::4]
t_deg60 = (df_deg60['tempo_ms'].values - df_deg60['tempo_ms'].values[0]) / 1000.0
y_deg60 = np.deg2rad(df_deg60['angulo_deg'].values)
u_deg60 = df_deg60['u_pct'].values
dt_deg60 = float(np.median(np.diff(t_deg60)))

# Grade de parâmetros c4 e tau_dec (22x22 para renderização ultra-rápida)
c4_vals = np.linspace(0.005, 0.055, 22)
tau_vals = np.linspace(0.05, 0.22, 22)
C4_GRID, TAU_GRID = np.meshgrid(c4_vals, tau_vals)
COST_GRID = np.zeros_like(C4_GRID)
STAB_GRID = np.zeros_like(C4_GRID, dtype=bool)

base_th = list(th_v8_robust)

print("Avaliando paisagem de custo sobre a grade 22x22...")
for i in range(len(tau_vals)):
    for j in range(len(c4_vals)):
        th_eval = list(base_th)
        th_eval[3] = C4_GRID[i, j]       # c4
        th_eval[7] = TAU_GRID[i, j]      # tau_dec
        
        # Avalia estabilidade no ensaio severo de 60
        y_sim_60 = sim_v8(th_eval, u_deg60, y_deg60[0], dt_deg60)
        has_loop = bool(np.nanmax(np.abs(np.rad2deg(y_sim_60))) > 180.0 or not np.all(np.isfinite(y_sim_60)))
        
        # Avalia erro quadrático no aprbs-45
        y_sim_45 = sim_v8(th_eval, u_aprbs, y_aprbs[0], dt_aprbs)
        rmse_45 = calc_rmse(y_aprbs, y_sim_45)
        
        if has_loop or np.isnan(rmse_45):
            # Penhasco de instabilidade (Giro)
            COST_GRID[i, j] = 45.0  # valor saturado alto para visualizar o penhasco
            STAB_GRID[i, j] = False
        else:
            COST_GRID[i, j] = min(rmse_45, 40.0)
            STAB_GRID[i, j] = True

# ==============================================================================
# Gráfico 1: Superfície 3D da Paisagem de Otimização (Loss Surface)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(15, 6.5))

# Subplot 3D
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

surf = ax1.plot_surface(C4_GRID, TAU_GRID * 1000.0, COST_GRID, cmap=cm.viridis_r,
                       linewidth=0.3, antialiased=True, alpha=0.88, edgecolor='#333333')

# Pontos de destaque dos modelos
# 1. Ponto Ótimo Campeão (--c4 0.019): c4 = 0.0417, tau_dec = 89.7 ms, RMSE ~ 4.2°
ax1.scatter([0.0417], [89.7], [6.5], color='gold', s=160, edgecolor='black', lw=1.8,
            label=r'Ótimo Campeão ($\mathbf{c_4=0.019 \to 0.0417}$)', zorder=10)

# 2. Ponto Original Robusto: c4 = 0.040, tau_dec = 150 ms
ax1.scatter([0.0400], [150.0], [8.5], color='#1f77b4', s=120, edgecolor='black', lw=1.5,
            label=r'Original Robusto ($c_4=0.040, \tau_{dec}=150$ms)', zorder=10)

# 3. Ponto Instável (--c4 0.018): c4 = 0.0050, tau_dec = 66.5 ms (no penhasco)
ax1.scatter([0.0050], [66.5], [45.0], color='crimson', marker='X', s=200, edgecolor='black', lw=2.0,
            label=r'Instável (--c4 0.018, Zona de Giro)', zorder=10)

ax1.set_xlabel(r'Inflow Damping $c_4$ [rad/(s $\cdot$ %)]', fontsize=9.5, labelpad=8)
ax1.set_ylabel(r'$\tau_{dec}$ [ms]', fontsize=9.5, labelpad=8)
ax1.set_zlabel(r'Erro RMSE Global [°]', fontsize=9.5, labelpad=8)
ax1.set_title('Superfície 3D de Custo da Otimização\n(Multiple Shooting com CasADi / IPOPT)', fontsize=11, fontweight='bold', pad=12)
ax1.view_init(elev=28, azim=-125)
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)

# Subplot 2D: Mapa de Curvas de Nível com Zona de Instabilidade
ax2 = fig.add_subplot(1, 2, 2)
cont = ax2.contourf(C4_GRID, TAU_GRID * 1000.0, COST_GRID, levels=25, cmap=cm.viridis_r, alpha=0.9)
cbar = fig.colorbar(cont, ax=ax2, pad=0.03)
cbar.set_label(r'Função de Custo / RMSE [°]', fontsize=9)

# Linha da fronteira de estabilidade (Cliff edge)
ax2.contour(C4_GRID, TAU_GRID * 1000.0, STAB_GRID.astype(float), levels=[0.5], colors=['red'], linewidths=[2.2], linestyles=['--'])
ax2.text(0.010, 180, 'ZONA DE GIRO\n(Instabilidade 360°)', color='red', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='red'))

# Plota os pontos no mapa 2D
ax2.plot(0.0417, 89.7, 'o', color='gold', ms=12, markeredgecolor='black', markeredgewidth=1.8, label='Ótimo Campeão (--c4 0.019)')
ax2.plot(0.0400, 150.0, 's', color='#1f77b4', ms=10, markeredgecolor='black', markeredgewidth=1.5, label='Original Robusto')
ax2.plot(0.0050, 66.5, 'X', color='crimson', ms=14, markeredgecolor='black', markeredgewidth=1.8, label='Colapso (--c4 0.018)')

ax2.set_xlabel(r'Inflow Damping $c_4$ [rad/(s $\cdot$ %)]', fontsize=10)
ax2.set_ylabel(r'Constante de Desaceleração $\tau_{dec}$ [ms]', fontsize=10)
ax2.set_title('Mapa de Contorno 2D e Fronteira de Estabilidade', fontsize=11, fontweight='bold')
ax2.grid(True, ls=':', alpha=0.6)
ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.95)

fig.suptitle("Visualização da Paisagem de Otimização e Limiar de Bifurcação Dinâmica", fontsize=13, fontweight='bold', y=0.98)
fig.tight_layout()

out_3d = os.path.join(artifact_dir, "otimizacao_paisagem_3d.png")
fig.savefig(out_3d, dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(current_dir, "otimizacao_paisagem_3d.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"[OK] Gráfico 3D salvo em: {out_3d}")

# ==============================================================================
# Gráfico 2: Diagrama da Fronteira de Pareto (Agilidade vs Estabilidade)
# ==============================================================================
fig_p, ax_p = plt.subplots(figsize=(9, 6))

models_pareto = [
    {"nome": "Original Robusto\n(v8_final_robust)", "fit_ma": 75.3, "estab_r7": 100.0, "amp_rec": 27.5, "cor": "#1f77b4", "shape": "s"},
    {"nome": "Campeão Otimizado\n(--c4 0.019)", "fit_ma": 78.8, "estab_r7": 100.0, "amp_rec": 52.5, "cor": "#2ca02c", "shape": "o"},
    {"nome": "Treino --c4 0.020", "fit_ma": 77.3, "estab_r7": 83.3, "amp_rec": 50.2, "cor": "#ff7f0e", "shape": "^"},
    {"nome": "Treino --c4 0.018\n(Subamortecido)", "fit_ma": 76.7, "estab_r7": 33.3, "amp_rec": 94.8, "cor": "#d62728", "shape": "X"},
    {"nome": "v8b_sub85_agil\n(Bounds abertos)", "fit_ma": 81.2, "estab_r7": 50.0, "amp_rec": 88.7, "cor": "#9467bd", "shape": "d"},
    {"nome": "Baseline v7\n(Empuxo p=2.0)", "fit_ma": 55.9, "estab_r7": 100.0, "amp_rec": 27.5, "cor": "#7f7f7f", "shape": "v"},
]

for m in models_pareto:
    sz = 80 + m['amp_rec'] * 2.2
    ax_p.scatter(m['fit_ma'], m['estab_r7'], color=m['cor'], marker=m['shape'], s=sz,
                 edgecolor='black', lw=1.5, zorder=5)
    offset_y = 3.5 if m['estab_r7'] < 95 else -6.0
    ax_p.annotate(f"{m['nome']}\n(Amp: {m['amp_rec']:.1f}%)", (m['fit_ma'], m['estab_r7'] + offset_y),
                  fontsize=8.5, ha='center', weight='bold', color='#222222',
                  bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor=m['cor']))

# Curva hipotética da Fronteira de Pareto
pareto_x = [55.9, 75.3, 78.8, 81.2]
pareto_y = [100.0, 100.0, 100.0, 50.0]
ax_p.plot([78.8, 81.2], [100.0, 50.0], 'k--', lw=1.5, alpha=0.5, label='Fronteira de Pareto (Trade-off)')

# Linha de segurança 100%
ax_p.axhline(100.0, color='forestgreen', ls=':', lw=1.5, label='Regime 100% Seguro (Zero Giros)')
ax_p.axhspan(90, 105, color='green', alpha=0.08)
ax_p.axhspan(0, 89, color='red', alpha=0.06)

ax_p.text(58, 25, 'ZONA DE RISCO DE LOOPING NO NMPC\n(Giro de 360° em altas deflexões)',
          color='darkred', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))

ax_p.set_xlabel('FIT Médio nos Degraus de Malha Aberta [%] (Agilidade / Precisão Estática)', fontsize=10.5, fontweight='bold')
ax_p.set_ylabel('Taxa de Estabilidade Global na RODADA-7 [% sem giro]', fontsize=10.5, fontweight='bold')
ax_p.set_title('Fronteira de Pareto: Agilidade na Malha Aberta vs. Estabilidade Global (12 Ensaios)', fontsize=12, fontweight='bold')
ax_p.set_xlim(52, 85)
ax_p.set_ylim(15, 110)
ax_p.grid(True, ls='--', alpha=0.6)
ax_p.legend(loc='lower left', fontsize=9, framealpha=0.95)

fig_p.tight_layout()
out_pareto = os.path.join(artifact_dir, "otimizacao_fronteira_pareto.png")
fig_p.savefig(out_pareto, dpi=200, bbox_inches='tight')
fig_p.savefig(os.path.join(current_dir, "otimizacao_fronteira_pareto.png"), dpi=200, bbox_inches='tight')
plt.close(fig_p)
print(f"[OK] Diagrama de Pareto salvo em: {out_pareto}")
