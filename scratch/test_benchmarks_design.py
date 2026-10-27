import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# Estilo acadêmico PUCPR / IEEE
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Ordem de cima para baixo: -O0 até Alta Comp.
labels = [
    'C Nativo (-O0)',
    'C Nativo (-O1)',
    'C Nativo (-Os)',
    'C Nativo (-O3)',
    'X-CUBE-AI (Sem Comp.)',
    'X-CUBE-AI (Alta Comp.)'
]

tempos = [4.96, 2.46, 2.08, 1.98, 1.69, 1.82]
flashes = [130.50, 130.50, 130.50, 130.50, 130.50, 17.75]
speedups = [1.00, 2.02, 2.38, 2.51, 2.93, 2.73]
ocupacoes = [49.6, 24.6, 20.8, 19.8, 16.9, 18.2]

# Inverter para que a primeira categoria fique no TOPO no barh
y_labels = labels[::-1]
y_tempos = tempos[::-1]
y_flashes = flashes[::-1]
y_speedups = speedups[::-1]
y_ocupacoes = ocupacoes[::-1]

y_pos = np.arange(len(labels))

out_dir = r"c:\Users\vicio\Documents\AEROPENDULO\scratch"
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# OPÇÃO A-PREMIUM: HORIZONTAL COM DUPLO EIXO DE TOPO (% CPU e % Flash Total)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True)

# Paleta Coesa
# GCC: Slate Navy | X-CUBE: Vivid Azure | Destaque Best: Electric Blue
c_gcc = '#3b82f6'
c_best_time = '#0284c7'
c_flash_norm = '#94a3b8'
c_flash_comp = '#16a34a'

time_bar_colors = [
    '#0ea5e9',      # X-CUBE Alta Comp (1.82 ms)
    c_best_time,    # X-CUBE Sem Comp (1.69 ms) - MAIS RÁPIDO
    '#2563eb',      # C Nativo -O3 (1.98 ms)
    '#3b82f6',      # C Nativo -Os (2.08 ms)
    '#60a5fa',      # C Nativo -O1 (2.46 ms)
    '#64748b',      # C Nativo -O0 (4.96 ms) - BASE
]

flash_bar_colors = [
    c_flash_comp,   # Alta Comp - 17.75 KiB (-86.4%)
    c_flash_norm,
    c_flash_norm,
    c_flash_norm,
    c_flash_norm,
    c_flash_norm,
]

# --- PAINEL 1: TEMPO DE INFERÊNCIA ---
bars1 = ax1.barh(y_pos, y_tempos, height=0.55, color=time_bar_colors, edgecolor='#1e293b', linewidth=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(y_labels, fontweight='normal', color='#0f172a')
ax1.set_xlabel('Tempo de Inferência ($t_{\\mathrm{ann}}$) [ms]', fontweight='bold')
ax1.set_title('(a) Latência por Passo e Aceleração', fontweight='bold', pad=12)
ax1.set_xlim(0, 6.3)
ax1.grid(True, axis='x', alpha=0.45, linestyle='--')

# Eixo superior com % de Ocupação da Janela de 10 ms
ax1_top = ax1.twiny()
ax1_top.set_xlim(0, 6.3 / 10.0 * 100.0) # 0 a 63%
ax1_top.set_xlabel('Ocupação da Janela ($T_s = 10$ ms) [%]', fontsize=8.5, color='#475569')
ax1_top.tick_params(axis='x', labelsize=8, colors='#475569')

# Linha divisória entre GCC e X-CUBE-AI (está entre índice 3 e 4 no array invertido, ou seja y=1.5 no invertido)
# Array invertido: 0=AltaComp, 1=SemComp (X-CUBE-AI) | 2=-O3, 3=-Os, 4=-O1, 5=-O0 (C Nativo)
# Divisão é em y = 1.5!
ax1.axhline(1.5, color='#cbd5e1', linestyle='-', linewidth=1.2)
ax2.axhline(1.5, color='#cbd5e1', linestyle='-', linewidth=1.2)

# Adicionar anotações em cada barra de tempo
for i, (b, t, sp, oc) in enumerate(zip(bars1, y_tempos, y_speedups, y_ocupacoes)):
    w = b.get_width()
    is_best = (t == 1.69)
    if is_best:
        txt = f" {t:.2f} ms ({sp:.2f}×) ★ Melhor"
        ax1.text(w + 0.08, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.5, fontweight='bold', color='#0369a1')
    elif sp == 1.0:
        txt = f" {t:.2f} ms (1.00× Ref.)"
        ax1.text(w + 0.08, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.5, color='#475569')
    else:
        txt = f" {t:.2f} ms ({sp:.2f}×)"
        ax1.text(w + 0.08, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.5, color='#1e293b')

# Rótulos de grupo na lateral esquerda
ax1.text(-0.02, 4.5, 'C Nativo\n(GCC)', transform=ax1.get_yaxis_transform(),
         ha='right', va='center', fontsize=8.5, fontweight='bold', color='#475569',
         bbox=dict(boxstyle='square,pad=0.2', facecolor='#f1f5f9', edgecolor='none', alpha=0.8))
ax1.text(-0.02, 0.5, 'ST Edge AI\n(CMSIS-NN)', transform=ax1.get_yaxis_transform(),
         ha='right', va='center', fontsize=8.5, fontweight='bold', color='#0369a1',
         bbox=dict(boxstyle='square,pad=0.2', facecolor='#e0f2fe', edgecolor='none', alpha=0.8))

# --- PAINEL 2: MEMÓRIA FLASH ---
bars2 = ax2.barh(y_pos, y_flashes, height=0.55, color=flash_bar_colors, edgecolor='#1e293b', linewidth=0.7)
ax2.set_xlabel('Flash Ocupada pelos Pesos [KiB]', fontweight='bold')
ax2.set_title('(b) Pegada em Memória Flash', fontweight='bold', pad=12)
ax2.set_xlim(0, 175)
ax2.grid(True, axis='x', alpha=0.45, linestyle='--')

# Eixo superior com % da Flash Total do STM32F446RE (512 KiB)
ax2_top = ax2.twiny()
ax2_top.set_xlim(0, 175.0 / 512.0 * 100.0) # 0 a ~34%
ax2_top.set_xlabel('Fração da Flash Total (512 KiB) [%]', fontsize=8.5, color='#475569')
ax2_top.tick_params(axis='x', labelsize=8, colors='#475569')

# Rótulos nas barras de Flash
for b, f in zip(bars2, y_flashes):
    w = b.get_width()
    is_comp = (f < 50)
    if is_comp:
        txt = f" {f:.2f} KiB  (-86,4%) ★"
        ax2.text(w + 3.0, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.5, fontweight='bold', color='#15803d')
    else:
        txt = f" {f:.1f} KiB"
        ax2.text(w + 3.0, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.5, color='#475569')

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "option_A_premium.png"), dpi=300)
plt.close(fig)

# ==============================================================================
# OPÇÃO C: TRADE-OFF PARETO (LATÊNCIA vs FLASH) - ESTILO ARTIGO CIENTÍFICO TOP
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.4))

# Pontos de dados
# x: Flash (KiB), y: Tempo (ms)
# ou x: Tempo (ms), y: Flash (KiB)
x_time = tempos
y_flash = flashes

markers = ['o', 's', '^', 'D', 'p', '*']
point_colors = ['#64748b', '#3b82f6', '#2563eb', '#1d4ed8', '#0284c7', '#16a34a']

for i in range(len(labels)):
    sz = 140 if i == 5 or i == 4 else 85
    ax.scatter(x_time[i], y_flash[i], s=sz, color=point_colors[i], edgecolors='black',
               linewidth=0.8, zorder=5, label=labels[i])

# Fronteira de Pareto conectando os pontos ótimos:
# -O0 (4.96, 130.5) -> -O1 (2.46, 130.5) -> -Os (2.08, 130.5) -> -O3 (1.98, 130.5) -> X-CUBE None (1.69, 130.5)
# e de (1.69, 130.5) para X-CUBE High (1.82, 17.75)
ax.plot([1.69, 1.82], [130.5, 17.75], color='#16a34a', linestyle='--', linewidth=1.5,
        alpha=0.8, zorder=3, label='Fronteira de Pareto (Trade-off)')

# Anotações em cada ponto
ax.annotate('C Nativo (-O0)\n4.96 ms | 130.5 KiB', (4.96, 130.5), xytext=(4.3, 110),
            arrowprops=dict(arrowstyle='->', color='#64748b', lw=0.8), fontsize=8, ha='center')
ax.annotate('C Nativo (-O1)\n2.46 ms', (2.46, 130.5), xytext=(2.6, 142),
            arrowprops=dict(arrowstyle='->', color='#3b82f6', lw=0.8), fontsize=8, ha='center')
ax.annotate('C Nativo (-Os)\n2.08 ms', (2.08, 130.5), xytext=(2.15, 115),
            arrowprops=dict(arrowstyle='->', color='#2563eb', lw=0.8), fontsize=8, ha='center')
ax.annotate('C Nativo (-O3)\n1.98 ms', (1.98, 130.5), xytext=(1.8, 142),
            arrowprops=dict(arrowstyle='->', color='#1d4ed8', lw=0.8), fontsize=8, ha='center')
ax.annotate('X-CUBE-AI (Sem Comp.)\n1.69 ms (Mais Rápido)', (1.69, 130.5), xytext=(1.25, 110),
            arrowprops=dict(arrowstyle='->', color='#0284c7', lw=1.0), fontsize=8.5, ha='center',
            fontweight='bold', color='#0369a1')
ax.annotate('X-CUBE-AI (Alta Comp.)\n1.82 ms | 17.8 KiB (-86.4% Flash)', (1.82, 17.75), xytext=(2.6, 25),
            arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1.2), fontsize=8.5, ha='left',
            fontweight='bold', color='#15803d',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dcfce7', edgecolor='#86efac', alpha=0.9))

# Linha de deadline
ax.axvline(10.0, color='#dc2626', linestyle=':', label='Deadline Ts = 10 ms')

ax.set_xlabel('Tempo de Inferência por Passo ($t_{\\mathrm{ann}}$) [ms]', fontweight='bold')
ax.set_ylabel('Memória Flash Ocupada pelos Pesos [KiB]', fontweight='bold')
ax.set_title('Espaço de Projeto: Latência vs. Pegada de Memória Flash (STM32F446RE)', fontweight='bold')
ax.set_xlim(1.0, 5.5)
ax.set_ylim(0, 160)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "option_C_pareto.png"), dpi=300)
plt.close(fig)

print("Opções A-Premium e C geradas com sucesso!")
