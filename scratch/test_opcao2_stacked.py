import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Lista completa com os novos dados
labels = [
    'C Nativo (-O0)',
    'C Nativo (-O1)',
    'C Nativo (-O2)',
    'C Nativo (-O3)',
    'C Nativo (-Ofast)',
    'X-CUBE-AI (Sem Comp.)',
    'X-CUBE-AI (Alta Comp.)'
]

tempos = [4.96, 2.46, 2.12, 1.98, 1.96, 1.69, 1.82]
speedups = [1.00, 2.02, 2.34, 2.51, 2.53, 2.93, 2.73]
ocupacoes = [49.6, 24.6, 21.2, 19.8, 19.6, 16.9, 18.2]

# Dados de Flash
# Pesos:
pesos = [130.50, 130.50, 130.50, 130.50, 130.50, 130.50, 17.75]
# Flash Total do Firmware:
# GCC medidos: 183.05, 173.70, 174.14, 175.63, 175.57 KB
# X-CUBE-AI: base firmware (~43.20 KB) + total modelo (139.05 KiB = 182.25 KB | 26.31 KiB = 69.51 KB)
flash_totais = [183.05, 173.70, 174.14, 175.63, 175.57, 182.25, 69.51]
codigo_app = [tot - p for tot, p in zip(flash_totais, pesos)]

# Inverter para que -O0 fique no topo
y_labels = labels[::-1]
y_tempos = tempos[::-1]
y_speedups = speedups[::-1]
y_ocupacoes = ocupacoes[::-1]
y_pesos = pesos[::-1]
y_totais = flash_totais[::-1]
y_codigo = codigo_app[::-1]

# Posições com separação entre GCC e X-CUBE
# 0: X-CUBE-AI Alta Comp
# 1: X-CUBE-AI Sem Comp
# (gap)
# 2.6: -Ofast
# 3.6: -O3
# 4.6: -O2
# 5.6: -O1
# 6.6: -O0
y_pos = np.array([0.5, 1.5, 2.9, 3.9, 4.9, 5.9, 6.9])

time_colors = [
    '#0ea5e9',  # X-CUBE Alta Comp
    '#0284c7',  # X-CUBE Sem Comp (Mais Rápido)
    '#1e40af',  # -Ofast
    '#1d4ed8',  # -O3
    '#2563eb',  # -O2
    '#3b82f6',  # -O1
    '#64748b',  # -O0
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.5), sharey=True)

# Fundos para agrupar blocos
ax1.axhspan(2.2, 7.6, facecolor='#f8fafc', alpha=0.9, zorder=0)
ax1.axhspan(-0.2, 2.2, facecolor='#f0fdf4', alpha=0.6, zorder=0)

ax2.axhspan(2.2, 7.6, facecolor='#f8fafc', alpha=0.9, zorder=0)
ax2.axhspan(-0.2, 2.2, facecolor='#f0fdf4', alpha=0.6, zorder=0)

ax1.axhline(2.2, color='#cbd5e1', linestyle='-', linewidth=1.0, zorder=1)
ax2.axhline(2.2, color='#cbd5e1', linestyle='-', linewidth=1.0, zorder=1)

# --- PAINEL 1: TEMPO DE INFERÊNCIA ---
bars1 = ax1.barh(y_pos, y_tempos, height=0.62, color=time_colors,
                 edgecolor='#0f172a', linewidth=0.65, zorder=3)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(y_labels, color='#0f172a')
ax1.set_xlabel('Tempo de Inferência ($t_{\\mathrm{ann}}$) [ms]', fontweight='bold')
ax1.set_title('(a) Tempo de Inferência por Passo ($t_{\\mathrm{ann}}$)', fontweight='bold', pad=10)
ax1.set_xlim(0, 6.8)
ax1.set_ylim(-0.3, 7.6)
ax1.grid(True, axis='x', zorder=2)

for b, t, sp, oc in zip(bars1, y_tempos, y_speedups, y_ocupacoes):
    w = b.get_width()
    is_best = (t == 1.69)
    y_mid = b.get_y() + b.get_height() / 2
    if is_best:
        txt = f" {t:.2f} ms ({sp:.2f}× | {oc:.1f}%) [Mais Rápido]"
        ax1.text(w + 0.1, y_mid, txt, va='center', ha='left',
                 fontsize=8.5, fontweight='bold', color='#0284c7', zorder=4)
    elif sp == 1.0:
        txt = f" {t:.2f} ms (1.00× Ref. | {oc:.1f}%)"
        ax1.text(w + 0.1, y_mid, txt, va='center', ha='left',
                 fontsize=8.5, color='#64748b', zorder=4)
    else:
        txt = f" {t:.2f} ms ({sp:.2f}× | {oc:.1f}%)"
        ax1.text(w + 0.1, y_mid, txt, va='center', ha='left',
                 fontsize=8.5, color='#1e293b', zorder=4)

# --- PAINEL 2: BARRAS EMPILHADAS (FLASH PESOS + CÓDIGO/APP) ---
# Barra base: Pesos
bar_weights_colors = [
    '#16a34a',  # Alta Comp - pesos 17.75 KiB (verde)
    '#94a3b8',  # Demais pesos 130.5 KiB (cinza ardósia)
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
]

# Barra sobreposta: Código / Overhead
bar_code_colors = [
    '#86efac',  # Runtime X-CUBE quantizado (verde claro)
    '#cbd5e1',  # Runtime X-CUBE float32 (cinza claro)
    '#cbd5e1',
    '#cbd5e1',
    '#cbd5e1',
    '#cbd5e1',
    '#cbd5e1',
]

b_pesos = ax2.barh(y_pos, y_pesos, height=0.62, color=bar_weights_colors,
                   edgecolor='#0f172a', linewidth=0.65, zorder=3, label='Pesos da Rede (Flash)')
b_code = ax2.barh(y_pos, y_codigo, left=y_pesos, height=0.62, color=bar_code_colors,
                  edgecolor='#475569', linewidth=0.65, hatch='///', zorder=3, label='Código da Aplicação / Runtime')

ax2.set_xlabel('Ocupação de Memória Flash [KB]', fontweight='bold')
ax2.set_title('(b) Pegada Total de Flash no Firmware', fontweight='bold', pad=10)
ax2.set_xlim(0, 240)
ax2.set_ylim(-0.3, 7.6)
ax2.grid(True, axis='x', zorder=2)
ax2.legend(loc='lower right', fontsize=8, framealpha=0.92)

# Rótulos de Flash Total
for y_i, tot, p_i in zip(y_pos, y_totais, y_pesos):
    is_comp = (p_i < 50)
    if is_comp:
        txt = f" {tot:.1f} KB [Pesos: {p_i:.1f} KB] (-61,8% Total)"
        ax2.text(tot + 2.0, y_i, txt, va='center', ha='left',
                 fontsize=8.5, fontweight='bold', color='#15803d', zorder=4)
    else:
        txt = f" {tot:.1f} KB [Pesos: {p_i:.1f} KB]"
        ax2.text(tot + 2.0, y_i, txt, va='center', ha='left',
                 fontsize=8.5, color='#334155', zorder=4)

plt.tight_layout()
out_test = r"c:\Users\vicio\Documents\AEROPENDULO\scratch\benchmark_opcao2_stacked.png"
fig.savefig(out_test, dpi=300)
plt.close(fig)
print("Gerado:", out_test)
