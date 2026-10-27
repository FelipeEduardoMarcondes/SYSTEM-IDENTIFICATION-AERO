import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Labels e dados
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

# Inverter para que -O0 fique no topo
y_labels = labels[::-1]
y_tempos = tempos[::-1]
y_flashes = flashes[::-1]
y_speedups = speedups[::-1]
y_ocupacoes = ocupacoes[::-1]

# Posições Y com espaçamento adicional entre os dois grupos:
# Índices invertidos:
# 0: X-CUBE-AI (Alta Comp.) -> y = 0.5
# 1: X-CUBE-AI (Sem Comp.)  -> y = 1.5
# (gap de 1.0 unidade para separar os grupos)
# 2: C Nativo (-O3)         -> y = 3.0
# 3: C Nativo (-Os)         -> y = 4.0
# 4: C Nativo (-O1)         -> y = 5.0
# 5: C Nativo (-O0)         -> y = 6.0
y_pos = np.array([0.5, 1.5, 3.0, 4.0, 5.0, 6.0])

time_colors = [
    '#0ea5e9',  # Alta Comp
    '#0284c7',  # Sem Comp (mais rápido)
    '#2563eb',  # -O3
    '#3b82f6',  # -Os
    '#60a5fa',  # -O1
    '#64748b',  # -O0
]

flash_colors = [
    '#16a34a',  # Alta Comp (verde economia)
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.3), sharey=True)

# Faixas de fundo elegantes
ax1.axhspan(2.4, 6.6, facecolor='#f8fafc', alpha=0.9, zorder=0)
ax1.axhspan(-0.1, 2.1, facecolor='#f0fdf4', alpha=0.6, zorder=0)

ax2.axhspan(2.4, 6.6, facecolor='#f8fafc', alpha=0.9, zorder=0)
ax2.axhspan(-0.1, 2.1, facecolor='#f0fdf4', alpha=0.6, zorder=0)

# Divisória entre os grupos
ax1.axhline(2.25, color='#cbd5e1', linestyle='-', linewidth=1.0, zorder=1)
ax2.axhline(2.25, color='#cbd5e1', linestyle='-', linewidth=1.0, zorder=1)

# --- PAINEL 1: LATÊNCIA ---
bars1 = ax1.barh(y_pos, y_tempos, height=0.62, color=time_colors,
                 edgecolor='#0f172a', linewidth=0.7, zorder=3)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(y_labels, color='#0f172a', fontweight='normal')
ax1.set_xlabel('Tempo de Inferência ($t_{\\mathrm{ann}}$) [ms]', fontweight='bold')
ax1.set_title('(a) Latência por Passo e Aceleração', fontweight='bold', pad=10)
ax1.set_xlim(0, 7.2)
ax1.set_ylim(-0.2, 6.8)
ax1.grid(True, axis='x', zorder=2)

# Textos do Painel 1
for b, t, sp, oc in zip(bars1, y_tempos, y_speedups, y_ocupacoes):
    w = b.get_width()
    is_best = (t == 1.69)
    y_mid = b.get_y() + b.get_height() / 2
    if is_best:
        txt = f" {t:.2f} ms ({sp:.2f}× | {oc:.1f}%) [Mais Rápido]"
        ax1.text(w + 0.1, y_mid, txt, va='center', ha='left',
                 fontsize=8.5, fontweight='bold', color='#0369a1', zorder=4)
    elif sp == 1.0:
        txt = f" {t:.2f} ms (Ref. 1.00× | {oc:.1f}%)"
        ax1.text(w + 0.1, y_mid, txt, va='center', ha='left',
                 fontsize=8.5, color='#475569', zorder=4)
    else:
        txt = f" {t:.2f} ms ({sp:.2f}× | {oc:.1f}%)"
        ax1.text(w + 0.1, y_mid, txt, va='center', ha='left',
                 fontsize=8.5, color='#1e293b', zorder=4)

# Cabeçalhos de seção nos subplots (fora de conflito com barras)
ax1.text(6.9, 6.45, 'Compilador GCC (C Nativo)', ha='right', va='center',
         fontsize=8.5, fontweight='bold', color='#475569', style='italic')
ax1.text(6.9, 1.95, 'ST Edge AI (CMSIS-NN)', ha='right', va='center',
         fontsize=8.5, fontweight='bold', color='#0284c7', style='italic')

# --- PAINEL 2: FLASH ---
bars2 = ax2.barh(y_pos, y_flashes, height=0.62, color=flash_colors,
                 edgecolor='#0f172a', linewidth=0.7, zorder=3)
ax2.set_xlabel('Flash Ocupada pelos Pesos [KiB]', fontweight='bold')
ax2.set_title('(b) Ocupação de Memória Flash', fontweight='bold', pad=10)
ax2.set_xlim(0, 195)
ax2.set_ylim(-0.2, 6.8)
ax2.grid(True, axis='x', zorder=2)

# Textos do Painel 2
for b, f in zip(bars2, y_flashes):
    w = b.get_width()
    is_comp = (f < 50)
    pct_flash = (f / 512.0) * 100.0
    y_mid = b.get_y() + b.get_height() / 2
    if is_comp:
        txt = f" {f:.2f} KiB (-86,4% Flash | {pct_flash:.1f}%) [Econômico]"
        ax2.text(w + 3.0, y_mid, txt, va='center', ha='left',
                 fontsize=8.5, fontweight='bold', color='#15803d', zorder=4)
    else:
        txt = f" {f:.1f} KiB ({pct_flash:.1f}% da Flash)"
        ax2.text(w + 3.0, y_mid, txt, va='center', ha='left',
                 fontsize=8.5, color='#475569', zorder=4)

ax2.text(190, 6.45, '512 KiB Flash Total', ha='right', va='center',
         fontsize=8.5, color='#64748b', style='italic')
ax2.text(190, 1.95, 'Quantização TinyML', ha='right', va='center',
         fontsize=8.5, color='#15803d', style='italic')

plt.tight_layout()
out_file = r"c:\Users\vicio\Documents\AEROPENDULO\scratch\benchmark_final_v2.png"
fig.savefig(out_file, dpi=300)
plt.close(fig)
print("Gerado:", out_file)
