import os
import numpy as np
import matplotlib.pyplot as plt

# Estilo acadêmico refinado
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
    "grid.alpha": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Configurações na ordem cronológica (de cima para baixo)
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

# Inverter para que o primeiro (-O0) fique no topo no barh
y_labels = labels[::-1]
y_tempos = tempos[::-1]
y_flashes = flashes[::-1]
y_speedups = speedups[::-1]
y_ocupacoes = ocupacoes[::-1]

y_pos = np.arange(len(labels))

# Cores:
# GCC: azul sóbrio degradê de cinza-ardósia a azul
# X-CUBE Sem Comp: azul ciano de destaque
# X-CUBE Alta Comp: verde esmeralda para Flash / ciano para tempo
time_colors = [
    '#0284c7',  # X-CUBE Alta Comp (1.82 ms)
    '#0369a1',  # X-CUBE Sem Comp (1.69 ms) - MAIS RÁPIDO
    '#2563eb',  # C Nativo -O3 (1.98 ms)
    '#3b82f6',  # C Nativo -Os (2.08 ms)
    '#60a5fa',  # C Nativo -O1 (2.46 ms)
    '#64748b',  # C Nativo -O0 (4.96 ms)
]

flash_colors = [
    '#16a34a',  # Alta Comp - verde esmeralda
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)

# Faixas de fundo para agrupar visualmente C Nativo vs X-CUBE-AI
# y_pos: 0=AltaComp, 1=SemComp (X-CUBE) | 2=-O3, 3=-Os, 4=-O1, 5=-O0 (C Nativo)
ax1.axhspan(1.5, 5.5, facecolor='#f8fafc', alpha=0.9, zorder=0)
ax1.axhspan(-0.5, 1.5, facecolor='#f0fdf4', alpha=0.6, zorder=0)

ax2.axhspan(1.5, 5.5, facecolor='#f8fafc', alpha=0.9, zorder=0)
ax2.axhspan(-0.5, 1.5, facecolor='#f0fdf4', alpha=0.6, zorder=0)

# Linha divisória limpa
ax1.axhline(1.5, color='#cbd5e1', linestyle='-', linewidth=1.2, zorder=1)
ax2.axhline(1.5, color='#cbd5e1', linestyle='-', linewidth=1.2, zorder=1)

# --- SUBPLOT 1: TEMPO DE INFERÊNCIA ---
bars1 = ax1.barh(y_pos, y_tempos, height=0.56, color=time_colors,
                 edgecolor='#0f172a', linewidth=0.6, zorder=3)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(y_labels, color='#0f172a', fontweight='500')
ax1.set_xlabel('Tempo de Inferência ($t_{\\mathrm{ann}}$) [ms]', fontweight='bold')
ax1.set_title('(a) Latência por Passo e Aceleração', fontweight='bold', pad=10)
ax1.set_xlim(0, 6.6)
ax1.grid(True, axis='x', zorder=2)

# Adicionar textos explicativos nas barras de tempo
for b, t, sp, oc in zip(bars1, y_tempos, y_speedups, y_ocupacoes):
    w = b.get_width()
    is_best = (t == 1.69)
    if is_best:
        txt = f" {t:.2f} ms ({sp:.2f}× | {oc:.1f}%) [Mais Rápido]"
        ax1.text(w + 0.08, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.2, fontweight='bold', color='#0369a1', zorder=4)
    elif sp == 1.0:
        txt = f" {t:.2f} ms (Ref. 1.00× | {oc:.1f}%)"
        ax1.text(w + 0.08, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.2, color='#475569', zorder=4)
    else:
        txt = f" {t:.2f} ms ({sp:.2f}× | {oc:.1f}%)"
        ax1.text(w + 0.08, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.2, color='#1e293b', zorder=4)

# --- SUBPLOT 2: FLASH ---
bars2 = ax2.barh(y_pos, y_flashes, height=0.56, color=flash_colors,
                 edgecolor='#0f172a', linewidth=0.6, zorder=3)
ax2.set_xlabel('Flash Ocupada pelos Pesos [KiB]', fontweight='bold')
ax2.set_title('(b) Ocupação de Memória Flash', fontweight='bold', pad=10)
ax2.set_xlim(0, 188)
ax2.grid(True, axis='x', zorder=2)

# Adicionar textos explicativos nas barras de flash
for b, f in zip(bars2, y_flashes):
    w = b.get_width()
    is_comp = (f < 50)
    pct_flash = (f / 512.0) * 100.0  # % da Flash total de 512 KiB do STM32F446RE
    if is_comp:
        txt = f" {f:.2f} KiB (-86,4% Flash | {pct_flash:.1f}%) [Econômico]"
        ax2.text(w + 3.0, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.2, fontweight='bold', color='#15803d', zorder=4)
    else:
        txt = f" {f:.1f} KiB ({pct_flash:.1f}% total)"
        ax2.text(w + 3.0, b.get_y() + b.get_height()/2, txt,
                 va='center', ha='left', fontsize=8.2, color='#475569', zorder=4)

# Legenda sutil dos grupos
ax1.text(0.98, 0.96, 'C Nativo (GCC)', transform=ax1.transAxes,
         ha='right', va='top', fontsize=8.5, fontweight='bold', color='#64748b')
ax1.text(0.98, 0.32, 'ST X-CUBE-AI', transform=ax1.transAxes,
         ha='right', va='top', fontsize=8.5, fontweight='bold', color='#0284c7')

plt.tight_layout()
out_file = r"c:\Users\vicio\Documents\AEROPENDULO\scratch\benchmark_final_v1.png"
fig.savefig(out_file, dpi=300)
plt.close(fig)
print("Gerado:", out_file)
