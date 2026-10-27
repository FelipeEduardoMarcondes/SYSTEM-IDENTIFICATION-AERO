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
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

labels = [
    'C Nativo\n(-O0)',
    'C Nativo\n(-O1)',
    'C Nativo\n(-Os)',
    'C Nativo\n(-O3)',
    'X-CUBE-AI\n(Nenhum)',
    'X-CUBE-AI\n(Alta Comp.)'
]

tempos = [4.96, 2.46, 2.08, 1.98, 1.69, 1.82]
flashes = [130.50, 130.50, 130.50, 130.50, 130.50, 17.75]

out_dir = r"c:\Users\vicio\Documents\AEROPENDULO\scratch"
x = np.arange(len(labels))

# ==============================================================================
# OPÇÃO D: BARRA (FLASH) + LINHA COM PONTOS (TEMPO) EM EIXO DUPLO - SEM COLISÃO
# ==============================================================================
fig, ax_flash = plt.subplots(figsize=(7.5, 4.0))

# Barras de Flash no eixo 1 (fundo)
bar_colors = ['#cbd5e1', '#cbd5e1', '#cbd5e1', '#cbd5e1', '#cbd5e1', '#86efac']
edge_colors = ['#64748b', '#64748b', '#64748b', '#64748b', '#64748b', '#16a34a']

rects = ax_flash.bar(x, flashes, width=0.52, color=bar_colors, edgecolor=edge_colors,
                     linewidth=1.0, zorder=2, label='Memória Flash dos Pesos (KiB)')
ax_flash.set_ylabel('Flash Ocupada pelos Pesos [KiB]', color='#334155', fontweight='bold')
ax_flash.tick_params(axis='y', labelcolor='#334155')
ax_flash.set_ylim(0, 160)
ax_flash.grid(True, axis='y', alpha=0.35, zorder=1)

# Rótulos nas barras de flash
for r, f in zip(rects, flashes):
    h = r.get_height()
    if f < 50:
        ax_flash.text(r.get_x() + r.get_width()/2, h + 3.0, f'{f:.1f} KiB\n(-86,4%)',
                      ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#15803d')
    else:
        ax_flash.text(r.get_x() + r.get_width()/2, h + 3.0, f'{f:.1f} KiB',
                      ha='center', va='bottom', fontsize=8, color='#475569')

# Linha com marcadores no eixo 2 (tempo)
ax_time = ax_flash.twinx()
color_time = '#1d4ed8'
line = ax_time.plot(x, tempos, color=color_time, marker='o', markersize=7,
                    linewidth=2.0, zorder=4, label='Tempo de Inferência (ms)')
ax_time.set_ylabel('Tempo de Inferência ($t_{\\mathrm{ann}}$) [ms]', color=color_time, fontweight='bold')
ax_time.tick_params(axis='y', labelcolor=color_time)
ax_time.set_ylim(0, 6.0)

# Rótulos dos pontos de tempo
for xi, yi in zip(x, tempos):
    if yi == 1.69:
        ax_time.annotate(f'{yi:.2f} ms\n(Mais Rápido)', (xi, yi), xytext=(0, -26),
                         textcoords='offset points', ha='center', fontsize=8.5,
                         fontweight='bold', color='#1d4ed8')
    elif xi == 0:
        ax_time.annotate(f'{yi:.2f} ms\n(Base)', (xi, yi), xytext=(0, 10),
                         textcoords='offset points', ha='center', fontsize=8,
                         fontweight='bold', color='#1d4ed8')
    else:
        ax_time.annotate(f'{yi:.2f} ms', (xi, yi), xytext=(0, 10),
                         textcoords='offset points', ha='center', fontsize=8,
                         fontweight='bold', color='#1d4ed8')

ax_flash.set_xticks(x)
ax_flash.set_xticklabels(labels, fontsize=8.5)
ax_flash.set_title('Comparativo de Eficiência Computacional e Uso de Flash (STM32F446RE)',
                   fontweight='bold', pad=12)

# Divisória entre GCC e X-CUBE
ax_flash.axvline(3.5, color='#94a3b8', linestyle=':', linewidth=1.2, zorder=3)

# Legenda combinada
lines1, labels1 = ax_flash.get_legend_handles_labels()
lines2, labels2 = ax_time.get_legend_handles_labels()
ax_flash.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.95)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "option_D_bar_line.png"), dpi=300)
plt.close(fig)
print("Opção D gerada com sucesso!")
