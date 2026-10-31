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
    "figure.autolayout": False,
})

labels = [
    'C Nativo (-O0)',
    'C Nativo (-O1)',
    'C Nativo (-Os)',
    'C Nativo (-O3)',
    'X-CUBE-AI (Sem Comp.)',
    'X-CUBE-AI (Alta Comp.)'
]

tempos = [10.07, 2.46, 2.08, 1.98, 1.69, 1.82]

pesos = [130.50, 130.50, 130.50, 130.50, 130.50, 17.75]
flash_totais = [183.05, 173.70, 174.14, 175.63, 182.25, 69.51]
codigo_app = [tot - p for tot, p in zip(flash_totais, pesos)]

y_labels = labels[::-1]
y_tempos = tempos[::-1]
y_pesos = pesos[::-1]
y_totais = flash_totais[::-1]
y_codigo = codigo_app[::-1]

y_pos = np.array([0.5, 1.5, 2.8, 3.8, 4.8, 5.8])

time_colors = [
    '#0ea5e9',
    '#0284c7',
    '#1d4ed8',
    '#2563eb',
    '#3b82f6',
    '#ef4444',
]

bar_weights_colors = [
    '#16a34a',
    '#475569',
    '#475569',
    '#475569',
    '#475569',
    '#475569',
]

bar_code_colors = [
    '#86efac',
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
    '#94a3b8',
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)

# Fundos sutis
ax1.axhspan(2.15, 6.4, facecolor='#f8fafc', alpha=0.9, zorder=0)
ax1.axhspan(-0.1, 2.15, facecolor='#f0fdf4', alpha=0.6, zorder=0)

ax2.axhspan(2.15, 6.4, facecolor='#f8fafc', alpha=0.9, zorder=0)
ax2.axhspan(-0.1, 2.15, facecolor='#f0fdf4', alpha=0.6, zorder=0)

ax1.axhline(2.15, color='#cbd5e1', linestyle='-', linewidth=1.0, zorder=1)
ax2.axhline(2.15, color='#cbd5e1', linestyle='-', linewidth=1.0, zorder=1)

# --- SUBPLOT 1: TEMPO DE INFERENCIA LIMPO ---
bars1 = ax1.barh(y_pos, y_tempos, height=0.60, color=time_colors,
                 edgecolor='#0f172a', linewidth=0.65, zorder=3)

# Linha limite Ts = 10 ms
ax1.axvline(10.0, color='#dc2626', linestyle=':', linewidth=1.2, zorder=2)
ax1.text(10.0, 6.18, 'Limite $T_s = 10$ ms', color='#b91c1c', fontsize=8.0,
         fontweight='bold', ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#fee2e2', edgecolor='#f87171', linewidth=0.6))

ax1.set_yticks(y_pos)
ax1.set_yticklabels(y_labels, color='#0f172a')
ax1.set_xlabel('Tempo de Inferência ($t_{\\mathrm{ann}}$) [ms]', fontweight='bold')
ax1.set_title('(a) Tempo de Inferência por Passo ($t_{\\mathrm{ann}}$)', fontweight='bold', pad=10)
ax1.set_xlim(0, 12.2)
ax1.set_ylim(-0.2, 6.5)
ax1.grid(True, axis='x', zorder=2)

for b, t in zip(bars1, y_tempos):
    w = b.get_width()
    y_mid = b.get_y() + b.get_height() / 2
    if t == 1.69:
        ax1.text(w + 0.18, y_mid, f"{t:.2f} ms", va='center', ha='left',
                 fontsize=8.8, fontweight='bold', color='#0284c7', zorder=4)
    elif t > 10.0:
        ax1.text(w + 0.18, y_mid, f"{t:.2f} ms", va='center', ha='left',
                 fontsize=8.8, fontweight='bold', color='#b91c1c', zorder=4)
    else:
        ax1.text(w + 0.18, y_mid, f"{t:.2f} ms", va='center', ha='left',
                 fontsize=8.8, color='#1e293b', zorder=4)

# --- SUBPLOT 2: FLASH TOTAL EMPILHADA ---
b_pesos = ax2.barh(y_pos, y_pesos, height=0.60, color=bar_weights_colors,
                   edgecolor='#0f172a', linewidth=0.65, zorder=3, label='Pesos da Rede (Flash)')
b_code = ax2.barh(y_pos, y_codigo, left=y_pesos, height=0.60, color=bar_code_colors,
                  edgecolor='#475569', linewidth=0.65, hatch='///', zorder=3, label='Código da Aplicação / Runtime')

ax2.set_xlabel('Flash Ocupada no Microcontrolador [KB]', fontweight='bold')
ax2.set_title('(b) Pegada Total de Flash no Firmware STM32', fontweight='bold', pad=10)
ax2.set_xlim(0, 255)
ax2.set_ylim(-0.2, 6.5)
ax2.grid(True, axis='x', zorder=2)

for y_i, tot, p_i in zip(y_pos, y_totais, y_pesos):
    is_comp = (p_i < 50)
    if is_comp:
        txt = f" {tot:.1f} KB  (-61,8% Total | Pesos -86,4%)"
        ax2.text(tot + 2.0, y_i, txt, va='center', ha='left',
                 fontsize=8.5, fontweight='bold', color='#15803d', zorder=4)
    else:
        txt = f" {tot:.1f} KB"
        ax2.text(tot + 2.0, y_i, txt, va='center', ha='left',
                 fontsize=8.5, color='#334155', zorder=4)

for y_i, p_i in zip(y_pos, y_pesos):
    if p_i > 50:
        ax2.text(p_i / 2, y_i, f"Pesos: {p_i:.1f} KiB", ha='center', va='center',
                 fontsize=8.0, color='white', fontweight='bold', zorder=4)

fig.legend(handles=[b_pesos, b_code],
           labels=['Pesos da Rede Neural (Flash)', 'Código da Aplicação e Runtime ST'],
           loc='lower center', bbox_to_anchor=(0.52, 0.00), ncol=2,
           fontsize=8.5, frameon=True, framealpha=0.95, edgecolor='#cbd5e1')

plt.tight_layout(rect=[0, 0.07, 1, 1])

out_img = r"c:\Users\vicio\Documents\AEROPENDULO\scratch\benchmark_opcao2_final.png"
fig.savefig(out_img, dpi=300)
plt.close(fig)
print("Gerado:", out_img)
