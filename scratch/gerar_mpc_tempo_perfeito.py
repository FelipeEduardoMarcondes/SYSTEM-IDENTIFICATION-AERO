import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img_path = r'data/experimentos_novos/EXP001_legado_narx-mpc/2_mpc_python/MPC_Computational_Time.png'
orig_time_img = mpimg.imread(img_path)
r, g, b = orig_time_img[:,:,0], orig_time_img[:,:,1], orig_time_img[:,:,2]
blue_mask = (b > 0.6) & (r < 0.3) & (g < 0.6)

x_coords = np.arange(192, 1039)
t_extracted = (x_coords - 192.0) / (1038.0 - 192.0) * 25.0
dt_ms = []

for x in x_coords:
    y_pix = np.where(blue_mask[:, x])[0]
    if len(y_pix) > 0:
        span = np.max(y_pix) - np.min(y_pix)
        if span > 8:
            y_val = np.min(y_pix)
        else:
            y_val = np.mean(y_pix)
        val = (298.0 - y_val) / 550.0 * 1000.0
        dt_ms.append(val)
    else:
        dt_ms.append(dt_ms[-1] if dt_ms else 100.0)

dt_ms = np.array(dt_ms)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
    'grid.alpha': 0.45,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.85',
    'figure.autolayout': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.plot(t_extracted, dt_ms, color='#1f77b4', linestyle='-', linewidth=1.2, 
        label='Tempo de Solução NMPC (IPOPT)')
ax.axhline(10.0, color='#d62728', linestyle='--', linewidth=1.4, 
           label='Período de Amostragem $T_s = 10$ ms (Deadline)')

# Seta para o pico inicial de cold-start (t = 0 s, 458 ms)
ax.annotate('Cold-start inicial\n(458 ms)', 
            xy=(0.0, 458), xytext=(1.8, 420),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5),
            fontsize=8.5)

# Seta para o pico de transição dinâmica (t = 9.17 s, 333 ms)
ax.annotate('Pico em transição\nde degrau (333 ms)', 
            xy=(9.17, 333), xytext=(11.5, 320),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5),
            fontsize=8.5)

ax.set_xlabel('Tempo [s]')
ax.set_ylabel('Tempo de Solução [ms]')
ax.set_xlim(0, 25)
ax.set_ylim(0, 500)
ax.grid(True)
ax.legend(loc='upper right', framealpha=0.95)

plt.tight_layout()

dest_dirs = [
    r'tcc/img/resultados',
    r'tcc/img',
    r'tcc/resultado',
    r'img/resultados',
    r'resultado',
    r'relatorio_latex/imagens',
    r'fluxograma-tcc/05_nmpc_casadi_e_dataset/inviabilidade_tempo_real',
    r'data/experimentos_novos/EXP001_legado_narx-mpc/plots_academicos',
    r'data/experimentos_novos/EXP001_legado_narx-mpc/2_mpc_python',
    r'data/experimentos_novos/EXP001_otimizacao_narx/plots_academicos',
    r'data/experimentos_novos/EXP001_otimizacao_narx/2_mpc_python',
    r'data/experimentos_novos/EXP001_reteste_narx-mpc/plots_academicos',
    r'data/experimentos_novos/EXP001_reteste_narx-mpc/2_mpc_python',
]

for d in dest_dirs:
    if os.path.exists(d):
        png_p = os.path.join(d, 'mpc_tempo.png')
        pdf_p = os.path.join(d, 'mpc_tempo.pdf')
        fig.savefig(png_p, dpi=300)
        fig.savefig(pdf_p, format='pdf')
        print('Atualizado:', png_p)

plt.close(fig)
print('Concluído com perfeição!')
