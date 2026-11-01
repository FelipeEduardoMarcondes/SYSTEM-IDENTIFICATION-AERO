import os
import sys
import json
import re
import numpy as np
import matplotlib.pyplot as plt

# Import aerodata
try:
    from aerodata import readData
except ImportError:
    sys.path.append(os.path.abspath('.'))
    from aerodata import readData

current_dir = 'sysid/notebooks/identificação-collab/IDENT-HELON'
with open(os.path.join(current_dir, 'narx_model.json'), 'r') as f:
    narx_data = json.load(f)

# Load multisine dataset (RODADA-7 45° 0.50Hz)
fname = 'RODADA-7/multi-seno-45-050Hz_0904_20-29.csv'
y_raw, u_raw, t_raw, ref_raw = readData(dataset_name=fname, return_ref=True)
sl = slice(1000, -1300, 1)
u_ms, y_ms, t_ms = u_raw[sl], y_raw[sl], t_raw[sl] - t_raw[sl][0]
ref_ms = ref_raw[sl] if len(ref_raw) > 0 else np.full_like(t_ms, np.nan)

# Simulate NARX
terms = narx_data['terms']
theta = narx_data['theta']
ml = 15
N = len(u_ms)
y_sim = np.zeros(N)
y_sim[:ml] = y_ms[:ml]

rules = []
for term, th in zip(terms, theta):
    th_v = float(th)
    if term == 'constant':
        rules.append(('const', th_v))
    else:
        m = re.findall(r'([yu])\(k-(\d+)\)', term)
        if len(m) == 1:
            rules.append(('lin', th_v, m[0][0], int(m[0][1])))
        elif len(m) == 2:
            rules.append(('quad', th_v, m[0][0], int(m[0][1]), m[1][0], int(m[1][1])))

for k in range(ml, N):
    val = 0.0
    for r in rules:
        if r[0] == 'const':
            val += r[1]
        elif r[0] == 'lin':
            v = y_sim[k-r[3]] if r[2] == 'y' else u_ms[k-r[3]]
            val += r[1] * v
        elif r[0] == 'quad':
            v1 = y_sim[k-r[3]] if r[2] == 'y' else u_ms[k-r[3]]
            v2 = y_sim[k-r[5]] if r[4] == 'y' else u_ms[k-r[5]]
            val += r[1] * (v1 * v2)
    y_sim[k] = np.clip(val, -10.0, 135.0)

err = y_ms[ml:] - y_sim[ml:]
rmse = np.sqrt(np.mean(err**2))
mae = np.mean(np.abs(err))
denom = np.linalg.norm(y_ms[ml:] - np.mean(y_ms[ml:]))
fit = 100.0 * (1.0 - np.linalg.norm(err)/denom)

# Setup academic style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.85",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.3), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2]})

# Explicit raw strings with full LaTeX math $y(k)$
ax1.plot(t_ms, y_ms, color='#111111', lw=1.1, label=r'Medido $y(k)$')
ax1.plot(t_ms, y_sim, color='#d62728', ls='--', lw=1.3, label=r'CasADi NARX Free-Run $\hat{y}(k)$')
if not np.all(np.isnan(ref_ms)):
    ax1.plot(t_ms, ref_ms, color='#7f7f7f', ls=':', lw=0.9, alpha=0.7, label=r'Referência $r(k)$')

ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
ax1.grid(True, linestyle='--', alpha=0.5)

# Compact legend
ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.92,
           borderpad=0.25, labelspacing=0.2, handlelength=1.2, handletextpad=0.35)
ax1.set_xlim(0, t_ms[-1])

# Headroom so peaks do not touch legend or metric box
y_peak = max(float(np.max(y_ms)), float(np.max(y_sim)))
y_valley = min(float(np.min(y_ms)), float(np.min(y_sim)))
ax1.set_ylim(y_valley - 5.0, y_peak + max(16.0, (y_peak - y_valley) * 0.20))

metric_text = f"RMSE = {rmse:.3f}°\nMAE = {mae:.3f}°\nFIT = {fit:.1f}%"
ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes, verticalalignment='top', fontsize=7.5,
         bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.85', alpha=0.92))

ax2.plot(t_ms, u_ms, color='#1f77b4', lw=1.0, label=r'Excitação Multisseno $u(k)$')
ax2.set_xlabel('Tempo [s]')
ax2.set_ylabel(r'Controle $u$ [%]')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.92,
           borderpad=0.25, labelspacing=0.2, handlelength=1.2)

u_peak = float(np.max(u_ms))
u_valley = float(np.min(u_ms))
ax2.set_ylim(u_valley - 4.0, u_peak + max(12.0, (u_peak - u_valley) * 0.22))

plt.tight_layout()

targets = [
    'tcc/img/resultados',
    'tcc/img',
    'img/resultados',
    'sysid/notebooks/identificação-collab/IDENT-HELON/plots'
]

for d in targets:
    if os.path.exists(d):
        png_p = os.path.join(d, 'casadi_narx_freerun.png')
        pdf_p = os.path.join(d, 'casadi_narx_freerun.pdf')
        fig.savefig(png_p, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_p, format='pdf', bbox_inches='tight')
        print(f'Successfully saved clean casadi_narx_freerun to {d}')

plt.close(fig)
print('DONE!')

