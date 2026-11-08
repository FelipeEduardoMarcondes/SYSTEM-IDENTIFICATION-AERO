import os
import sys

root_dir = os.path.abspath('.')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import matplotlib.pyplot as plt
from aerodata import readData
from sysid import NARX
import re
out_dir = os.path.join(root_dir, 'artigo-ifac', 'img')
os.makedirs(out_dir, exist_ok=True)

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

from scratch.generate_english_figures import load_processed, train_files_en, simulate_freerun, compile_fast_evaluator

# Train the NARX model exactly as in standard pipeline
train_data = []
for meta in train_files_en.values():
    u, y, t, ref = load_processed(meta['file'])
    train_data.append((u, y))

narx_model = NARX(nu=15, ny=15, poly_order_l=2, n_components=10)
narx_model.fit(train_data)
ml = narx_model._max_lag_internal_

# Define 4 free run test signals
# Option 1: 4 distinct excitation classes at 45°
four_tests = [
    {
        'title': r'(a) Multisine ($45^\circ$, $0.30$ Hz)',
        'file': 'RODADA-7/multi-seno-45-030Hz_0904_20-23.csv'
    },
    {
        'title': r'(b) Chirp ($45^\circ$, Amp. $\pm 35^\circ$)',
        'file': 'RODADA-7/chirp-45-amp35_0904_20-13.csv'
    },
    {
        'title': r'(c) Multistep ($45^\circ$, $25^\circ$--$65^\circ$)',
        'file': 'RODADA-7/seq-degraus-45-2_0904_20-45.csv'
    },
    {
        'title': r'(d) APRBS ($45^\circ$)',
        'file': 'RODADA-7/aprbs-45-2_0904_19-57.csv'
    }
]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.0), sharex=False)
axes_flat = axes.flatten()

for i, tst in enumerate(four_tests):
    ax = axes_flat[i]
    u, y, t, ref = load_processed(tst['file'])
    y_fr = simulate_freerun(narx_model, u, y)
    
    t_plot = t[ml:]
    y_true = y[ml:]
    ref_plot = ref[ml:]
    
    err = y_true - y_fr
    rmse = np.sqrt(np.mean(err**2))
    fit = 100.0 * (1.0 - np.linalg.norm(err) / np.linalg.norm(y_true - np.mean(y_true)))
    
    ax.plot(t_plot, y_true, color='#111111', lw=1.1, label=r'Measured $\theta(t)$')
    ax.plot(t_plot, y_fr, color='#d62728', ls='--', lw=1.2, label=r'NARX Free-Run $\hat{\theta}(t)$')
    if not np.all(np.isnan(ref_plot)):
        ax.plot(t_plot, ref_plot, color='#7f7f7f', ls=':', lw=0.8, alpha=0.6, label=r'Reference $r(t)$')
        
    ax.set_title(tst['title'], fontsize=10, fontweight='bold', pad=4)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Angle $\theta$ [$^\circ$]')
    ax.grid(True)
    ax.set_xlim(0, t_plot[-1])
    
    y_peak = max(float(np.max(y_true)), float(np.max(y_fr)))
    y_valley = min(float(np.min(y_true)), float(np.min(y_fr)))
    margin_top = max(12.0, (y_peak - y_valley) * 0.22)
    margin_bottom = max(4.0, (y_peak - y_valley) * 0.08)
    ax.set_ylim(y_valley - margin_bottom, y_peak + margin_top)
    
    # Metrics box
    txt = f"RMSE = {rmse:.2f}°\nFIT = {fit:.1f}%"
    ax.text(0.03, 0.94, txt, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.85', alpha=0.9))
    
    if i == 0:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.92)

plt.suptitle(r"Polynomial NARX Free-Run Simulation on Unseen Validation Datasets", 
             fontsize=11, fontweight='bold', y=0.995)
plt.tight_layout()

png_out = os.path.join(out_dir, 'narx_val_freerun_grid.png')
pdf_out = os.path.join(out_dir, 'narx_val_freerun_grid.pdf')
fig.savefig(png_out, dpi=300)
fig.savefig(pdf_out, format='pdf')
plt.close(fig)

print("Saved:", png_out)
print("Saved:", pdf_out)
