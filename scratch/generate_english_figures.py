import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Base setup
root_dir = os.path.abspath('.')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

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

print("==================================================")
print("1. Generating 'mpc_tempo' in English...")
print("==================================================")

img_path = os.path.join(root_dir, 'data', 'experimentos_novos', 'EXP001_legado_narx-mpc', '2_mpc_python', 'MPC_Computational_Time.png')
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

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.plot(t_extracted, dt_ms, color='#1f77b4', linestyle='-', linewidth=1.2, 
        label='NMPC Solution Time (IPOPT)')
ax.axhline(10.0, color='#d62728', linestyle='--', linewidth=1.4, 
           label=r'Sampling Period $T_s = 10$ ms (Deadline)')

ax.annotate('Initial cold start\n(458 ms)', 
            xy=(0.0, 458), xytext=(1.8, 420),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5),
            fontsize=8.5)

ax.annotate('Step transition\npeak (333 ms)', 
            xy=(9.17, 333), xytext=(11.5, 320),
            arrowprops=dict(facecolor='black', shrink=0.08, width=0.8, headwidth=5),
            fontsize=8.5)

ax.set_xlabel('Time [s]')
ax.set_ylabel('Solution Time [ms]')
ax.set_xlim(0, 25)
ax.set_ylim(0, 500)
ax.grid(True)
ax.legend(loc='upper right', framealpha=0.95)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'mpc_tempo.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'mpc_tempo.pdf'), format='pdf')
plt.close(fig)
print("-> Saved: mpc_tempo.png / .pdf")


print("\n==================================================")
print("2. Generating 'comparacao_hibrido_val1' in English...")
print("==================================================")

real_dir = os.path.join(root_dir, 'data', 'experimentos_novos', 'EXP001_reteste_narx-mpc', '4_aeropendulo')
f_sem = os.path.join(real_dir, "20261023_180508_referencia_legado_val1_mpc_hybrid.csv")
f_com = os.path.join(real_dir, "20261023_181157_referencia_legado_val1_mpc_hybrid.csv")

df_s = pd.read_csv(f_sem)
df_c = pd.read_csv(f_com)

t_s = df_s["tempo_ms"] / 1000.0
t_c = df_c["tempo_ms"] / 1000.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, 
                               gridspec_kw={'height_ratios': [2.2, 1.2]})

# Subplot 1: Angle
ax1.plot(t_c, df_c["referencia"], color="black", linestyle="--", linewidth=1.1, alpha=0.85, label="Reference")
ax1.plot(t_s, df_s["angulo_deg"], color="#1f77b4", linestyle="-", linewidth=1.2, alpha=0.85, label="Bench Nominal (No Integrator)")
ax1.plot(t_c, df_c["angulo_deg"], color="#d62728", linestyle="-", linewidth=1.3, alpha=0.9, label="Bench Hybrid (With Integrator)")

ax1.set_ylabel(r"Angle $\theta$ [$^\circ$]")
ax1.grid(True)
ax1.legend(loc="lower right", ncol=3, framealpha=0.95)

# Subplot 2: Control
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax2.axhline(80, color="gray", linestyle="--", linewidth=0.7)
ax2.plot(t_s, df_s["u_pct"], color="#1f77b4", linestyle="-", linewidth=1.1, alpha=0.8, label="No Integrator")
ax2.plot(t_c, df_c["u_pct"], color="#d62728", linestyle="-", linewidth=1.2, alpha=0.85, label="With Integrator")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel(r"Control $u$ [\%]")
ax2.set_ylim(-15, 95)
ax2.grid(True)
ax2.legend(loc="upper right", ncol=2, framealpha=0.95)

max_t = min(t_s.max(), t_c.max())
ax1.set_xlim(0, max_t)
plt.tight_layout()

fig.savefig(os.path.join(out_dir, 'comparacao_hibrido_val1.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'comparacao_hibrido_val1.pdf'), format="pdf")
plt.close(fig)
print("-> Saved: comparacao_hibrido_val1.png / .pdf")


print("\n==================================================")
print("3. Generating 'narx_train_datasets' in English...")
print("==================================================")
from aerodata import readData

TRIM_START_SEC = 10.0
TRIM_END_SEC = 13.0

def load_processed(fname, trim_start=TRIM_START_SEC, trim_end=TRIM_END_SEC, decimation=1):
    y, u, t, ref = readData(dataset_name=fname, return_ref=True)
    t_start_target = t[0] + trim_start
    t_end_target = t[-1] - trim_end
    idx_start = np.searchsorted(t, t_start_target)
    idx_end = np.searchsorted(t, t_end_target)
    if idx_start >= idx_end:
        idx_start, idx_end = 0, len(t)
    sl = slice(idx_start, idx_end, decimation)
    u_c, y_c, t_c = u[sl], y[sl], t[sl]
    t_c = t_c - t_c[0]
    ref_c = ref[sl] if len(ref) > 0 else np.full_like(t_c, np.nan)
    return u_c, y_c, t_c, ref_c

train_files_en = {
    'multisine_45_train': {
        'file': 'RODADA-7/multi-seno-45-050Hz_0904_20-29.csv',
        'label': r'Multisine ($45^\circ$, $0.50$ Hz)'
    },
    'chirp_45_train': {
        'file': 'RODADA-7/chirp-45-amp45_0904_20-15.csv',
        'label': r'Chirp ($45^\circ$, Amp. $\pm 45^\circ$)'
    },
    'steps_45_train': {
        'file': 'RODADA-7/aprbs-45-1_0904_19-54.csv',
        'label': r'APRBS ($45^\circ$)'
    },
    'multisine_60_train': {
        'file': 'RODADA-7/multi-seno-60-050Hz_0904_20-38.csv',
        'label': r'Multisine ($60^\circ$, $0.50$ Hz)'
    },
    'chirp_60_train': {
        'file': 'RODADA-7/chirp-60-amp60_0904_20-20.csv',
        'label': r'Chirp ($60^\circ$, Amp. $\pm 60^\circ$)'
    },
    'steps_60_train': {
        'file': 'RODADA-7/aprbs-60-1_0904_20-04.csv',
        'label': r'APRBS ($60^\circ$)'
    }
}

fig, axes = plt.subplots(3, 2, figsize=(11.0, 7.0), sharex=False)
axes_flat = axes.flatten()

for i, (key, meta) in enumerate(train_files_en.items()):
    ax = axes_flat[i]
    u, y, t, ref = load_processed(meta['file'])
    
    # Plot reference r(t) and measured angle theta(t)
    ax.plot(t, ref, color='#555555', linestyle='--', linewidth=1.1, alpha=0.85, label=r'Reference $r(t)$')
    ax.plot(t, y, color='#1f77b4', linestyle='-', linewidth=1.2, alpha=0.9, label=r'Measured $\theta(t)$')
    
    ax.set_ylabel(r'Angle [$^\circ$]')
    ax.set_xlabel('Time [s]')
    ax.set_title(meta['label'], fontsize=9.5, fontweight='bold', pad=4)
    ax.grid(True)
    ax.set_xlim(0, t[-1])
    
    y_min_val = min(float(np.nanmin(ref)), float(np.nanmin(y)))
    y_max_val = max(float(np.nanmax(ref)), float(np.nanmax(y)))
    y_min = np.floor(y_min_val / 10.0) * 10.0 - 5.0
    y_max = np.ceil(y_max_val / 10.0) * 10.0 + 15.0  # extra headroom for legend
    ax.set_ylim(y_min, y_max)
    
    if i == 0:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.92)

plt.suptitle(r"Closed-Loop Identification Training Experiments: Reference $r(t)$ vs. Measured Angle $\theta(t)$ ($f_{\max} = 0.50$ Hz)", 
             fontsize=11, fontweight='bold', y=0.995)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'narx_train_datasets.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'narx_train_datasets.pdf'), format='pdf')
plt.close(fig)
print("-> Saved: narx_train_datasets.png / .pdf")


print("\n==================================================")
print("4. Generating 'narx_val_chirp' in English...")
print("==================================================")
from sysid import NARX
import re

# Build training data list of tuples [(u, y), ...]
train_data = []
for meta in train_files_en.values():
    u, y, t, ref = load_processed(meta['file'])
    train_data.append((u, y))

narx_model = NARX(nu=15, ny=15, poly_order_l=2, n_components=10)
narx_model.fit(train_data)
ml = narx_model._max_lag_internal_

def compile_fast_evaluator(model):
    parsed = []
    for term, th in zip(model.selected_P_colnames_, model.theta_):
        th_val = float(th)
        if term == 'constant':
            parsed.append(('const', th_val))
            continue
        matches = re.findall(r'([yu])\(k-(\d+)\)', term)
        if len(matches) == 1:
            var, lag = matches[0]
            parsed.append(('linear', th_val, var, int(lag)))
        elif len(matches) == 2:
            (var1, lag1), (var2, lag2) = matches
            parsed.append(('quad', th_val, var1, int(lag1), var2, int(lag2)))
    return parsed

def simulate_freerun(model, u_in, y_init, bounds=(-10.0, 135.0)):
    N = len(u_in)
    y_hat = np.zeros(N)
    y_hat[:ml] = y_init[:ml]
    parsed_rules = compile_fast_evaluator(model)
    for k in range(ml, N):
        val = 0.0
        for rule in parsed_rules:
            rtype = rule[0]
            th = rule[1]
            if rtype == 'const':
                val += th
            elif rtype == 'linear':
                val += th * (y_hat[k - rule[3]] if rule[2] == 'y' else u_in[k - rule[3]])
            elif rtype == 'quad':
                v1 = y_hat[k - rule[3]] if rule[2] == 'y' else u_in[k - rule[3]]
                v2 = y_hat[k - rule[5]] if rule[4] == 'y' else u_in[k - rule[5]]
                val += th * (v1 * v2)
        if bounds is not None:
            val = np.clip(val, bounds[0], bounds[1])
        y_hat[k] = val
    return y_hat[ml:]

# Load validation chirp 45
u_val, y_val, t_val, ref_val = load_processed('RODADA-7/chirp-45-amp35_0904_20-13.csv')
y_fr = simulate_freerun(narx_model, u_val, y_val)
t_val = t_val[ml:]
y_true = y_val[ml:]
u_plot = u_val[ml:]
ref_plot = ref_val[ml:]

err = y_true - y_fr
rmse = np.sqrt(np.mean(err ** 2))
mae = np.mean(np.abs(err))
denom = np.linalg.norm(y_true - np.mean(y_true))
fit = 100.0 * (1.0 - np.linalg.norm(err) / denom)
print(f"Chirp validation metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, FIT={fit:.1f}%")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.3), sharex=True,
                               gridspec_kw={'height_ratios': [2.2, 1.2]})

ax1.plot(t_val, y_true, color='#111111', lw=1.1, label=r'Measured $y(k)$')
ax1.plot(t_val, y_fr, color='#d62728', ls='--', lw=1.3, label=r'NARX Free-Run $\hat{y}(k)$')
if not np.all(np.isnan(ref_plot)):
    ax1.plot(t_val, ref_plot, color='#7f7f7f', ls=':', lw=0.9, alpha=0.7, label=r'Reference $r(k)$')

ax1.set_ylabel(r'Angle $\theta$ [$^\circ$]')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.92,
           borderpad=0.25, labelspacing=0.2, handlelength=1.2, handletextpad=0.35)
ax1.set_xlim(0, t_val[-1])

y_peak = max(float(np.max(y_true)), float(np.max(y_fr)))
y_valley = min(float(np.min(y_true)), float(np.min(y_fr)))
margin_top = max(16.0, (y_peak - y_valley) * 0.20)
margin_bottom = max(5.0, (y_peak - y_valley) * 0.05)
ax1.set_ylim(y_valley - margin_bottom, y_peak + margin_top)

metric_text = f"RMSE = {rmse:.3f}°\nMAE = {mae:.3f}°\nFIT = {fit:.1f}%"
ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes,
         verticalalignment='top', fontsize=7.5,
         bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.85', alpha=0.92))

ax2.plot(t_val, u_plot, color='#1f77b4', lw=1.0, label=r'Control input $u(k)$')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel(r'Control $u$ [%]')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.92,
           borderpad=0.25, labelspacing=0.2, handlelength=1.2)

u_peak = float(np.max(u_plot))
u_valley = float(np.min(u_plot))
ax2.set_ylim(u_valley - 4.0, u_peak + max(12.0, (u_peak - u_valley) * 0.22))

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'narx_val_chirp.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'narx_val_chirp.pdf'), format='pdf')
plt.close(fig)
print("-> Saved: narx_val_chirp.png / .pdf")

print("\nAll 4 figures successfully generated in English!")
