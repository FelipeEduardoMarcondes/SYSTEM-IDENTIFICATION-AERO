# %% [markdown]
# # NARX model
# 
# NARX model for the 1/4 drone dataset.

# %%
import numpy as np
import matplotlib.pyplot as plt

# --- Install/import the sysid course library (works locally and on Google Colab) ---
try:
    from sysid import NARX
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/helonayala/sysid.git'])
    from sysid import NARX

import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path para importar aerodata
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from aerodata import readData


# %% [markdown]
# ## Load and preprocess the datasets
# 
# Closed-loop 1/4 drone acquisitions. Each signal is sliced to a fixed useful time window (20–80 s) and decimated by a common factor, then stored in a `data` dictionary.

# %%
DECIMATION = 5
TRIM_START_PCT = 0.02  # Descarta os primeiros 5% do sinal
TRIM_END_PCT = 0.05    # Descarta os ultimos 5% do sinal

def load_processed(name, trim_start=TRIM_START_PCT, trim_end=TRIM_END_PCT, decimation=DECIMATION):
    """Load a dataset from aerodata, trim by percentage, and decimate. Returns u, y, t, ref."""
    # Carrega o sinal original inteiro do aerodata
    y, u, t, ref = readData(dataset_name=name, return_ref=True)
    
    # Calcula os índices de corte baseado na porcentagem
    N = len(y)
    idx_start = int(N * trim_start)
    idx_end = int(N * (1.0 - trim_end))
    
    # Aplica o corte e a decimação
    sl = slice(idx_start, idx_end, decimation)
    u, y, t = u[sl], y[sl], t[sl]
    
    if len(ref) > 0:
        ref = ref[sl]
    else:
        ref = np.full_like(t, np.nan)
        
    return u, y, t, ref

def plot_io(u, y, t, ref, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(t, ref, '--', color='red', label='Referência')
    ax1.plot(t, y, color='blue', label='Saída y')
    ax1.set_ylabel('Ângulo [°]'); ax1.set_title(title)
    ax1.legend(loc='upper right'); ax1.grid(True)
    ax2.plot(t, u, color='green', label='Controle u')
    ax2.set_xlabel('Tempo [s]'); ax2.set_ylabel('u')
    ax2.legend(loc='upper right'); ax2.grid(True)
    plt.tight_layout(); plt.show()

train_datasets = {
    'mix_45': load_processed('data/experimentos/RODADA-7/MIX_DC_45.csv'),
    'mix_60': load_processed('data/experimentos/RODADA-7/MIX_DC_60.csv'),
}

val_datasets = {
    'multisine': load_processed('data/experimentos/RODADA-7/multi-seno-60-030Hz_0904_20-32.csv'),
    'chirp':     load_processed('data/experimentos/RODADA-7/chirp-60-amp50_0904_20-18.csv'),
    'steps':     load_processed('data/experimentos/RODADA-7/seq-degraus-45-2_0904_20-45.csv'),
}

# %% [markdown]
# ## Processed datasets

# %%
for name, d in train_datasets.items():
    plot_io(*d, f'Treinamento: {name}')
for name, d in val_datasets.items():
    plot_io(*d, f'Validação: {name}')

# %% [markdown]
# ## Train / test split
# 
# **Explicit split**: `mix_45` and `mix_60` are used entirely for training. Validation sets are used entirely for testing.

# %%
train_data, test_mask = [], {}

# Use entire train datasets for fitting
for name, d in train_datasets.items():
    u, y, t, ref = d
    train_data.append((u, y))

# Test masks for validation sets are all True (100% test)
for name, d in val_datasets.items():
    u, y, t, ref = d
    test_mask[name] = np.ones(len(y), bool)

print(f'Training blocks: {len(train_data)} (entire datasets)')

# %% [markdown]
# ## Model identification (multiple datasets)

# %%
ny_model = 15
nu_model = 15
poly_order_model = 2
n_components = 20

narx_model = NARX(nu=nu_model, ny=ny_model, poly_order_l=poly_order_model,
                  n_components=n_components)
narx_model.fit(train_data)
narx_model.print()


# %% [markdown]
# ## Final evaluation
# 
# Free-run simulation over each validation dataset. The reported RMSE is computed on the entire signal.

# %%
def free_run_full(name, title=None):
    u, y, t, ref = val_datasets[name]
    ml = narx_model._max_lag_internal_
    y_fr = narx_model.predict(u, y_history_for_lags_or_osa=y[:ml], mode='FR')
    tt, ym = t[ml:], y[ml:]
    m = test_mask[name][ml:]
    rmse_test = np.sqrt(np.mean((ym[m] - y_fr[m]) ** 2))
    plt.figure(figsize=(12, 5))
    plt.plot(tt, ym, color='black', label='Measured y(k)')
    plt.plot(tt, y_fr, '--', color='crimson', label='NARX free-run')
    plt.axvspan(t[ml], t[-1], color='orange', alpha=0.15, label='Test region')
    plt.xlabel('Tempo [s]'); plt.ylabel('Ângulo [°]')
    plt.title(f'{title or name} - full free-run  (test RMSE = {rmse_test:.3f})')
    plt.legend(loc='upper right'); plt.grid(True); plt.tight_layout(); plt.show()
    return rmse_test

print('\nFree-run test RMSE [deg]:')
results = {}
for name in val_datasets.keys():
    results[name] = free_run_full(name, title=name.capitalize())
    print(f'  {name:10s} : {results[name]:.3f}')

# %%
import json

model_data = {
    'ny': narx_model.ny,
    'nu': narx_model.nu,
    'l': narx_model.poly_order_l,
    'terms': list(narx_model.selected_P_colnames_),
    'theta': [float(t) for t in narx_model.theta_]
}

with open('narx_model.json', 'w') as f:
    json.dump(model_data, f, indent=4)

print("\nModel parameters exported to narx_model.json successfully!")
