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
DECIMATION = 1        # Aumente isso para 10 ou 20 se o treino estiver demorando muito!
TRIM_START_SEC = 10.0  # Descarta exatamente os primeiros 10 segundos
TRIM_END_SEC = 13.0     # Descarta exatamente os últimos 5 segundos

def load_processed(name, trim_start=TRIM_START_SEC, trim_end=TRIM_END_SEC, decimation=DECIMATION):
    """Load a dataset from aerodata, trim by fixed time margins, and decimate. Returns u, y, t, ref."""
    # Carrega o sinal original inteiro do aerodata
    y, u, t, ref = readData(dataset_name=name, return_ref=True)
    
    # Calcula os índices baseados no tempo (ex: tempo inicial + 10s)
    t_start_target = t[0] + trim_start
    t_end_target = t[-1] - trim_end
    
    idx_start = np.searchsorted(t, t_start_target)
    idx_end = np.searchsorted(t, t_end_target)
    
    # Se por acaso o corte for maior que o sinal, pega tudo
    if idx_start >= idx_end:
        idx_start, idx_end = 0, len(t)
    
    # Aplica o corte e a decimação
    sl = slice(idx_start, idx_end, decimation)
    u, y, t = u[sl], y[sl], t[sl]
    
    if len(ref) > 0:
        ref = ref[sl]
    else:
        ref = np.full_like(t, np.nan)
        
    return u, y, t, ref

def plot_all_datasets(datasets, main_title):
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3*n), sharex=False)
    if n == 1: axes = [axes]
    for ax, (name, d) in zip(axes, datasets.items()):
        u, y, t, ref = d
        ax.plot(t, ref, '--', color='red', label='Ref')
        ax.plot(t, y, color='blue', label='y')
        ax.plot(t, u, color='green', label='u', alpha=0.5)
        ax.set_title(name)
        ax.legend(loc='upper right')
        ax.grid(True)
    plt.suptitle(main_title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

val_files = {
    'multisine_45_val': 'multi-seno-45-030Hz_0904_20-23.csv',
    'chirp_45_val':     'chirp-45-amp35_0904_20-13.csv',
    'steps_45_val':     'seq-degraus-45-2_0904_20-45.csv'
}

val_datasets = {
    name: load_processed(f'data/experimentos/RODADA-7/{fname}')
    for name, fname in val_files.items()
}

train_files = {
    'multisine_45_train': 'multi-seno-45-040Hz_0904_20-26.csv',
    'chirp_45_train':     'chirp-45-amp25_0904_20-11.csv',
    'steps_45_train':     'aprbs-45-1_0904_19-54.csv',
    'multisine_60_train': 'multi-seno-60-030Hz_0904_20-32.csv',
    'chirp_60_train':     'chirp-60-amp40_0904_20-16.csv',
    'steps_60_train':     'aprbs-60-1_0904_20-04.csv'
}

train_datasets = {
    name: load_processed(f'data/experimentos/RODADA-7/{fname}')
    for name, fname in train_files.items()
}

# %% [markdown]
# ## Processed datasets

# Plot Treinamento
plot_all_datasets(train_datasets, "Conjuntos de Treinamento")

# Plot Validação
plot_all_datasets(val_datasets, "Conjuntos de Validação")

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
n_components = 10

narx_model = NARX(nu=nu_model, ny=ny_model, poly_order_l=poly_order_model,
                  n_components=n_components)
narx_model.fit(train_data)
narx_model.print()


# %% [markdown]
# ## Final evaluation
# 
# Free-run simulation over each validation dataset. The reported RMSE is computed on the entire signal.

# %%
def plot_all_free_runs(val_datasets, narx_model):
    n = len(val_datasets)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3*n), sharex=False)
    if n == 1: axes = [axes]
    ml = narx_model._max_lag_internal_
    results = {}
    for ax, (name, d) in zip(axes, val_datasets.items()):
        u, y, t, ref = d
        y_fr = narx_model.predict(u, y_history_for_lags_or_osa=y[:ml], mode='FR')
        tt, ym = t[ml:], y[ml:]
        m = test_mask[name][ml:]
        rmse_test = np.sqrt(np.mean((ym[m] - y_fr[m]) ** 2))
        results[name] = rmse_test
        ax.plot(tt, ym, color='black', label='Measured y')
        ax.plot(tt, y_fr, '--', color='crimson', label='NARX Free-Run')
        ax.axvspan(t[ml], t[-1], color='orange', alpha=0.15)
        ax.set_title(f"{name} (RMSE = {rmse_test:.3f})")
        ax.legend(loc='upper right')
        ax.grid(True)
    plt.suptitle("Free-Run Validation")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return results

print('\nFree-run test RMSE [deg]:')
results = plot_all_free_runs(val_datasets, narx_model)
for name, rmse in results.items():
    print(f'  {name:10s} : {rmse:.3f}')

# %%
import json

model_data = {
    'ny': narx_model.ny,
    'nu': narx_model.nu,
    'l': narx_model.poly_order_l,
    'terms': list(narx_model.selected_P_colnames_),
    'theta': [float(t) for t in narx_model.theta_]
}

caminho_json = os.path.join(current_dir, 'narx_model.json')
with open(caminho_json, 'w') as f:
    json.dump(model_data, f, indent=4)

print(f"\nModel parameters exported to {caminho_json} successfully!")
plt.show()
