import os
import sys

root_dir = os.path.abspath('.')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from scratch.generate_english_figures import load_processed, train_files_en, simulate_freerun
from sysid import NARX
import numpy as np

# Reconfigure stdout to utf-8
sys.stdout.reconfigure(encoding='utf-8')

train_data = []
for meta in train_files_en.values():
    u, y, t, ref = load_processed(meta['file'])
    train_data.append((u, y))

narx_model = NARX(nu=15, ny=15, poly_order_l=2, n_components=10)
narx_model.fit(train_data)
ml = narx_model._max_lag_internal_

folder = os.path.join(root_dir, 'data', 'experimentos', 'coleta_dados', 'RODADA-7')
files_60 = sorted([f for f in os.listdir(folder) if '-60-' in f and f.endswith('.csv')])

print(f"=== RESULTADOS DO NARX (10 TERMOS) EM FREE-RUN NOS DATASETS 60° (DC 60) ===\n")
print(f"{'Arquivo / Experimento':38s} | {'Tipo':7s} | {'RMSE':6s} | {'FIT':6s} | {'MAE':6s} | {'R2':6s}")
print('-'*80)

train_file_basenames = [os.path.basename(meta['file']) for meta in train_files_en.values()]

for f in files_60:
    rel_path = os.path.join('RODADA-7', f)
    u, y, t, ref = load_processed(rel_path)
    y_fr = simulate_freerun(narx_model, u, y)
    
    y_true = y[ml:]
    y_pred = y_fr # simulate_freerun already returns y_hat[ml:] or y_hat? Let's check
    if len(y_fr) == len(y):
        y_pred = y_fr[ml:]
    else:
        y_pred = y_fr
        
    err = y_true - y_pred
    
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    den = np.linalg.norm(y_true - np.mean(y_true))
    fit = 100.0 * (1.0 - np.linalg.norm(err) / den) if den > 1e-6 else 0.0
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - np.sum(err**2) / ss_tot if ss_tot > 1e-6 else 0.0
    
    is_train = f in train_file_basenames
    tipo_str = 'TREINO' if is_train else 'VALID'
    
    print(f"{f[:38]:38s} | {tipo_str:7s} | {rmse:5.2f}° | {fit:5.1f}% | {mae:5.2f}° | {r2:6.3f}")

# MIX_DC_60
mix_path = os.path.join(folder, 'MIX_DC_60.csv')
if os.path.exists(mix_path):
    u, y, t, ref = load_processed('RODADA-7/MIX_DC_60.csv')
    y_fr = simulate_freerun(narx_model, u, y)
    y_true = y[ml:]
    y_pred = y_fr[ml:] if len(y_fr) == len(y) else y_fr
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    den = np.linalg.norm(y_true - np.mean(y_true))
    fit = 100.0 * (1.0 - np.linalg.norm(err) / den)
    r2 = 1.0 - np.sum(err**2) / np.sum((y_true - np.mean(y_true))**2)
    print('-'*80)
    print(f"{'MIX_DC_60.csv (concatenado total)':38s} | {'CONCAT':7s} | {rmse:5.2f}° | {fit:5.1f}% | {mae:5.2f}° | {r2:6.3f}")

