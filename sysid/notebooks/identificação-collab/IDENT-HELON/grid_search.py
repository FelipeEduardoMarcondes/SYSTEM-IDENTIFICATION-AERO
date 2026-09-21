import os
import sys
import glob
import numpy as np
import warnings
import io

# Ignora os avisos de overflow para não sujar o terminal
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuração de Caminhos ---
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from aerodata import readData
from sysid import NARX

# --- Funções de Carregamento ---
DECIMATION = 10
TRIM_START_SEC = 10.0
TRIM_END_SEC = 10.0

def load_processed(name, trim_start=TRIM_START_SEC, trim_end=TRIM_END_SEC, decimation=DECIMATION):
    y, u, t, ref = readData(dataset_name=name, return_ref=True)
    t_start_target, t_end_target = t[0] + trim_start, t[-1] - trim_end
    idx_start = np.searchsorted(t, t_start_target)
    idx_end = np.searchsorted(t, t_end_target)
    if idx_start >= idx_end: idx_start, idx_end = 0, len(t)
    sl = slice(idx_start, idx_end, decimation)
    return u[sl], y[sl], t[sl], ref[sl] if len(ref) > 0 else np.full_like(t[sl], np.nan)

# --- Carregar Dados ---
print("Carregando datasets de treino e validação...")
val_files = {
    'multisine': 'multi-seno-60-030Hz_0904_20-32.csv',
    'chirp':     'chirp-45-amp35_0904_20-13.csv',
    'steps':     'seq-degraus-45-2_0904_20-45.csv'
}

val_datasets = {name: load_processed(f'data/experimentos/RODADA-7/{fname}') for name, fname in val_files.items()}
test_mask = {name: np.ones(len(y), bool) for name, (u, y, t, ref) in val_datasets.items()}

train_data = []
rodada7_dir = os.path.join(root_dir, 'data', 'experimentos', 'RODADA-7')
for file_path in sorted(glob.glob(os.path.join(rodada7_dir, '*.csv'))):
    fname = os.path.basename(file_path)
    if "MIX_" in fname or fname in val_files.values():
        continue
    if "-45-" in fname or "-60-" in fname:
        dataset_path = f"data/experimentos/RODADA-7/{fname}"
        u, y, t, ref = load_processed(dataset_path)
        train_data.append((u, y))

# --- GRID SEARCH ---
poly_order = 2
n_components = 15

best_rmse = float('inf')
best_params = None

print("\nIniciando Grid Search (ny de 1 a 15, nu de 1 a 15)...")
print("Isso pode levar alguns minutos. Modelos instáveis (nan) serão ignorados.\n")

old_stdout = sys.stdout

for ny in range(1, 16):
    for nu in range(1, 16):
        try:
            model = NARX(nu=nu, ny=ny, poly_order_l=poly_order, n_components=n_components)
            
            # Ocultamos a saída padrão do fit para não floodar o terminal
            sys.stdout = io.StringIO()
            model.fit(train_data)
            sys.stdout = old_stdout
            
            # Avalia o erro em Free-Run
            rmses = []
            for name, (u, y, t, ref) in val_datasets.items():
                ml = model._max_lag_internal_
                y_fr = model.predict(u, y_history_for_lags_or_osa=y[:ml], mode='FR')
                ym = y[ml:]
                m = test_mask[name][ml:]
                rmse = np.sqrt(np.mean((ym[m] - y_fr[m]) ** 2))
                rmses.append(rmse)
                
            avg_rmse = np.mean(rmses)
            
            if np.isnan(avg_rmse):
                print(f"ny={ny:2d}, nu={nu:2d} -> Instável (nan)")
            else:
                print(f"ny={ny:2d}, nu={nu:2d} -> RMSE Médio: {avg_rmse:.3f}", end="")
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params = (ny, nu)
                    print("  <-- NOVO MELHOR!")
                else:
                    print()
                    
        except KeyboardInterrupt:
            sys.stdout = old_stdout
            print("\nBusca interrompida pelo usuário.")
            sys.exit(0)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"ny={ny:2d}, nu={nu:2d} -> Erro: {e}")

print("\n" + "="*50)
if best_params:
    print(f"MELHOR MODELO ENCONTRADO:")
    print(f"ny = {best_params[0]}")
    print(f"nu = {best_params[1]}")
    print(f"RMSE Médio = {best_rmse:.3f}")
else:
    print("Nenhum modelo estável encontrado.")
print("="*50)
