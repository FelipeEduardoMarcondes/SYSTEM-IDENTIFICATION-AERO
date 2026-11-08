import json
import os

notebook_path = r"sysid\notebooks\identificação-collab\IDENT-FELIPE\BUSCA_NA_NB_BLS.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_0 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, os, sys
from scipy.signal import decimate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# Garante acesso à raiz do repositório para importar aerodata.py
for parent in ['.', '..', '../..', '../../..', '../../../..']:
    p = os.path.abspath(parent)
    if os.path.exists(os.path.join(p, 'aerodata.py')) and p not in sys.path:
        sys.path.append(p)
        break

from aerodata import readData
"""

cell_1 = """# Lista de ensaios de Treinamento (F_Mix) e Validação (Cross-Session)
F_MIX_FILES = [
    "aprbs-1_0827_17-19.csv", "aprbs-2_0827_17-25.csv", "aprbs-3_0827_17-28.csv", "aprbs-4_0827_17-31.csv",
    "multi-seno-1_0827_17-34.csv", "multi-seno-2_0827_17-37.csv", "multi-seno-3_0827_17-40.csv", "multi-seno-4_0827_17-43.csv",
    "swept-sine-1_0827_17-58.csv", "swept-sine-4_0827_18-00.csv",
    "seq-degraus-1_0827_17-46.csv", "seq-degraus-2_0827_17-49.csv", "seq-degraus-3_0827_17-52.csv", "seq-degraus-4_0827_17-55.csv",
    "RODADA-3/chirp-1_0807_16-34.csv", "RODADA-3/chirp-2_0807_17-07.csv", "RODADA-3/chirp-2_0807_17-09.csv"
]

TEST_FILES_BY_TYPE = {
    "APRBS":     "RODADA-4/aprbs-2_0819_18-51.csv",
    "MultiSeno": "RODADA-2/multi-seno-1_0804_19-06.csv",
    "SweptSine": "RODADA-4/swept-sine-1_0819_19-02.csv",
    "Degraus":   "RODADA-2/seq-degraus-2_0804_19-38.csv",
    "Chirp":     "RODADA-2/chirp-1_0804_19-19.csv",
}

DECIMACAO = 1  # 1 = 100 Hz original
START_IDX = 200
END_IDX   = -200

# Normalização de entrada u [-100, 100] -> [-1, 1] e saída y [-180, 180] -> [-1, 1]
scaler_u = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_u.fit(np.array([[-100], [100]]))
scaler_y.fit(np.array([[-180], [180]]))
"""

cell_2 = """def carregar_experimento(nome_arquivo, decimacao=DECIMACAO, start_idx=START_IDX, end_idx=END_IDX):
    \"\"\"Carrega o experimento via aerodata (localmente ou via GitHub) e aplica o pré-processamento padrão.\"\"\"
    y_raw, u_raw, t_raw, ref = readData(nome_arquivo, return_ref=True)
    
    # Remove trechos iniciais inativos onde a referência ainda era <= 0
    if len(ref) > 0 and np.any(ref > 0):
        mask = ref > 0
        y_raw, u_raw, t_raw = y_raw[mask], u_raw[mask], t_raw[mask]
        
    # Subamostragem / decimação se solicitado
    if decimacao > 1:
        y_raw = decimate(y_raw, decimacao, ftype='iir', zero_phase=True)
        u_raw = u_raw[::decimacao]
        dt = 0.010 * decimacao
        t_raw = np.arange(len(y_raw)) * dt
        
    # Trimming de transientes de borda
    if start_idx is not None and end_idx is not None:
        t_raw = t_raw[start_idx:end_idx]
        u_raw = u_raw[start_idx:end_idx]
        y_raw = y_raw[start_idx:end_idx]
        
    t_raw = t_raw - t_raw[0]
    return t_raw, u_raw, y_raw

def processar_dataset(t_raw, u_raw, y_raw):
    u_norm = scaler_u.transform(u_raw.reshape(-1, 1)).flatten()
    y_norm = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()
    return t_raw, u_norm, y_norm, u_raw, y_raw

def carregar_lista(file_list):
    datasets = []
    for f in file_list:
        t, u_raw, y_raw = carregar_experimento(f)
        t, u_norm, y_norm, _, _ = processar_dataset(t, u_raw, y_raw)
        datasets.append({
            'name': os.path.basename(f),
            't': t, 'u_norm': u_norm, 'y_norm': y_norm,
            'u_raw': u_raw, 'y_raw': y_raw
        })
    return datasets
"""

cell_3 = """print("Carregando dataset misto (F_Mix) de treinamento...")
train_datasets = carregar_lista(F_MIX_FILES)

print("\\nCarregando conjuntos de Teste (Cross-Session)...")
test_datasets = []
for tipo, arquivo in TEST_FILES_BY_TYPE.items():
    test_datasets.extend(carregar_lista([arquivo]))
print(f"Carregamento concluído: {len(train_datasets)} treinos e {len(test_datasets)} testes.")
"""

cell_4 = """def create_regression_matrix(u, y, na, nb):
    u = np.asarray(u).flatten()
    y = np.asarray(y).flatten()
    N = len(y)
    p = max(na, nb)
    y_target = y[p:]
    Phi = np.zeros((N - p, na + nb))
    for i in range(1, na + 1):
        Phi[:, i - 1] = -y[p - i: N - i]
    for i in range(1, nb + 1):
        Phi[:, na + i - 1] = u[p - i: N - i]
    return Phi, y_target

def build_combined_matrices(datasets, na, nb, use_norm=True):
    Phi_all = []
    y_all = []
    for ds in datasets:
        u_data = ds['u_norm'] if use_norm else ds['u_raw']
        y_data = ds['y_norm'] if use_norm else ds['y_raw']
        Phi, y_target = create_regression_matrix(u_data, y_data, na, nb)
        Phi_all.append(Phi)
        y_all.append(y_target)
    return np.vstack(Phi_all), np.concatenate(y_all)

def free_run_simulation(th_hat, u, y_measured, na, nb):
    u = np.asarray(u).flatten()
    y_measured = np.asarray(y_measured).flatten()
    N = len(u)
    p = max(na, nb)
    y_sim = np.zeros(N)
    y_sim[:p] = y_measured[:p]
    a = th_hat[:na]
    b = th_hat[na:]
    
    # Simulação multi-passo vetorizada direta (8-10x mais rápida)
    for k in range(p, N):
        s_y = k - na - 1
        s_u = k - nb - 1
        val = -np.dot(a, y_sim[k-1 : (None if s_y < 0 else s_y) : -1]) + np.dot(b, u[k-1 : (None if s_u < 0 else s_u) : -1])
        
        # Proteção numérica contra divergência de polos instáveis
        if np.isnan(val) or np.abs(val) > 1e4:
            y_sim[k:] = 1e4
            break
        y_sim[k] = val
        
    return y_sim[p:]
"""

cell_5 = """# Configuração da varredura até 30, 30:
# Opção 1 (Padrão): Varredura diagonal na = nb de 1 a 30 (30 modelos - rápida e ideal para comparar complexidade)
ordens_teste = [(i, i) for i in range(1, 31)]

# Opção 2: Grade completa na x nb até 30x30 (descomente se desejar testar todas as 900 combinações)
# ordens_teste = [(na, nb) for na in range(1, 31) for nb in range(1, 31)]

resultados = []
use_norm = True  # Utiliza os dados normalizados em [-1, 1]

print(f"\\n--- INICIANDO BUSCA DE ORDENS (Total: {len(ordens_teste)} modelos) ---")
for idx, (na_t, nb_t) in enumerate(ordens_teste, start=1):
    print(f"[{idx}/{len(ordens_teste)}] Testando na={na_t}, nb={nb_t}...", end="\\r")
    Phi_tra, y_tra = build_combined_matrices(train_datasets, na_t, nb_t, use_norm)
    
    # Estimação por Mínimos Quadrados Ordinários (BLS)
    th_t, _, _, _ = np.linalg.lstsq(Phi_tra, y_tra, rcond=None)
    
    # Avaliação 1-Passo à Frente (One-Step-Ahead) no Treino
    osa_tra = Phi_tra @ th_t
    r2_osa_tra = r2_score(y_tra, osa_tra)
    rmse_osa_tra = np.sqrt(mean_squared_error(y_tra, osa_tra))
    
    # Simulação Livre (Free-Run) no Treinamento
    r2_fr_tra_list = []
    rmse_fr_tra_list = []
    for ds in train_datasets:
        u_data = ds['u_norm'] if use_norm else ds['u_raw']
        y_data = ds['y_norm'] if use_norm else ds['y_raw']
        fr_sim = free_run_simulation(th_t, u_data, y_data, na_t, nb_t)
        y_real_valid = y_data[max(na_t, nb_t):]
        r2_fr_tra_list.append(r2_score(y_real_valid, fr_sim))
        rmse_fr_tra_list.append(np.sqrt(mean_squared_error(y_real_valid, fr_sim)))
        
    # Simulação Livre (Free-Run) nos Ensaios de Validação/Teste
    r2_fr_tst_list = []
    rmse_fr_tst_list = []
    for ds in test_datasets:
        u_data = ds['u_norm'] if use_norm else ds['u_raw']
        y_data = ds['y_norm'] if use_norm else ds['y_raw']
        fr_sim = free_run_simulation(th_t, u_data, y_data, na_t, nb_t)
        y_real_valid = y_data[max(na_t, nb_t):]
        r2_fr_tst_list.append(r2_score(y_real_valid, fr_sim))
        rmse_fr_tst_list.append(np.sqrt(mean_squared_error(y_real_valid, fr_sim)))

    resultados.append({
        'na': na_t, 'nb': nb_t,
        'RMSE_OSA_TRA': rmse_osa_tra,
        'R2_OSA_TRA':   r2_osa_tra,
        'RMSE_FR_TRA':  np.mean(rmse_fr_tra_list),
        'R2_FR_TRA':    np.mean(r2_fr_tra_list),
        'RMSE_FR_TST':  np.mean(rmse_fr_tst_list),
        'R2_FR_TST':    np.mean(r2_fr_tst_list),
    })

print(f"\\nVarredura concluída com sucesso!")
"""

cell_6 = """df_res = pd.DataFrame(resultados)
print("\\n--- RESULTADOS FINAIS DA VARREDURA ---")
print(df_res[['na', 'nb', 'R2_OSA_TRA', 'RMSE_OSA_TRA', 'R2_FR_TRA', 'RMSE_FR_TRA', 'R2_FR_TST', 'RMSE_FR_TST']].to_string(index=False, float_format="{:.4f}".format))

best = df_res.loc[df_res['R2_FR_TST'].idxmax()]
print(f"\\nMelhor configuração (R² Free-Run na Validação): na={int(best.na)}, nb={int(best.nb)}")
print(f"  TRA -> RMSE_FR={best.RMSE_FR_TRA:.4f}  R²_FR={best.R2_FR_TRA:.4f}")
print(f"  TST -> RMSE_FR={best.RMSE_FR_TST:.4f}  R²_FR={best.R2_FR_TST:.4f}")
"""

cell_7 = """labels = [f"({int(r.na)},{int(r.nb)})" for r in df_res.itertuples()]

fig, axes = plt.subplots(2, 1, figsize=(16, 9))
fig.suptitle('Varredura de Ordens (BLS Linear) — Avaliação em Free-Run', fontsize=14)

x = np.arange(len(labels))
width = 0.35

# Gráfico de R² Free-Run
axes[0].bar(x - width/2, df_res['R2_FR_TRA'], width, label='Treinamento', alpha=0.75)
axes[0].bar(x + width/2, df_res['R2_FR_TST'], width, label='Validação (Teste)', alpha=0.75)
axes[0].set_title('R² Médio em Simulação Livre (Free-Run)')
axes[0].set_ylabel('R²')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=60, fontsize=8)
axes[0].set_ylim(-1.0, 1.05)  # Limita para manter visibilidade
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].grid(axis='y', alpha=0.5)
axes[0].legend()

# Gráfico de RMSE Free-Run
axes[1].bar(x - width/2, df_res['RMSE_FR_TRA'], width, label='Treinamento', alpha=0.75)
axes[1].bar(x + width/2, df_res['RMSE_FR_TST'], width, label='Validação (Teste)', alpha=0.75)
axes[1].set_title('RMSE Médio em Simulação Livre (Dados Normalizados)')
axes[1].set_ylabel('RMSE')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=60, fontsize=8)
axes[1].set_ylim(0, 0.4)
axes[1].grid(axis='y', alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.show()
"""

cells_src = [cell_0, cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7]

for i, src in enumerate(cells_src):
    lines = [line + '\n' for line in src.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    nb['cells'][i]['source'] = lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook successfully updated!")

