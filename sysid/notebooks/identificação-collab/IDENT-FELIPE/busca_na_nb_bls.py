"""
busca_na_nb_bls.py
==================
Varredura e Otimização de Ordens (na, nb) para Identificação Linear ARX/BLS
utilizando o aerodata.py para carregamento oficial dos ensaios da bancada física.

Funcionalidades:
  1. Carregamento padronizado de dados via `from aerodata import readData` (local ou GitHub).
  2. Varredura flexível de ordens até (30, 30) no modelo ARX estimado por BLS (Ordinary Least Squares).
  3. Simulação livre multi-passo (Free-Run) vetorizada de alta performance com proteção numérica.
  4. Exportação dos resultados:
     - CSV consolidado de métricas (OSA e Free-Run no Treino e Teste).
     - JSON com parâmetros e coeficientes do melhor modelo identificado.
     - Gráficos comparativos de R² e RMSE salvos em PNG (300 DPI) e PDF vetorial.

Uso:
  python busca_na_nb_bls.py                # Varredura diagonal na = nb de 1 a 30 (padrão, rápida)
  python busca_na_nb_bls.py --grid         # Grade completa 30x30 (900 modelos)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# ==============================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS E IMPORTAÇÃO DO AERODATA
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))

for p in (root_dir, current_dir):
    if p not in sys.path:
        sys.path.append(p)

try:
    from aerodata import readData
except ImportError:
    # Se ainda não estiver no path, tenta encontrar recursivamente
    for parent in ['.', '..', '../..', '../../..', '../../../..']:
        p = os.path.abspath(parent)
        if os.path.exists(os.path.join(p, 'aerodata.py')) and p not in sys.path:
            sys.path.append(p)
            break
    from aerodata import readData

# Diretório de exportação dos resultados
OUT_DIR = os.path.join(current_dir, "RESULTADOS_BUSCA_BLS")
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================================================
# 2. DEFINIÇÃO DOS ENSAIOS E PRÉ-PROCESSAMENTO
# ==============================================================================
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

DECIMACAO = 1       # 1 = 100 Hz nativo da bancada
START_IDX = 200     # Descarta transiente de início
END_IDX   = -200    # Descarta repouso final

# Normalização padrão [-1, 1]
scaler_u = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_u.fit(np.array([[-100], [100]]))
scaler_y.fit(np.array([[-180], [180]]))

def carregar_experimento(nome_arquivo, decimacao=DECIMACAO, start_idx=START_IDX, end_idx=END_IDX):
    """Carrega dados via aerodata.py e aplica filtros de repouso e trimming."""
    y_raw, u_raw, t_raw, ref = readData(nome_arquivo, return_ref=True)
    
    # Filtra período inativo inicial onde a referência ainda era zero
    if len(ref) > 0 and np.any(ref > 0):
        mask = ref > 0
        y_raw, u_raw, t_raw = y_raw[mask], u_raw[mask], t_raw[mask]
        
    if decimacao > 1:
        y_raw = decimate(y_raw, decimacao, ftype='iir', zero_phase=True)
        u_raw = u_raw[::decimacao]
        dt = 0.010 * decimacao
        t_raw = np.arange(len(y_raw)) * dt
        
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

# ==============================================================================
# 3. REGRESSÃO LINEAR ARX / BLS E SIMULAÇÃO LIVRE
# ==============================================================================
def create_regression_matrix(u, y, na, nb):
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
    """Simulação livre vetorizada direta com proteção numérica contra divergência."""
    u = np.asarray(u).flatten()
    y_measured = np.asarray(y_measured).flatten()
    N = len(u)
    p = max(na, nb)
    y_sim = np.zeros(N)
    y_sim[:p] = y_measured[:p]
    a = th_hat[:na]
    b = th_hat[na:]
    
    for k in range(p, N):
        s_y = k - na - 1
        s_u = k - nb - 1
        val = -np.dot(a, y_sim[k-1 : (None if s_y < 0 else s_y) : -1]) + np.dot(b, u[k-1 : (None if s_u < 0 else s_u) : -1])
        
        # Proteção numérica contra estouro de modelos instáveis
        if np.isnan(val) or np.abs(val) > 1e4:
            y_sim[k:] = 1e4
            break
        y_sim[k] = val
        
    return y_sim[p:]

# ==============================================================================
# 4. ROTINA PRINCIPAL DE EXECUÇÃO
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Varredura de Ordens ARX/BLS usando aerodata.py")
    parser.add_argument("--max_na", type=int, default=30, help="Ordem máxima de na (padrão: 30)")
    parser.add_argument("--max_nb", type=int, default=30, help="Ordem máxima de nb (padrão: 30)")
    parser.add_argument("--grid", action="store_true", help="Executar grade completa 2D (na x nb) ao invés da varredura diagonal")
    args = parser.parse_args()

    print("=" * 78)
    print("VARREDURA DE ORDENS ARX / BLS — AEROPÊNDULO (100 Hz)")
    print("=" * 78)

    print("\n[1/4] Carregando datasets de treinamento via aerodata.py...")
    train_datasets = carregar_lista(F_MIX_FILES)
    print(f"  -> {len(train_datasets)} ensaios carregados para treinamento.")

    print("\n[2/4] Carregando datasets de teste (Cross-Session) via aerodata.py...")
    test_datasets = []
    for tipo, arquivo in TEST_FILES_BY_TYPE.items():
        test_datasets.extend(carregar_lista([arquivo]))
    print(f"  -> {len(test_datasets)} ensaios carregados para validação.")

    # Definição das ordens
    if args.grid:
        ordens_teste = [(na, nb) for na in range(1, args.max_na + 1) for nb in range(1, args.max_nb + 1)]
        print(f"\n[3/4] Iniciando grade 2D completa ({args.max_na}x{args.max_nb} = {len(ordens_teste)} modelos)...")
    else:
        max_ord = min(args.max_na, args.max_nb)
        ordens_teste = [(i, i) for i in range(1, max_ord + 1)]
        print(f"\n[3/4] Iniciando varredura diagonal na = nb de 1 a {max_ord} ({len(ordens_teste)} modelos)...")

    resultados = []
    use_norm = True
    t_start = time.time()
    best_th = None
    best_config = None
    best_r2_tst = -np.inf

    for idx, (na_t, nb_t) in enumerate(ordens_teste, start=1):
        Phi_tra, y_tra = build_combined_matrices(train_datasets, na_t, nb_t, use_norm)
        th_t, _, _, _ = np.linalg.lstsq(Phi_tra, y_tra, rcond=None)
        
        # One-Step-Ahead Treino
        osa_tra = Phi_tra @ th_t
        r2_osa_tra = r2_score(y_tra, osa_tra)
        rmse_osa_tra = np.sqrt(mean_squared_error(y_tra, osa_tra))
        
        # Free-Run Treino
        r2_fr_tra_list, rmse_fr_tra_list = [], []
        for ds in train_datasets:
            fr_sim = free_run_simulation(th_t, ds['u_norm'], ds['y_norm'], na_t, nb_t)
            y_real = ds['y_norm'][max(na_t, nb_t):]
            r2_fr_tra_list.append(r2_score(y_real, fr_sim))
            rmse_fr_tra_list.append(np.sqrt(mean_squared_error(y_real, fr_sim)))
            
        # Free-Run Validação / Teste
        r2_fr_tst_list, rmse_fr_tst_list = [], []
        for ds in test_datasets:
            fr_sim = free_run_simulation(th_t, ds['u_norm'], ds['y_norm'], na_t, nb_t)
            y_real = ds['y_norm'][max(na_t, nb_t):]
            r2_fr_tst_list.append(r2_score(y_real, fr_sim))
            rmse_fr_tst_list.append(np.sqrt(mean_squared_error(y_real, fr_sim)))
            
        r2_fr_tst_mean = np.mean(r2_fr_tst_list)
        rmse_fr_tst_mean = np.mean(rmse_fr_tst_list)
        r2_fr_tra_mean = np.mean(r2_fr_tra_list)
        rmse_fr_tra_mean = np.mean(rmse_fr_tra_list)

        if r2_fr_tst_mean > best_r2_tst:
            best_r2_tst = r2_fr_tst_mean
            best_th = th_t
            best_config = (na_t, nb_t)

        resultados.append({
            'na': na_t, 'nb': nb_t,
            'R2_OSA_TRA':   r2_osa_tra,
            'RMSE_OSA_TRA': rmse_osa_tra,
            'R2_FR_TRA':    r2_fr_tra_mean,
            'RMSE_FR_TRA':  rmse_fr_tra_mean,
            'R2_FR_TST':    r2_fr_tst_mean,
            'RMSE_FR_TST':  rmse_fr_tst_mean,
        })
        
        print(f"[{idx:02d}/{len(ordens_teste):02d}] na={na_t:2d}, nb={nb_t:2d} | R2_FR_TST={r2_fr_tst_mean:+.4f} | RMSE_FR_TST={rmse_fr_tst_mean:.4f}")

    elapsed = time.time() - t_start
    print(f"\nVarredura finalizada em {elapsed:.2f} segundos!")

    # ==============================================================================
    # 5. EXPORTAÇÃO DOS RESULTADOS
    # ==============================================================================
    print("\n[4/4] Exportando resultados...")
    df_res = pd.DataFrame(resultados)

    # 1. Exporta CSV consolidado
    csv_path = os.path.join(OUT_DIR, "resultados_busca_na_nb_bls.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"  -> Tabela completa salva em: {csv_path}")

    # 2. Exporta parâmetros do melhor modelo em JSON
    best = df_res.loc[df_res['R2_FR_TST'].idxmax()]
    best_json_path = os.path.join(OUT_DIR, "melhor_modelo_bls.json")
    best_data = {
        "na": int(best.na),
        "nb": int(best.nb),
        "sampling_rate_hz": 100,
        "Ts_seconds": 0.01,
        "scaler_u_range": [-100.0, 100.0],
        "scaler_y_range": [-180.0, 180.0],
        "metrics": {
            "R2_FR_TST": float(best.R2_FR_TST),
            "RMSE_FR_TST": float(best.RMSE_FR_TST),
            "R2_FR_TRA": float(best.R2_FR_TRA),
            "RMSE_FR_TRA": float(best.RMSE_FR_TRA),
            "R2_OSA_TRA": float(best.R2_OSA_TRA),
            "RMSE_OSA_TRA": float(best.RMSE_OSA_TRA),
        },
        "a_coeffs": [float(x) for x in best_th[:int(best.na)]],
        "b_coeffs": [float(x) for x in best_th[int(best.na):]],
        "th_hat": [float(x) for x in best_th]
    }
    with open(best_json_path, 'w', encoding='utf-8') as f:
        json.dump(best_data, f, indent=2)
    print(f"  -> Melhor modelo (na={int(best.na)}, nb={int(best.nb)}) salvo em: {best_json_path}")

    # 3. Gráficos de Publicação
    if not args.grid:
        labels = [f"({int(r.na)},{int(r.nb)})" for r in df_res.itertuples()]
        fig, axes = plt.subplots(2, 1, figsize=(16, 9))
        fig.suptitle('Varredura de Ordens ARX / BLS — Aeropêndulo (100 Hz)', fontsize=14, fontweight='bold')

        x = np.arange(len(labels))
        width = 0.35

        # R2
        axes[0].bar(x - width/2, df_res['R2_FR_TRA'], width, label='Treinamento (F_Mix)', color='#1f77b4', alpha=0.8)
        axes[0].bar(x + width/2, df_res['R2_FR_TST'], width, label='Validação (Cross-Session)', color='#ff7f0e', alpha=0.8)
        axes[0].set_title('Coeficiente de Determinação Médio em Simulação Livre (R² Free-Run)')
        axes[0].set_ylabel('R²')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=60, fontsize=8)
        axes[0].set_ylim(-1.0, 1.05)
        axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[0].grid(axis='y', alpha=0.4, linestyle=':')
        axes[0].legend(loc='lower right')

        # RMSE
        axes[1].bar(x - width/2, df_res['RMSE_FR_TRA'], width, label='Treinamento (F_Mix)', color='#1f77b4', alpha=0.8)
        axes[1].bar(x + width/2, df_res['RMSE_FR_TST'], width, label='Validação (Cross-Session)', color='#ff7f0e', alpha=0.8)
        axes[1].set_title('Erro Médio Quadrático em Simulação Livre (RMSE Free-Run Normalizado)')
        axes[1].set_ylabel('RMSE')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=60, fontsize=8)
        axes[1].set_ylim(0, 0.40)
        axes[1].grid(axis='y', alpha=0.4, linestyle=':')
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        png_path = os.path.join(OUT_DIR, "comparacao_ordens_bls.png")
        pdf_path = os.path.join(OUT_DIR, "comparacao_ordens_bls.pdf")
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path)
        plt.close()
        print(f"  -> Gráficos salvos em: {png_path} e {pdf_path}")

    print("\n" + "=" * 78)
    print("RESUMO DO MELHOR MODELO ENCONTRADO")
    print("=" * 78)
    print(f"Melhor Ordem: na = {int(best.na)}, nb = {int(best.nb)}")
    print(f"Validação (Cross-Session) -> R² Free-Run = {best.R2_FR_TST:.4f} | RMSE Free-Run = {best.RMSE_FR_TST:.4f}")
    print(f"Treinamento (F_Mix)        -> R² Free-Run = {best.R2_FR_TRA:.4f} | RMSE Free-Run = {best.RMSE_FR_TRA:.4f}")
    print(f"One-Step-Ahead             -> R² = {best.R2_OSA_TRA:.4f} | RMSE = {best.RMSE_OSA_TRA:.4f}")
    print("=" * 78)

if __name__ == "__main__":
    main()

