import os
import sys
import time
import re
import numpy as np

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from aerodata import readData
from sysid import NARX

# ==============================================================================
# DATA LOADING
# ==============================================================================
train_files = {
    'multisine_45_train': 'RODADA-7/multi-seno-45-050Hz_0904_20-29.csv',
    'chirp_45_train': 'RODADA-7/chirp-45-amp45_0904_20-15.csv',
    'steps_45_train': 'RODADA-7/aprbs-45-1_0904_19-54.csv',
    'multisine_60_train': 'RODADA-7/multi-seno-60-050Hz_0904_20-38.csv',
    'chirp_60_train': 'RODADA-7/chirp-60-amp60_0904_20-20.csv',
    'steps_60_train': 'RODADA-7/aprbs-60-1_0904_20-04.csv'
}

val_files = {
    'multisine_45_val': 'RODADA-7/multi-seno-45-030Hz_0904_20-23.csv',
    'chirp_45_val': 'RODADA-7/chirp-45-amp35_0904_20-13.csv',
    'steps_45_val': 'RODADA-7/seq-degraus-45-2_0904_20-45.csv',
    'multisine_60_val': 'RODADA-7/multi-seno-60-030Hz_0904_20-32.csv',
    'chirp_60_val': 'RODADA-7/chirp-60-amp40_0904_20-16.csv',
    'steps_60_val': 'RODADA-7/seq-degraus-60-2_0904_20-54.csv'
}

def load_processed(fname, trim_start=10.0, trim_end=13.0):
    y, u, t, ref = readData(dataset_name=fname, return_ref=True)
    idx_start = np.searchsorted(t, t[0] + trim_start)
    idx_end = np.searchsorted(t, t[-1] - trim_end)
    if idx_start >= idx_end:
        idx_start, idx_end = 0, len(t)
    return u[idx_start:idx_end], y[idx_start:idx_end], t[idx_start:idx_end] - t[idx_start], (ref[idx_start:idx_end] if len(ref)>0 else np.full(idx_end-idx_start, np.nan))

print("Carregando 6 datasets de treino e 6 de validacao...", flush=True)
train_datasets = {k: load_processed(v) for k, v in train_files.items()}
val_datasets = {k: load_processed(v) for k, v in val_files.items()}
train_data_tuples = [(d[0], d[1]) for d in train_datasets.values()]

# Fast simulator
def compile_fast_evaluator(colnames, theta):
    parsed = []
    for term, th in zip(colnames, theta):
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

def simulate_freerun(colnames, theta, max_lag, u_in, y_init, bounds=None):
    N = len(u_in)
    y_hat = np.zeros(N)
    y_hat[:max_lag] = y_init[:max_lag]
    parsed_rules = compile_fast_evaluator(colnames, theta)
    
    for k in range(max_lag, N):
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
        if np.isnan(val) or np.isinf(val) or abs(val) > 1e4:
            y_hat[k:] = np.nan
            break
        y_hat[k] = val
        
    return y_hat[max_lag:]

def compute_osa(colnames, theta, max_lag, u_in, y_in):
    N = len(u_in)
    parsed_rules = compile_fast_evaluator(colnames, theta)
    y_hat = np.zeros(N - max_lag)
    for i, k in enumerate(range(max_lag, N)):
        val = 0.0
        for rule in parsed_rules:
            rtype = rule[0]
            th = rule[1]
            if rtype == 'const':
                val += th
            elif rtype == 'linear':
                val += th * (y_in[k - rule[3]] if rule[2] == 'y' else u_in[k - rule[3]])
            elif rtype == 'quad':
                v1 = y_in[k - rule[3]] if rule[2] == 'y' else u_in[k - rule[3]]
                v2 = y_in[k - rule[5]] if rule[4] == 'y' else u_in[k - rule[5]]
                val += th * (v1 * v2)
        y_hat[i] = val
    return y_hat

def compute_metrics(y_true, y_pred):
    if np.any(np.isnan(y_pred)):
        return -999.0, 999.0
    err = y_true - y_pred
    denom = np.linalg.norm(y_true - np.mean(y_true))
    rmse = float(np.sqrt(np.mean(err**2)))
    fit = float(100.0 * (1.0 - np.linalg.norm(err) / denom)) if denom > 1e-6 else 0.0
    return fit, rmse

# ==============================================================================
# TEST 1: ERR ACCUMULATED & FIT SENSITIVITY TO NUMBER OF TERMS
# ==============================================================================
print("\n" + "="*80, flush=True)
print("TESTE 1: CURVA DE ERR ACUMULADO E SENSIBILIDADE DO FIT AO NUMERO DE TERMOS (1 a 30)", flush=True)
print("="*80, flush=True)

t0 = time.time()
narx_30 = NARX(nu=15, ny=15, poly_order_l=2, n_components=30)
narx_30.fit(train_data_tuples)
print(f"Treinamento de 30 componentes concluido em {time.time()-t0:.2f} s.", flush=True)

err_vals = narx_30.fit_results_['ERR_values']
selected_terms = narx_30.selected_P_colnames_
cum_err = np.cumsum(err_vals)
A_mat = narx_30.fit_results_['A']
g_vec = narx_30.fit_results_['g']

print("\n--- Sequencia Completa dos 30 Termos Selecionados e ERR ---", flush=True)
print(f"{'#':<3} | {'Regressor':<20} | {'ERR individual':<16} | {'ERR Acumulado (%)':<20} | {'1 - ERR Acumulado':<18}", flush=True)
print("-" * 85, flush=True)
for i in range(30):
    print(f"{i+1:<3} | {selected_terms[i]:<20} | {err_vals[i]:<16.4e} | {cum_err[i]*100:<20.6f} | {1.0 - cum_err[i]:<18.4e}", flush=True)

# Performance on validation set as a function of k
print("\n--- Sensibilidade do FIT Livre (Free-Run) vs Numero de Termos ---", flush=True)
print(f"{'#':<3} | {'ERR Acum (%)':<14} | {'FIT OSA':<9} | {'FIT FR (puro)':<15} | {'FIT FR (clip)':<15} | {'Status FR Puro':<15}", flush=True)
print("-" * 75, flush=True)

term_steps = list(range(1, 16)) + [18, 20, 22, 25, 28, 30]

results_k = []
for k in term_steps:
    th_k = np.linalg.solve(A_mat[:k, :k], g_vec[:k])
    terms_k = selected_terms[:k]
    
    fits_osa = []
    fits_fr_pure = []
    fits_fr_clip = []
    diverged = False
    
    for vname, vd in val_datasets.items():
        u_v, y_v, t_v, _ = vd
        ml = 15
        y_target = y_v[ml:]
        
        # 1-step ahead
        y_osa = compute_osa(terms_k, th_k, ml, u_v, y_v)
        f_osa, _ = compute_metrics(y_target, y_osa)
        fits_osa.append(f_osa)
        
        # Free-run pure (unbounded)
        y_fr_p = simulate_freerun(terms_k, th_k, ml, u_v, y_v, bounds=None)
        f_fr_p, _ = compute_metrics(y_target, y_fr_p)
        if f_fr_p < -100 or np.isnan(f_fr_p):
            diverged = True
        fits_fr_pure.append(f_fr_p)
        
        # Free-run with physical bounds [-10, 135]
        y_fr_c = simulate_freerun(terms_k, th_k, ml, u_v, y_v, bounds=(-10.0, 135.0))
        f_fr_c, _ = compute_metrics(y_target, y_fr_c)
        fits_fr_clip.append(f_fr_c)
        
    avg_osa = np.mean(fits_osa)
    avg_fr_p = np.mean(fits_fr_pure) if not diverged else -999.0
    avg_fr_c = np.mean(fits_fr_clip)
    status_str = "ESTAVEL" if not diverged else "DIVERGIU/EXPLODIU"
    
    fr_p_str = f"{avg_fr_p:.2f}%" if not diverged else "DIVERGIU"
    print(f"{k:<3} | {cum_err[k-1]*100:<14.6f} | {avg_osa:<9.2f}% | {fr_p_str:<15} | {avg_fr_c:<15.2f}% | {status_str:<15}", flush=True)
    results_k.append((k, cum_err[k-1], avg_osa, avg_fr_p, avg_fr_c, diverged))

# ==============================================================================
# TEST 2: BILLINGS RESIDUAL TESTS
# ==============================================================================
print("\n" + "="*80, flush=True)
print("TESTE 2: TESTES DE RESIDUOS DE BILLINGS (AUTOCORRELACAO E CORRELACAO CRUZADA)", flush=True)
print("="*80, flush=True)

def compute_billings_tests(u, y, y_hat, max_lag=30):
    e = y - y_hat
    N = len(e)
    e_bar = np.mean(e)
    e_cent = e - e_bar
    var_e = np.sum(e_cent**2)
    
    u_bar = np.mean(u)
    u_cent = u - u_bar
    var_u = np.sum(u_cent**2)
    
    # phi_ee (autocorr)
    phi_ee = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            phi_ee[0] = 1.0
        else:
            phi_ee[lag] = np.sum(e_cent[lag:] * e_cent[:-lag]) / var_e
            
    # phi_ue (crosscorr u and e)
    phi_ue = np.zeros(2 * max_lag + 1)
    lags_ue = np.arange(-max_lag, max_lag + 1)
    denom_ue = np.sqrt(var_u * var_e)
    for i, lag in enumerate(lags_ue):
        if lag >= 0:
            if lag == 0:
                phi_ue[i] = np.sum(u_cent * e_cent) / denom_ue
            else:
                phi_ue[i] = np.sum(u_cent[:-lag] * e_cent[lag:]) / denom_ue
        else:
            pos_lag = abs(lag)
            phi_ue[i] = np.sum(u_cent[pos_lag:] * e_cent[:-pos_lag]) / denom_ue

    # phi_u2e (crosscorr u^2 and e)
    u2 = u**2
    u2_bar = np.mean(u2)
    u2_cent = u2 - u2_bar
    var_u2 = np.sum(u2_cent**2)
    denom_u2e = np.sqrt(var_u2 * var_e)
    phi_u2e = np.zeros(2 * max_lag + 1)
    for i, lag in enumerate(lags_ue):
        if lag >= 0:
            if lag == 0:
                phi_u2e[i] = np.sum(u2_cent * e_cent) / denom_u2e
            else:
                phi_u2e[i] = np.sum(u2_cent[:-lag] * e_cent[lag:]) / denom_u2e
        else:
            pos_lag = abs(lag)
            phi_u2e[i] = np.sum(u2_cent[pos_lag:] * e_cent[:-pos_lag]) / denom_u2e

    conf_bound = 1.96 / np.sqrt(N)
    return phi_ee, phi_ue, phi_u2e, conf_bound, N

print("\nAvaliando testes de residuos para modelos com diferentes numeros de termos (OSA na validacao):", flush=True)
for k_terms in [5, 10, 15, 20]:
    th_k = np.linalg.solve(A_mat[:k_terms, :k_terms], g_vec[:k_terms])
    terms_k = selected_terms[:k_terms]
    
    # Concatenate all validation residuals
    all_u = []
    all_y = []
    all_y_hat = []
    for vname, vd in val_datasets.items():
        u_v, y_v, t_v, _ = vd
        ml = 15
        y_osa = compute_osa(terms_k, th_k, ml, u_v, y_v)
        all_u.append(u_v[ml:])
        all_y.append(y_v[ml:])
        all_y_hat.append(y_osa)
        
    u_concat = np.concatenate(all_u)
    y_concat = np.concatenate(all_y)
    y_hat_concat = np.concatenate(all_y_hat)
    
    p_ee, p_ue, p_u2e, conf, N = compute_billings_tests(u_concat, y_concat, y_hat_concat, max_lag=30)
    
    ee_viol = np.sum(np.abs(p_ee[1:]) > conf)
    ue_viol = np.sum(np.abs(p_ue) > conf)
    u2e_viol = np.sum(np.abs(p_u2e) > conf)
    
    print(f"\nModelo com {k_terms} termos (N total validacao = {N}, banda 95% = ±{conf:.4f}):", flush=True)
    print(f"  phi_ee  (autocorrelacao de e, lags 1..30): {ee_viol:2d}/30 fora | max |phi_ee|: {np.max(np.abs(p_ee[1:])):.4f} | lag1: {p_ee[1]:.4f}, lag2: {p_ee[2]:.4f}", flush=True)
    print(f"  phi_ue  (correlacao cruzada u-e, lags -30..30): {ue_viol:2d}/61 fora | max |phi_ue|: {np.max(np.abs(p_ue)):.4f}", flush=True)
    print(f"  phi_u2e (correlacao nao-linear u^2-e): {u2e_viol:2d}/61 fora | max |phi_u2e|: {np.max(np.abs(p_u2e)):.4f}", flush=True)

# ==============================================================================
# TEST 3: LAG CONSTRAINTS (ny = nu = 15 vs 20 vs 25)
# ==============================================================================
print("\n" + "="*80, flush=True)
print("TESTE 3: HIPOTESE DA JANELA DE ATRASOS (ny=nu=15 vs 20 vs 25)", flush=True)
print("="*80, flush=True)

lag_cases = [15, 20, 25]
lag_models = {}

for lag in lag_cases:
    print(f"\nAjustando modelo com ny = nu = {lag} (n_components = 15)...", flush=True)
    t0 = time.time()
    m_lag = NARX(nu=lag, ny=lag, poly_order_l=2, n_components=15)
    m_lag.fit(train_data_tuples)
    dt = time.time() - t0
    
    cand_count = len(m_lag.P_candidate_colnames_)
    terms = list(m_lag.selected_P_colnames_)
    errs = m_lag.fit_results_['ERR_values']
    A_l = m_lag.fit_results_['A']
    g_l = m_lag.fit_results_['g']
    
    # Evaluate free-run on validation for 10 terms and 15 terms
    fits_10 = []
    fits_15 = []
    
    th_10 = np.linalg.solve(A_l[:10, :10], g_l[:10])
    th_15 = np.linalg.solve(A_l[:15, :15], g_l[:15])
    
    for vname, vd in val_datasets.items():
        u_v, y_v, t_v, _ = vd
        y_fr_10 = simulate_freerun(terms[:10], th_10, lag, u_v, y_v, bounds=(-10.0, 135.0))
        y_fr_15 = simulate_freerun(terms[:15], th_15, lag, u_v, y_v, bounds=(-10.0, 135.0))
        f10, _ = compute_metrics(y_v[lag:], y_fr_10)
        f15, _ = compute_metrics(y_v[lag:], y_fr_15)
        fits_10.append(f10)
        fits_15.append(f15)
        
    lag_models[lag] = {
        'cand_count': cand_count,
        'terms': terms,
        'errs': errs,
        'time': dt,
        'fit_10': np.mean(fits_10),
        'fit_15': np.mean(fits_15)
    }
    
    print(f"  Candidatos: {cand_count} monomios | Tempo: {dt:.2f} s", flush=True)
    print(f"  ERR acumulado (10 termos): {np.sum(errs[:10])*100:.6f}% | FIT FR Val (10): {np.mean(fits_10):.2f}%", flush=True)
    print(f"  ERR acumulado (15 termos): {np.sum(errs[:15])*100:.6f}% | FIT FR Val (15): {np.mean(fits_15):.2f}%", flush=True)
    
    # Check terms beyond lag 15
    beyond_15 = []
    for i, t in enumerate(terms):
        matches = re.findall(r'([yu])\(k-(\d+)\)', t)
        for var, l_s in matches:
            if int(l_s) > 15:
                beyond_15.append((i+1, t, errs[i]))
    if beyond_15:
        print(f"  --> TERMOS COM ATRASO > 15 SELECIONADOS ({len(beyond_15)}):", flush=True)
        for rank, t_name, e_val in beyond_15:
            print(f"       Rank #{rank}: {t_name:<20} (ERR = {e_val:.3e})", flush=True)
    else:
        print("  --> Nenhum termo com atraso > 15 foi selecionado!", flush=True)

print("\n" + "="*80, flush=True)
print("COMPARACAO DOS 10 PRIMEIROS REGRESSORES ENTRE ny=nu=15, 20 e 25", flush=True)
print("="*80, flush=True)
print(f"{'#':<3} | {'ny=nu=15':<22} | {'ny=nu=20':<22} | {'ny=nu=25':<22}", flush=True)
print("-" * 75, flush=True)
for i in range(10):
    t15 = lag_models[15]['terms'][i]
    t20 = lag_models[20]['terms'][i]
    t25 = lag_models[25]['terms'][i]
    print(f"{i+1:<3} | {t15:<22} | {t20:<22} | {t25:<22}", flush=True)

print("\nTESTES CONCLUIDOS COM SUCESSO!", flush=True)

