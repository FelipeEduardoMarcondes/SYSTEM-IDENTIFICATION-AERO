import os
import sys
import time
import numpy as np

# Ensure root directory is in path
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

print("Carregando datasets de treino e validação...")
train_datasets = {k: load_processed(v) for k, v in train_files.items()}
val_datasets = {k: load_processed(v) for k, v in val_files.items()}

train_data_tuples = [(d[0], d[1]) for d in train_datasets.values()]

# Helper for simulation
import re
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

def simulate_freerun(model, u_in, y_init, bounds=None):
    ml = model._max_lag_internal_
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
        if np.isnan(val) or np.isinf(val) or abs(val) > 1e4:
            # Diverged
            y_hat[k:] = np.nan
            break
        y_hat[k] = val
        
    return y_hat[ml:]

def compute_fit(y_true, y_pred):
    if np.any(np.isnan(y_pred)):
        return -999.0, 999.0  # Diverged
    err = y_true - y_pred
    denom = np.linalg.norm(y_true - np.mean(y_true))
    rmse = float(np.sqrt(np.mean(err**2)))
    fit = float(100.0 * (1.0 - np.linalg.norm(err) / denom)) if denom > 1e-6 else 0.0
    return fit, rmse

# ==============================================================================
# TEST 1: ERR ACCUMULATED & FIT SENSITIVITY TO NUMBER OF TERMS
# ==============================================================================
print("\n" + "="*80)
print("TESTE 1: CURVA DE ERR ACUMULADO E SENSIBILIDADE DO FIT AO NÚMERO DE TERMOS")
print("="*80)

# We fit a 25-component model to get the sequence of ERR values
max_comp = 25
print(f"Treinando NARX com ny=15, nu=15, l=2 até {max_comp} componentes...")
t0 = time.time()
narx_25 = NARX(nu=15, ny=15, poly_order_l=2, n_components=max_comp)
narx_25.fit(train_data_tuples)
print(f"Treinamento concluído em {time.time()-t0:.2f} s.")

err_vals = narx_25.fit_results_['ERR_values']
selected_terms = narx_25.selected_P_colnames_
cum_err = np.cumsum(err_vals)

print("\n--- Sequência de Termos Selecionados e ERR ---")
print(f"{'#':<3} | {'Regressor':<20} | {'ERR':<12} | {'ERR Acumulado':<15} | {'1 - ERR Acumulado':<18}")
print("-" * 75)
for i in range(len(selected_terms)):
    print(f"{i+1:<3} | {selected_terms[i]:<20} | {err_vals[i]:<12.4e} | {cum_err[i]:<15.8f} | {1.0 - cum_err[i]:<18.4e}")

# Now test validation performance for various component counts
component_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25]
print("\n--- Sensibilidade do FIT Livre (Free-Run) vs Número de Termos ---")
print(f"{'# Termos':<9} | {'ERR Acum (%)':<14} | {'FIT Médio FR':<14} | {'RMSE Médio FR':<14} | {'Divergiu?':<10}")
print("-" * 70)

for n_c in component_steps:
    m = NARX(nu=15, ny=15, poly_order_l=2, n_components=n_c)
    m.fit(train_data_tuples)
    
    fits = []
    rmses = []
    diverged = False
    for k, vd in val_datasets.items():
        u_v, y_v, t_v, _ = vd
        ml = m._max_lag_internal_
        # Free-run WITHOUT clipping to test true stability
        y_fr = simulate_freerun(m, u_v, y_v, bounds=None)
        fit_val, rmse_val = compute_fit(y_v[ml:], y_fr)
        if fit_val < -100 or np.isnan(fit_val):
            diverged = True
        fits.append(fit_val)
        rmses.append(rmse_val)
        
    avg_fit = np.mean(fits) if not diverged else -999.0
    avg_rmse = np.mean(rmses) if not diverged else 999.0
    div_str = "SIM (Instável!)" if diverged else "Não (Estável)"
    cum_pct = cum_err[n_c-1] * 100
    print(f"{n_c:<9} | {cum_pct:<14.6f} | {avg_fit:<14.2f} | {avg_rmse:<14.2f} | {div_str:<10}")

# Also test with physical clipping [-10, 135]
print("\n--- Com saturação física [-10, 135] na simulação livre ---")
print(f"{'# Termos':<9} | {'FIT Médio FR (clip)':<20} | {'RMSE Médio FR (clip)':<20}")
print("-" * 55)
for n_c in component_steps:
    m = NARX(nu=15, ny=15, poly_order_l=2, n_components=n_c)
    m.fit(train_data_tuples)
    fits = []
    rmses = []
    for k, vd in val_datasets.items():
        u_v, y_v, t_v, _ = vd
        ml = m._max_lag_internal_
        y_fr = simulate_freerun(m, u_v, y_v, bounds=(-10.0, 135.0))
        fit_val, rmse_val = compute_fit(y_v[ml:], y_fr)
        fits.append(fit_val)
        rmses.append(rmse_val)
    print(f"{n_c:<9} | {np.mean(fits):<20.2f} | {np.mean(rmses):<20.2f}")

# ==============================================================================
# TEST 2: RESIDUAL ANALYSIS (BILLINGS CORRELATION TESTS)
# ==============================================================================
print("\n" + "="*80)
print("TESTE 2: TESTES DE RESÍDUOS (BILLINGS: AUTOCORRELAÇÃO E CORRELAÇÃO CRUZADA COM U)")
print("="*80)

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

    conf_bound = 1.96 / np.sqrt(N)
    return phi_ee, phi_ue, conf_bound, N

# Evaluate on 10-term model
m10 = NARX(nu=15, ny=15, poly_order_l=2, n_components=10)
m10.fit(train_data_tuples)

# One-step-ahead prediction on multisine validation
u_val, y_val, t_val, _ = val_datasets['multisine_45_val']
y_hat_osa, y_target_osa = m10.predict(u_val, y_val, mode='OSA')
u_osa = u_val[m10._max_lag_internal_:]

phi_ee, phi_ue, conf, N = compute_billings_tests(u_osa, y_target_osa, y_hat_osa, max_lag=30)

print(f"Número de amostras do ensaio: N = {N}")
print(f"Banda de confiança de 95% (±1.96 / sqrt(N)): ±{conf:.4f}")

# Check phi_ee
ee_violations = np.sum(np.abs(phi_ee[1:]) > conf)
print(f"phi_ee (autocorrelação de resíduos, lags 1 a 30): {ee_violations}/30 fora dos limites de 95%.")
print(f"  Valores iniciais de phi_ee: lag 1 = {phi_ee[1]:.4f}, lag 2 = {phi_ee[2]:.4f}, lag 3 = {phi_ee[3]:.4f}")

# Check phi_ue
ue_violations = np.sum(np.abs(phi_ue) > conf)
print(f"phi_ue (correlação cruzada u-e, lags -30 a 30): {ue_violations}/61 fora dos limites de 95%.")
print(f"  Max |phi_ue|: {np.max(np.abs(phi_ue)):.4f} (limite = {conf:.4f})")

# Compare with 15-term and 20-term models
for comp in [15, 20]:
    mc = NARX(nu=15, ny=15, poly_order_l=2, n_components=comp)
    mc.fit(train_data_tuples)
    y_hat_c, y_tgt_c = mc.predict(u_val, y_val, mode='OSA')
    p_ee, p_ue, _, _ = compute_billings_tests(u_osa, y_tgt_c, y_hat_c, max_lag=30)
    print(f"\nModelo com {comp} termos:")
    print(f"  phi_ee: {np.sum(np.abs(p_ee[1:]) > conf)}/30 fora")
    print(f"  phi_ue: {np.sum(np.abs(p_ue) > conf)}/61 fora (Max |phi_ue| = {np.max(np.abs(p_ue)):.4f})")

# ==============================================================================
# TEST 3: LAG CONSTRAINTS (ny = nu = 15 vs 20 vs 25)
# ==============================================================================
print("\n" + "="*80)
print("TESTE 3: EFEITO DE AMPLIAR A JANELA DE ATRASOS (ny=nu=15 vs 20 vs 25)")
print("="*80)

lag_cases = [15, 20, 25]
results_lags = {}

for lag in lag_cases:
    print(f"\nAvaliando com ny = nu = {lag}...")
    t0 = time.time()
    m_lag = NARX(nu=lag, ny=lag, poly_order_l=2, n_components=15)
    m_lag.fit(train_data_tuples)
    dt = time.time() - t0
    
    cand_count = len(m_lag.P_candidate_colnames_)
    terms = list(m_lag.selected_P_colnames_)
    errs = m_lag.fit_results_['ERR_values']
    
    # Check lags of selected terms
    max_y_lag = 0
    max_u_lag = 0
    terms_beyond_15 = []
    for term in terms:
        matches = re.findall(r'([yu])\(k-(\d+)\)', term)
        for var, l_str in matches:
            l_int = int(l_str)
            if var == 'y' and l_int > max_y_lag:
                max_y_lag = l_int
            if var == 'u' and l_int > max_u_lag:
                max_u_lag = l_int
            if l_int > 15:
                terms_beyond_15.append(term)
                
    # Evaluate free-run on validation
    fits = []
    for k, vd in val_datasets.items():
        u_v, y_v, t_v, _ = vd
        ml = m_lag._max_lag_internal_
        y_fr = simulate_freerun(m_lag, u_v, y_v, bounds=(-10.0, 135.0))
        fit_v, _ = compute_fit(y_v[ml:], y_fr)
        fits.append(fit_v)
        
    results_lags[lag] = {
        'cand_count': cand_count,
        'terms': terms,
        'errs': errs,
        'cum_err_10': np.sum(errs[:10]),
        'cum_err_15': np.sum(errs[:15]),
        'max_y_lag': max_y_lag,
        'max_u_lag': max_u_lag,
        'terms_beyond_15': list(set(terms_beyond_15)),
        'avg_val_fit': np.mean(fits),
        'fit_time': dt
    }
    
    print(f"  Candidatos: {cand_count} monômios")
    print(f"  Tempo de ajuste: {dt:.2f} s")
    print(f"  Maior atraso em y selecionado: k-{max_y_lag}")
    print(f"  Maior atraso em u selecionado: k-{max_u_lag}")
    print(f"  Termos com atraso > 15 ({len(terms_beyond_15)}): {set(terms_beyond_15)}")
    print(f"  ERR acumulado (10 termos): {results_lags[lag]['cum_err_10']*100:.6f}%")
    print(f"  ERR acumulado (15 termos): {results_lags[lag]['cum_err_15']*100:.6f}%")
    print(f"  FIT médio de validação (Free-Run): {results_lags[lag]['avg_val_fit']:.2f}%")

print("\n--- Comparação dos 10 primeiros termos entre ny=nu=15 e ny=nu=20 ---")
for i in range(10):
    t15 = results_lags[15]['terms'][i]
    t20 = results_lags[20]['terms'][i]
    e15 = results_lags[15]['errs'][i]
    e20 = results_lags[20]['errs'][i]
    print(f"Termo {i+1:2d} | 15: {t15:<20} (ERR: {e15:.3e}) | 20: {t20:<20} (ERR: {e20:.3e})")

print("\nFIM DOS TESTES.")

