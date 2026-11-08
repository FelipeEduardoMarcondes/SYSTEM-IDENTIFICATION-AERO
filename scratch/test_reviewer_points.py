import os
import sys
import re
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from aerodata import readData
from sysid import NARX

# ==============================================================================
# CARREGA OS 6 DATASETS DE VALIDAÇÃO DO PIPELINE ORIGINAL
# ==============================================================================
val_files = {
    'multisine_45_val': ('RODADA-7/multi-seno-45-030Hz_0904_20-23.csv', 'Multisseno 45° (0.30 Hz)'),
    'chirp_45_val':     ('RODADA-7/chirp-45-amp35_0904_20-13.csv',       'Chirp 45° (±35°)'),
    'steps_45_val':     ('RODADA-7/seq-degraus-45-2_0904_20-45.csv',      'Degraus 45° (25°-65°)'),
    'multisine_60_val': ('RODADA-7/multi-seno-60-030Hz_0904_20-32.csv', 'Multisseno 60° (0.30 Hz)'),
    'chirp_60_val':     ('RODADA-7/chirp-60-amp40_0904_20-16.csv',       'Chirp 60° (±40°)'),
    'steps_60_val':     ('RODADA-7/seq-degraus-60-2_0904_20-54.csv',      'Degraus 60° (40°-80°)')
}

def load_processed(fname, trim_start=10.0, trim_end=13.0):
    y, u, t, ref = readData(dataset_name=fname, return_ref=True)
    idx_start = np.searchsorted(t, t[0] + trim_start)
    idx_end = np.searchsorted(t, t[-1] - trim_end)
    if idx_start >= idx_end:
        idx_start, idx_end = 0, len(t)
    return u[idx_start:idx_end], y[idx_start:idx_end], t[idx_start:idx_end] - t[idx_start]

val_data = {k: load_processed(v[0]) for k, v in val_files.items()}

# Modelo Real do pipeline (IDENT-HELON / narx_testes.py)
terms_real = [
    "y(k-1)", "y(k-2)", "y(k-15)", "y(k-4)", "u(k-3)u(k-6)",
    "y(k-1)y(k-1)", "y(k-9)", "u(k-6)u(k-10)", "constant", "y(k-3)"
]
theta_real = np.array([
    1.4453516339161712,
    -0.15763384661289287,
    0.03186000306800809,
    -0.12955619806248944,
    5.441039764206217e-06,
    4.8057671941595836e-06,
    -0.08485785214159239,
    1.336143189451035e-05,
    -0.0053649645115774625,
    -0.10615592523256359
])

def compile_evaluator(colnames, theta):
    parsed = []
    for term, th in zip(colnames, theta):
        th_val = float(th)
        if term == 'constant': parsed.append(('c', th_val)); continue
        m = re.findall(r'([yu])\(k-(\d+)\)', term)
        if len(m)==1: parsed.append(('l', th_val, m[0][0], int(m[0][1])))
        elif len(m)==2: parsed.append(('q', th_val, m[0][0], int(m[0][1]), m[1][0], int(m[1][1])))
    return parsed

def simulate_trial(colnames, theta, u_in, y_in, max_lag=15, bounds=None):
    N = len(u_in)
    y_hat = np.zeros(N)
    y_hat[:max_lag] = y_in[:max_lag]
    parsed = compile_evaluator(colnames, theta)
    diverged = False
    for k in range(max_lag, N):
        v = 0.0
        for r in parsed:
            if r[0] == 'c': v += r[1]
            elif r[0] == 'l': v += r[1] * (y_hat[k-r[3]] if r[2]=='y' else u_in[k-r[3]])
            elif r[0] == 'q':
                v1 = y_hat[k-r[3]] if r[2]=='y' else u_in[k-r[3]]
                v2 = y_hat[k-r[5]] if r[4]=='y' else u_in[k-r[5]]
                v += r[1] * v1 * v2
        if bounds is not None:
            v = np.clip(v, bounds[0], bounds[1])
        if np.isnan(v) or np.isinf(v) or abs(v) > 1e4:
            diverged = True
            y_hat[k:] = np.nan
            break
        y_hat[k] = v
    return y_hat[max_lag:], diverged

def calc_metrics(y_true, y_pred):
    if np.any(np.isnan(y_pred)):
        return np.nan, np.nan, np.nan
    err = y_true - y_pred
    denom = np.linalg.norm(y_true - np.mean(y_true))
    rmse = np.sqrt(np.mean(err**2))
    fit = 100.0 * (1.0 - np.linalg.norm(err)/denom) if denom > 1e-6 else 0.0
    r2 = 1.0 - np.sum(err**2) / np.sum((y_true - np.mean(y_true))**2) if denom > 1e-6 else 0.0
    return fit, rmse, r2

# ==============================================================================
# TESTE 1: SIMULAÇÃO TRIAL POR TRIAL (SEM CLIP vs COM CLIP)
# ==============================================================================
print("="*85)
print("TESTE 1: SIMULAÇÃO TRIAL POR TRIAL DO MODELO REAL DE 10 TERMOS")
print("="*85)
print(f"{'Ensaio de Validação':<25} | {'N':<5} | {'FIT Sem Clip':<15} | {'FIT Com Clip':<15} | {'Estável Puro?':<12}")
print("-" * 85)

for k, (u_v, y_v, t_v) in val_data.items():
    label = val_files[k][1]
    y_tgt = y_v[15:]
    
    # Sem clip
    yh_raw, div_raw = simulate_trial(terms_real, theta_real, u_v, y_v, max_lag=15, bounds=None)
    fit_raw, rmse_raw, r2_raw = calc_metrics(y_tgt, yh_raw)
    
    # Com clip [-10, 135] (como em narx_testes.py)
    yh_clip, _ = simulate_trial(terms_real, theta_real, u_v, y_v, max_lag=15, bounds=(-10.0, 135.0))
    fit_clip, rmse_clip, r2_clip = calc_metrics(y_tgt, yh_clip)
    
    status = "SIM" if not div_raw else "EXPLODIU"
    raw_str = f"{fit_raw:6.2f}%" if not div_raw else "DIVERGIU"
    clip_str = f"{fit_clip:6.2f}%"
    print(f"{label:<25} | {len(y_tgt):<5} | {raw_str:<15} | {clip_str:<15} | {status:<12}")

# ==============================================================================
# TESTE 2: POLOS, AUTOVALORES E GANHO DC
# ==============================================================================
print("\n" + "="*85)
print("TESTE 2: AUTOVALORES DA LINEARIZAÇÃO E ESTABILIDADE LOCAL (45° e 60°)")
print("="*85)

# Polinômio característico AR: A(z) = 1 - sum(a_i z^-i)
# No regime linearizado em y0:
# y(k) = a1 y(k-1) + a2 y(k-2) + a3 y(k-3) + a4 y(k-4) + a9 y(k-9) + a15 y(k-15) + c_quad * y(k-1)^2
# Derivada de c_quad * y^2 em y0 = 2 * c_quad * y0
# Então o termo efetivo em y(k-1) é a1_eff = a1 + 2 * c_quad * y0.

for y0 in [0.0, 45.0, 60.0]:
    c_lin_eff = 2.0 * theta_real[5] * y0
    a_coeffs = np.zeros(16)
    a_coeffs[0] = 1.0 # z^0
    a_coeffs[1] = -(theta_real[0] + c_lin_eff) # z^-1
    a_coeffs[2] = -theta_real[1] # z^-2
    a_coeffs[3] = -theta_real[9] # z^-3
    a_coeffs[4] = -theta_real[3] # z^-4
    a_coeffs[9] = -theta_real[6] # z^-9
    a_coeffs[15] = -theta_real[2] # z^-15
    
    # Raízes de z^15 * A(z^-1)
    poles = np.roots(a_coeffs)
    mags = np.abs(poles)
    max_mag = np.max(mags)
    dom_idx = np.argmax(mags)
    dom_pole = poles[dom_idx]
    
    # Polos complexos conjugados dominantes
    complex_poles = [p for p in poles if abs(p.imag) > 1e-4]
    complex_poles.sort(key=lambda p: abs(p), reverse=True)
    p_osc = complex_poles[0] if complex_poles else dom_pole
    
    # Frequência natural e amortecimento discreto: z = exp( ( -zeta*wn +- j*wn*sqrt(1-zeta^2) ) * Ts )
    Ts = 0.010
    wn = np.abs(np.log(p_osc)) / Ts
    zeta = -np.real(np.log(p_osc)) / (wn * Ts)
    fn = wn / (2 * np.pi)
    
    sum_a = -np.sum(a_coeffs[1:])
    denom_dc = 1.0 - sum_a
    
    print(f"\nPonto de operação y0 = {y0:.1f}°:")
    print(f"  Soma dos coeficientes AR efetivos: {sum_a:.8f}")
    print(f"  Denominador DC (1 - sum): {denom_dc:+.8e}")
    print(f"  Maior módulo de polo |z|: {max_mag:.6f} ({'ESTÁVEL (|z|<1)' if max_mag < 1.0 else 'INSTÁVEL (|z|>=1)'})")
    print(f"  Polo oscilatório dominante: {p_osc.real:.5f} ± {abs(p_osc.imag):.5f}j (|z| = {abs(p_osc):.6f})")
    print(f"  Frequência natural identificada: fn = {fn:.3f} Hz | Fator de amortecimento: zeta = {zeta:.4f}")

# ==============================================================================
# TESTE 3: FORMULAÇÃO INCREMENTAL (Delta y) E DECIMAÇÃO (20 Hz)
# ==============================================================================
print("\n" + "="*85)
print("TESTE 3: FROLS SOBRE Delta y = y(k) - y(k-1) E FROLS A 20 Hz")
print("="*85)

train_files_list = [
    'RODADA-7/multi-seno-45-050Hz_0904_20-29.csv',
    'RODADA-7/chirp-45-amp45_0904_20-15.csv',
    'RODADA-7/aprbs-45-1_0904_19-54.csv',
    'RODADA-7/multi-seno-60-050Hz_0904_20-38.csv',
    'RODADA-7/chirp-60-amp60_0904_20-20.csv',
    'RODADA-7/aprbs-60-1_0904_20-04.csv'
]

train_raw = [load_processed(f) for f in train_files_list]

# Formulação em Delta y: delta_y(k) = y(k) - y(k-1)
# O modelo prevê delta_y a partir de lags passados de delta_y e u
train_delta = []
for u, y, t in train_raw:
    dy = np.diff(y, prepend=y[0])
    train_delta.append((u, dy))

m_delta = NARX(nu=15, ny=15, poly_order_l=2, n_components=10)
m_delta.fit(train_delta)

print("\nTermos selecionados no modelo incremental Delta y (100 Hz):")
err_dy = m_delta.fit_results_['ERR_values']
for i, (term, th, err_v) in enumerate(zip(m_delta.selected_P_colnames_, m_delta.theta_, err_dy)):
    print(f"  {i+1:2d}: {term:<20} = {th:+.6e} (ERR: {err_v:.4e})")

print(f"ERR acumulado do termo 1 no Delta y: {err_dy[0]*100:.2f}% (vs 99.994% no y bruto!)")

