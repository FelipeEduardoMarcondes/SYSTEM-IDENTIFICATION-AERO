import os
import sys
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from aerodata import readData
from sysid import NARX

# ==============================================================================
# 1. ANÁLISE DO GANHO ESTÁTICO DO MODELO IDENTIFICADO (100 Hz)
# ==============================================================================
print("="*80)
print("PARTE 1: ANÁLISE DO GANHO DC E SENSIBILIDADE DO POLO EM z=1 (100 Hz)")
print("="*80)

# Carrega os dados de treino originais
train_files = [
    'RODADA-7/multi-seno-45-050Hz_0904_20-29.csv',
    'RODADA-7/chirp-45-amp45_0904_20-15.csv',
    'RODADA-7/aprbs-45-1_0904_19-54.csv',
    'RODADA-7/multi-seno-60-050Hz_0904_20-38.csv',
    'RODADA-7/chirp-60-amp60_0904_20-20.csv',
    'RODADA-7/aprbs-60-1_0904_20-04.csv'
]

def load_data(decimation=1):
    train_data = []
    for f in train_files:
        y, u, t, _ = readData(f, return_ref=True)
        idx_start = np.searchsorted(t, t[0] + 10.0)
        idx_end = np.searchsorted(t, t[-1] - 13.0)
        sl = slice(idx_start, idx_end, decimation)
        train_data.append((u[sl], y[sl]))
    return train_data

# Treina modelo original a 100 Hz com precisão total (float64)
m100 = NARX(nu=15, ny=15, poly_order_l=2, n_components=10)
m100.fit(load_data(decimation=1))

terms = list(m100.selected_P_colnames_)
th_exact = np.array(m100.theta_, dtype=np.float64)

print("\n--- Coeficientes com Precisão Total (float64) vs Arredondamento (4 casas) ---")
for i, (term, th) in enumerate(zip(terms, th_exact)):
    print(f"Termo {i+1:2d}: {term:<20} | float64: {th:+.10e} | Tabela (4 casas): {th:+.4f}")

# Coeficientes autorregressivos lineares
ar_indices = [i for i, t in enumerate(terms) if t.startswith('y') and not t.endswith('y(k-1)') and not '^2' in t and not 'u' in t]
ar_sum_exact = sum(th_exact[i] for i in ar_indices)

# Termo quadrático y(k-1)^2
quad_idx = [i for i, t in enumerate(terms) if 'y(k-1)y(k-1)' in t or 'y(k-1)^2' in t]
c_quad_exact = th_exact[quad_idx[0]] if quad_idx else 0.0

print(f"\nSoma dos coeficientes AR lineares: {ar_sum_exact:.8f}")
print(f"Coeficiente quadrático y(k-1)^2: {c_quad_exact:.8e}")

for y_op in [45.0, 60.0]:
    # Derivada de c_quad * y^2 em y_op: 2 * c_quad * y_op
    c_lin_equiv = 2.0 * c_quad_exact * y_op
    sum_eff = ar_sum_exact + c_lin_equiv
    denom = 1.0 - sum_eff
    print(f"\nEm {y_op}°:")
    print(f"  Linearização do termo quadrático: 2 * ({c_quad_exact:.4e}) * {y_op} = {c_lin_equiv:+.8e}")
    print(f"  Soma efetiva dos coeficientes AR: {sum_eff:.8f}")
    print(f"  Denominador do ganho DC (1 - sum): {denom:+.8e}")
    
    # Sensibilidade: o que acontece se um coeficiente AR mudar em apenas 1e-4?
    denom_perturbed = denom + 1e-4
    delta_gain_pct = ((denom / denom_perturbed) - 1.0) * 100.0
    print(f"  Variação no ganho DC para erro de apenas 1e-4 (0.0001) nos coeficientes: {delta_gain_pct:+.2f}%!")

# ==============================================================================
# 2. CÁLCULO DA CURVA ESTÁTICA u_ss vs y_ss E ERRO DE TRUNCAMENTO
# ==============================================================================
print("\n" + "="*80)
print("PARTE 2: CURVA DE EQUILÍBRIO ESTÁTICO (u_ss -> y_ss): FLOAT64 vs TABELA (4 CASAS)")
print("="*80)

# Na equação estática:
# y = (sum_ar) * y + c_quad * y^2 + (sum_b_uu) * u^2 + const
# c_quad * y^2 + (sum_ar - 1) * y + (sum_b_uu * u^2 + const) = 0

b_indices = [i for i, t in enumerate(terms) if 'u' in t and not 'y' in t]
sum_b_exact = sum(th_exact[i] for i in b_indices)
const_idx = [i for i, t in enumerate(terms) if t == 'constant']
const_exact = th_exact[const_idx[0]] if const_idx else 0.0

# Arredondados para 4 casas como na Tabela 2 do artigo
th_table = np.array([
    +1.4230, -0.2120, -0.1568, -0.0944, +0.0388,
    +6.90e-6, -9.30e-7, +1.77e-5, +9.30e-6, -0.0059
])
ar_sum_table = sum(th_table[:5])
c_quad_table = th_table[5]
sum_b_table = sum(th_table[6:9])
const_table = th_table[9]

def solve_yss(u_val, ar_s, c_q, b_s, c_0):
    # c_q * y^2 + (ar_s - 1) * y + (b_s * u^2 + c_0) = 0
    A = c_q
    B = ar_s - 1.0
    C = b_s * (u_val**2) + c_0
    delta = B**2 - 4 * A * C
    if delta < 0:
        return np.nan
    # Queremos a raiz positiva mais próxima do regime
    y1 = (-B - np.sqrt(delta)) / (2 * A)
    y2 = (-B + np.sqrt(delta)) / (2 * A)
    # Seleciona a raiz física (entre 0 e 120)
    for y in [y1, y2]:
        if 0 <= y <= 120:
            return y
    return y1

print(f"{'Controle u [%]':<15} | {'y_ss (float64) [°]':<20} | {'y_ss (Tabela 4 casas) [°]':<25} | {'Diferença (Viés) [°]':<20}")
print("-" * 85)
for u_test in [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]:
    y_exact = solve_yss(u_test, ar_sum_exact, c_quad_exact, sum_b_exact, const_exact)
    y_tab = solve_yss(u_test, ar_sum_table, c_quad_table, sum_b_table, const_table)
    diff = y_tab - y_exact
    print(f"{u_test:<15.1f} | {y_exact:<20.2f} | {y_tab:<25.2f} | {diff:<+20.2f}")

# ==============================================================================
# 3. DECIMAÇÃO E SUPERAMOSTRAGEM (100 Hz vs 20 Hz vs 10 Hz)
# ==============================================================================
print("\n" + "="*80)
print("PARTE 3: DECIMAÇÃO E SUPERAMOSTRAGEM (100 Hz vs 20 Hz vs 10 Hz)")
print("="*80)

# Decimação 1 = 100 Hz (Ts = 10 ms)
# Decimação 5 = 20 Hz (Ts = 50 ms)
# Decimação 10 = 10 Hz (Ts = 100 ms)

decimations = [1, 5, 10]
for dec in decimations:
    fs = 100 // dec
    ts = dec * 0.010
    # Ajusta ny e nu para cobrir horizonte similar (~150 ms)
    # Em 100 Hz: ny = 15 (150 ms)
    # Em 20 Hz: ny = 3 (150 ms) ou ny = 5 (250 ms)
    # Em 10 Hz: ny = 2 (200 ms)
    ny_d = max(3, 15 // dec)
    nu_d = max(3, 15 // dec)
    
    t_data = load_data(decimation=dec)
    m_dec = NARX(nu=nu_d, ny=ny_d, poly_order_l=2, n_components=10)
    m_dec.fit(t_data)
    
    dec_terms = list(m_dec.selected_P_colnames_)
    dec_th = np.array(m_dec.theta_, dtype=np.float64)
    
    # Soma AR
    ar_idx = [i for i, t in enumerate(dec_terms) if t.startswith('y') and not t.endswith('y(k-1)') and not '^2' in t and not 'u' in t]
    ar_s = sum(dec_th[i] for i in ar_idx)
    
    denom_dc = 1.0 - ar_s
    
    print(f"\n--- Frequência de Amostragem: {fs} Hz (Ts = {ts*1000:.0f} ms, decimação={dec}) ---")
    print(f"  Ordens de atraso: ny={ny_d}, nu={nu_d} (cobre {ny_d*ts*1000:.0f} ms)")
    print(f"  Candidatos: {len(m_dec.P_candidate_colnames_)}")
    print(f"  Soma dos coeficientes AR lineares: {ar_s:.6f}")
    print(f"  Denominador DC (1 - sum(a_i)): {denom_dc:.6f}  (em 100 Hz era {1.0-ar_sum_exact:.6f})")
    print(f"  Termos selecionados:")
    for i, (t_name, th_v) in enumerate(zip(dec_terms, dec_th)):
        print(f"    {i+1:2d}: {t_name:<18} = {th_v:+.6e}")

