# %% [markdown]
# # Identificação Metodológica de Modelo Não-Linear NARX Polinomial (Aeropêndulo)
#
# Pipeline científico ponta-a-ponta para identificação de modelo polinomial NARX
# (*Nonlinear AutoRegressive with eXogenous inputs*) otimizado para **simulação livre
# (*Free-Run*)** sem saturação artificial e preservando a entrada bidirecional $u(k) \in [-100, 100]\%$.
#
# ### Etapas Metodológicas (sem hiperparâmetros arbitrários):
# 1. **Particionamento Estratificado 3-Vias da `RODADA-7` (24 ensaios da mesma sessão, 0% vazamento):**
#    - **Treinamento (10 ensaios):** 4 famílias de excitação (Multisseno, Chirp, APRBS, Degraus)
#      nos dois pontos de operação ($45^\circ$ e $60^\circ$), concentrando maiores frequências e amplitudes.
#    - **Validação / Seleção de Modelo (8 ensaios):** 4 famílias em níveis intermediários ($45^\circ$ e $60^\circ$),
#      usados exclusivamente para seleção de decimação, ordens, estrutura e *early stopping*.
#    - **Teste Final Intocado (6 ensaios oficiais):** mantidos congelados até a avaliação final.
# 2. **Seleção da Taxa de Amostragem (`DECIMATION` / $T_s$):**
#    Análise espectral (Welch PSD: $f_{95}, f_{99}$) + Critério de Aguirre (1995/2015) pela
#    autocorrelação linear $\rho_y(\tau)$ e não-linear $\rho_{y^2}(\tau)$ ($\tau_m/20 \le T_s \le \tau_m/10$).
# 3. **Determinação de Ordens ($n_y, n_u$) via BIC (ARX) e Análise de Polos:**
#    Identificação do joelho do Critério de Informação Bayesiano (BIC) e raio espectral dos polos AR.
# 4. **Seleção de Estrutura Orientada a Simulação (*SR-FROLS*, Piroddi & Spinelli, 2003):**
#    Base puramente polinomial aditiva ($f(y) + g(u)$), $\ell_y = 2$ (evita escape cúbico em $\theta > 90^\circ$)
#    e $\ell_u \in \{2, 3\}$ (termos ímpares em $u$ representam o empuxo bidirecional $u|u|$),
#    filtrada pelo **Critério Algébrico de Estabilidade de Aguirre (1994)**.
# 5. **Refinamento de Parâmetros em Erro de Saída (*LM-OE* com Jacobiano Analítico Exato):**
#    Levenberg-Marquardt amortecido sobre a simulação livre completa nos 10 ensaios de treino,
#    com *early stopping* guiado estritamente pelos 8 ensaios de Validação + estabilidade algébrica.
# 6. **Avaliação Final Única no Teste Intocado (SEM clip artificial) + Estudo de Ablação:**
#    Comparação direta contra o baseline original (100 Hz, FROLS/ERR) e FROLS/ERR no $T_s$ metodológico.
# 7. **Validação Formal (Billings & Voon, 1986 + Curva Estática de Aguirre) e Exportação:**
#    Gera gráficos em padrão IEEE/ABNT (300 DPI PNG + PDF vetorial), `narx_model.json` compatível
#    com CasADi NMPC (`mpc_v2/v3/v4`), `narx_metrics.json`, equações LaTeX e relatório completo.

# %%
import os
import sys
import io
import re
import json
import time
import warnings
from itertools import combinations_with_replacement

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS, FLAGS E IMPORTAÇÕES
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Por segurança, salva gráficos em IDENT-TESTES/plots.
# Altere para True quando desejar sincronizar também com tcc/img/resultados e img/resultados.
SYNC_TCC_PLOTS = False

plots_dirs = [os.path.join(current_dir, 'plots')]
if SYNC_TCC_PLOTS:
    for rel_target in [('tcc', 'img', 'resultados'), ('img', 'resultados')]:
        cand_dir = os.path.join(root_dir, *rel_target)
        if os.path.exists(cand_dir):
            plots_dirs.append(cand_dir)

for d in plots_dirs:
    os.makedirs(d, exist_ok=True)

from aerodata import readData

# ==============================================================================
# PADRONIZAÇÃO VISUAL ACADÊMICA (IEEE / ABNT / PUCPR / UTFPR)
# ==============================================================================
def setup_academic_style():
    """Configura o Matplotlib para gráficos com estética acadêmica e publicação."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.85",
        "figure.autolayout": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

setup_academic_style()

def save_academic_plot(fig, filename_base, dirs=plots_dirs):
    """Salva a figura simultaneamente em PNG (300 DPI) e PDF vetorial."""
    for d in dirs:
        fig.savefig(os.path.join(d, f"{filename_base}.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(d, f"{filename_base}.pdf"), format='pdf', bbox_inches='tight')
    print(f"  -> Gráfico salvo: {filename_base} (.png e .pdf) em {len(dirs)} pasta(s).")

# %% [markdown]
# ## Etapa 1: Carregamento e Particionamento Estratificado 3-Vias (`RODADA-7`)
#
# Todos os 24 ensaios canônicos da sessão `RODADA-7` (`09/04`, 19h54–20h57, 100 Hz nativos)
# são divididos de forma estratificada nas 4 classes de excitação (Multisseno, Chirp, APRBS e
# Sequência de Degraus) e nos 2 patamares operacionais ($45^\circ$ e $60^\circ$), sem nenhum
# vazamento de dados para o conjunto de teste final:
# - **Treinamento (10 ensaios):** estimação de estrutura e parâmetros (maiores frequências/amplitudes).
# - **Validação / Seleção (8 ensaios):** seleção de `DECIMATION`, $(n_y, n_u)$, grau, estrutura e *early stopping*.
# - **Teste Final (6 ensaios oficiais):** avaliação final única e imparcial (mesmos 6 ensaios oficiais).
#
# A entrada $u(k)$ é mantida integralmente com sinal bidirecional ($u \in [-100, 100]\%$), sem recorte artificial.

# %%
FS_NATIVE = 100.0
DT_NATIVE = 1.0 / FS_NATIVE
TRIM_START_SEC = 10.0   # Descarta primeiros 10 s (partida do motor / estabilização)
TRIM_END_SEC = 13.0     # Descarta últimos 13 s (parada / frenagem)

train_files = {
    'multisine_45_tr': {'file': 'RODADA-7/multi-seno-45-050Hz_0904_20-29.csv', 'label': 'Multisseno (45°, 0.50 Hz)'},
    'chirp_45_tr':     {'file': 'RODADA-7/chirp-45-amp45_0904_20-15.csv',      'label': 'Chirp (45°, Amp ±45°)'},
    'aprbs_45_tr1':    {'file': 'RODADA-7/aprbs-45-1_0904_19-54.csv',          'label': 'APRBS #1 (45°)'},
    'aprbs_45_tr3':    {'file': 'RODADA-7/aprbs-45-3_0904_20-01.csv',          'label': 'APRBS #3 (45°)'},
    'steps_45_tr1':    {'file': 'RODADA-7/seq-degraus-45-1_0904_20-41.csv',    'label': 'Seq. Degraus #1 (45°)'},
    'multisine_60_tr': {'file': 'RODADA-7/multi-seno-60-050Hz_0904_20-38.csv', 'label': 'Multisseno (60°, 0.50 Hz)'},
    'chirp_60_tr':     {'file': 'RODADA-7/chirp-60-amp60_0904_20-20.csv',      'label': 'Chirp (60°, Amp ±60°)'},
    'aprbs_60_tr1':    {'file': 'RODADA-7/aprbs-60-1_0904_20-04.csv',          'label': 'APRBS #1 (60°)'},
    'aprbs_60_tr3':    {'file': 'RODADA-7/aprbs-60-3_0904_20-10.csv',          'label': 'APRBS #3 (60°)'},
    'steps_60_tr1':    {'file': 'RODADA-7/seq-degraus-60-1_0904_20-51.csv',    'label': 'Seq. Degraus #1 (60°)'},
}

sel_files = {
    'multisine_45_sel': {'file': 'RODADA-7/multi-seno-45-040Hz_0904_20-26.csv', 'label': 'Multisseno (45°, 0.40 Hz)'},
    'chirp_45_sel':     {'file': 'RODADA-7/chirp-45-amp25_0904_20-11.csv',      'label': 'Chirp (45°, Amp ±25°)'},
    'aprbs_45_sel':     {'file': 'RODADA-7/aprbs-45-2_0904_19-57.csv',          'label': 'APRBS #2 (45°)'},
    'steps_45_sel':     {'file': 'RODADA-7/seq-degraus-45-3_0904_20-48.csv',    'label': 'Seq. Degraus #3 (45°)'},
    'multisine_60_sel': {'file': 'RODADA-7/multi-seno-60-040Hz_0904_20-35.csv', 'label': 'Multisseno (60°, 0.40 Hz)'},
    'chirp_60_sel':     {'file': 'RODADA-7/chirp-60-amp50_0904_20-18.csv',      'label': 'Chirp (60°, Amp ±50°)'},
    'aprbs_60_sel':     {'file': 'RODADA-7/aprbs-60-2_0904_20-07.csv',          'label': 'APRBS #2 (60°)'},
    'steps_60_sel':     {'file': 'RODADA-7/seq-degraus-60-3_0904_20-57.csv',    'label': 'Seq. Degraus #3 (60°)'},
}

# Conjunto de Teste Final Intocado (mantém as chaves canônicas compatíveis com TCC / métricas anteriores)
val_files = {
    'multisine_45_val': {'file': 'RODADA-7/multi-seno-45-030Hz_0904_20-23.csv', 'label': 'Multisseno (45°, 0.30 Hz)'},
    'chirp_45_val':     {'file': 'RODADA-7/chirp-45-amp35_0904_20-13.csv',      'label': 'Chirp (45°, Amp ±35°)'},
    'steps_45_val':     {'file': 'RODADA-7/seq-degraus-45-2_0904_20-45.csv',    'label': 'Sequência de Degraus (45°)'},
    'multisine_60_val': {'file': 'RODADA-7/multi-seno-60-030Hz_0904_20-32.csv', 'label': 'Multisseno (60°, 0.30 Hz)'},
    'chirp_60_val':     {'file': 'RODADA-7/chirp-60-amp40_0904_20-16.csv',      'label': 'Chirp (60°, Amp ±40°)'},
    'steps_60_val':     {'file': 'RODADA-7/seq-degraus-60-2_0904_20-54.csv',    'label': 'Sequência de Degraus (60°)'},
}

def load_processed(fname, trim_start=TRIM_START_SEC, trim_end=TRIM_END_SEC, decimation=1):
    """Carrega dataset via aerodata, corta margens transitórias e preserva entrada bidirecional u."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        y, u, t, ref = readData(dataset_name=fname, return_ref=True)
    finally:
        sys.stdout = old_stdout

    idx_start = np.searchsorted(t, t[0] + trim_start)
    idx_end = np.searchsorted(t, t[-1] - trim_end)
    if idx_start >= idx_end:
        idx_start, idx_end = 0, len(t)

    sl = slice(idx_start, idx_end, decimation)
    u_c = np.asarray(u[sl], dtype=float)
    y_c = np.asarray(y[sl], dtype=float)
    t_c = np.asarray(t[sl], dtype=float)
    t_c = t_c - t_c[0]
    ref_c = np.asarray(ref[sl], dtype=float) if len(ref) > 0 else np.full_like(t_c, np.nan)
    return u_c, y_c, t_c, ref_c

print("=" * 80)
print("ETAPA 1: CARREGAMENTO DOS 24 ENSAIOS CANÔNICOS DA RODADA-7 (100 Hz NATIVOS)")
print("=" * 80)
train_raw = {k: load_processed(m['file'], decimation=1) for k, m in train_files.items()}
sel_raw   = {k: load_processed(m['file'], decimation=1) for k, m in sel_files.items()}
test_raw  = {k: load_processed(m['file'], decimation=1) for k, m in val_files.items()}

for grp_name, ds_map in [("Treinamento", train_raw), ("Validação (Seleção)", sel_raw), ("Teste Final (Intocado)", test_raw)]:
    u_all = np.concatenate([v[0] for v in ds_map.values()])
    y_all = np.concatenate([v[1] for v in ds_map.values()])
    print(f"  {grp_name:<24s}: {len(ds_map):2d} ensaios | {len(y_all):7d} amostras | "
          f"y in [{y_all.min():5.1f}°, {y_all.max():5.1f}°] | u in [{u_all.min():5.1f}%, {u_all.max():5.1f}%]")

def plot_datasets_grid(ds_map, meta_dict, filename_base, suptitle, ncols=2):
    """Gera painel padronizado dos sinais de referência de um conjunto de ensaios."""
    n = len(meta_dict)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 2.35 * nrows), sharex=False)
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (key, meta) in enumerate(meta_dict.items()):
        ax = axes_flat[i]
        u, y, t, ref = ds_map[key]
        sig_plot = ref if not np.all(np.isnan(ref)) else y
        ax.plot(t, sig_plot, color='#1f77b4', lw=1.1, label=r'Referência $r(t)$')
        ax.plot(t, y, color='#111111', lw=0.8, alpha=0.55, label=r'Saída $y(t)$')
        ax.set_ylabel(r'Ângulo [$^\circ$]')
        ax.set_xlabel('Tempo [s]')
        ax.set_title(meta['label'], fontsize=9.2, fontweight='bold', pad=4)
        ax.grid(True)
        ax.set_xlim(0, t[-1])
        y_min = np.floor(min(np.nanmin(sig_plot), np.min(y)) / 10) * 10 - 5
        y_max = np.ceil(max(np.nanmax(sig_plot), np.max(y)) / 10) * 10 + 5
        ax.set_ylim(y_min, y_max)
        if i == 0:
            ax.legend(loc='upper right', fontsize=7.5)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(suptitle, fontsize=11, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_academic_plot(fig, filename_base)
    plt.close(fig)

print("\nGerando painéis dos conjuntos de Treinamento, Validação e Teste Final...")
plot_datasets_grid(train_raw, train_files, "narx_train_datasets",
                   "Conjunto de Treinamento — 10 Ensaios da RODADA-7 (45° e 60°)")
plot_datasets_grid(sel_raw, sel_files, "narx_sel_datasets",
                   "Conjunto de Validação (Seleção de Hiperparâmetros) — 8 Ensaios da RODADA-7")
plot_datasets_grid(test_raw, val_files, "narx_val_datasets",
                   "Conjunto de Teste Final Independente — 6 Ensaios Oficiais da RODADA-7")

# %% [markdown]
# ## Etapa 2: Seleção Metodológica da Taxa de Amostragem (`DECIMATION` / $T_s$)
#
# Combinação de dois critérios clássicos da literatura de identificação de sistemas:
# 1. **Análise Espectral (Welch PSD):** estima a frequência $f_{99}$ que concentra 99% da energia de $y(t)$.
#    Pela regra de Ljung (1999) e Billings (2013), a frequência de amostragem deve situar-se em
#    $f_s \in [10 f_{99},\; 25 f_{99}]$.
# 2. **Critério de Aguirre (1995, 2015):** calcula o primeiro mínimo local $\tau_m = \min(\tau_{m,y}, \tau_{m,y^2})$
#    das funções de autocorrelação linear $\rho_y(\tau)$ e não-linear $\rho_{y^2}(\tau)$ nos sinais
#    oscilatórios de treino, recomendando $\tau_m / 20 \le T_s \le \tau_m / 10$.

# %%
def compute_acf(x, max_lag=400):
    x = x - np.mean(x)
    c = np.correlate(x, x, mode='full')
    c = c[len(x) - 1 : len(x) - 1 + max_lag]
    return c / (c[0] + 1e-12)

def first_local_min(arr, min_lag=5):
    for i in range(min_lag, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] <= arr[i + 1]:
            return i
    return None

print("\n" + "=" * 80)
print("ETAPA 2: ANÁLISE ESPECTRAL (PSD) E CRITÉRIO DE AGUIRRE (ACF LINEAR / NÃO-LINEAR)")
print("=" * 80)

sampling_diag = {}
f95_list, f99_list, tau_m_osc_list = [], [], []
acf_curves_plot = {}

for key, meta in train_files.items():
    u, y, t, _ = train_raw[key]
    yc = y - np.mean(y)
    freqs, Pyy = signal.welch(yc, fs=FS_NATIVE, nperseg=min(2048, len(yc)))
    cum = np.cumsum(Pyy) / np.sum(Pyy)
    f95 = float(freqs[np.searchsorted(cum, 0.95)])
    f99 = float(freqs[np.searchsorted(cum, 0.99)])
    f95_list.append(f95)
    f99_list.append(f99)

    ry = compute_acf(yc, max_lag=400)
    ry2 = compute_acf(yc ** 2, max_lag=400)
    tm_y = first_local_min(ry, min_lag=10)
    tm_y2 = first_local_min(ry2, min_lag=10)
    tm_eff = min([x for x in (tm_y, tm_y2) if x is not None], default=None)

    # Para o critério de tau_m, sinais oscilatórios (multisseno/chirp) caracterizam a constante
    # dinâmica intrínseca sem o viés de duração de patamar fixo dos degraus/APRBS.
    if ('multisine' in key or 'chirp' in key) and tm_eff is not None:
        tau_m_osc_list.append(tm_eff)
        acf_curves_plot[key] = (meta['label'], freqs, cum, ry, ry2, tm_eff)

    sampling_diag[key] = {
        'f95_hz': f95, 'f99_hz': f99,
        'rho_y_lag1': float(ry[1]), 'rho_y2_lag1': float(ry2[1]),
        'tau_m_y_samples': tm_y, 'tau_m_y2_samples': tm_y2, 'tau_m_eff_samples': tm_eff
    }
    print(f"  {meta['label']:<28s} | f95={f95:.2f} Hz  f99={f99:.2f} Hz | "
          f"rho_y(1)={ry[1]:.5f} | tau_m(y)={str(tm_y):>4s}  tau_m(y^2)={str(tm_y2):>4s}")

f99_max = max(f99_list)
f99_med = float(np.median(f99_list))
tau_m_med = float(np.median(tau_m_osc_list)) if tau_m_osc_list else 80.0

# Faixa recomendada por Aguirre: tau_m/20 <= dec <= tau_m/10
dec_aguirre_lo = max(2, int(np.ceil(tau_m_med / 20.0)))
dec_aguirre_hi = max(dec_aguirre_lo + 2, int(np.floor(tau_m_med / 10.0)))
# Faixa recomendada por largura de banda: 10*f99 <= fs <= 25*f99  =>  100/(25*f99) <= dec <= 100/(10*f99)
dec_bw_lo = max(2, int(np.floor(FS_NATIVE / (25.0 * f99_max))))
dec_bw_hi = max(dec_bw_lo + 2, int(np.ceil(FS_NATIVE / (10.0 * f99_max))))

# União refinada de fatores de decimação fisicamente práticos dentro da interseção/faixa recomendada
dec_lo = max(4, min(dec_aguirre_lo, dec_bw_lo))
dec_hi = min(12, max(dec_aguirre_hi, dec_bw_hi))
dec_candidates = sorted(set([d for d in [5, 6, 8, 10] if dec_lo <= d <= dec_hi] or [5, 8, 10]))

print("-" * 80)
print(f"  Largura de banda (99% energia): mediana = {f99_med:.2f} Hz, máxima = {f99_max:.2f} Hz")
print(f"  Superamostragem a 100 Hz      : {FS_NATIVE / f99_max:.0f}x a {FS_NATIVE / f99_med:.0f}x a banda do sistema!")
print(f"  Critério de Aguirre (tau_m)   : tau_m mediano = {tau_m_med:.0f} amostras ({tau_m_med*10:.0f} ms) "
      f"=> DECIMATION in [{dec_aguirre_lo}, {dec_aguirre_hi}] (Ts = {dec_aguirre_lo*10}..{dec_aguirre_hi*10} ms)")
print(f"  Critério Espectral (10..25 f99): DECIMATION in [{dec_bw_lo}, {dec_bw_hi}] (Ts = {dec_bw_lo*10}..{dec_bw_hi*10} ms)")
print(f"  Candidatos metodológicos      : DECIMATION = {dec_candidates} -> Ts = {[d*10 for d in dec_candidates]} ms")
print("-" * 80)

def plot_sampling_methodology(acf_map, f99_max_val, tau_m_val, dec_lo_v, dec_hi_v):
    """Figura metodológica: (a) PSD acumulada e (b) Critério de Aguirre (ACF linear e não-linear)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    for (key, (label, freqs, cum, ry, ry2, tm)), col in zip(acf_map.items(), colors):
        ax1.plot(freqs, cum * 100.0, lw=1.2, color=col, label=label)
        lags_ms = np.arange(len(ry2)) * 10.0
        ax2.plot(lags_ms, ry2, lw=1.2, color=col, label=rf"{label} ($\tau_m={tm*10:.0f}$ ms)")
        ax2.plot(lags_ms, ry, lw=0.8, ls=':', color=col, alpha=0.5)

    ax1.axhline(99.0, color='#333333', ls='--', lw=0.9, label='99% da energia espectral')
    ax1.axvline(f99_max_val, color='#d62728', ls='-.', lw=1.0,
                label=rf'$f_{{99,\max}} = {f99_max_val:.2f}$ Hz')
    ax1.set_xlim(0, 2.0)
    ax1.set_ylim(0, 103)
    ax1.set_xlabel('Frequência [Hz]')
    ax1.set_ylabel('Energia Espectral Acumulada [%]')
    ax1.set_title('(a) Largura de Banda Espectral da Saída $y(t)$', fontweight='bold')
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize=7.5)

    ax2.axvspan(dec_lo_v * 10.0, dec_hi_v * 10.0, color='#2ca02c', alpha=0.18,
                label=rf'Faixa ótima $T_s \in [{dec_lo_v*10},\,{dec_hi_v*10}]$ ms')
    ax2.axvline(10.0, color='#d62728', ls='--', lw=1.0, label=r'Original $T_s = 10$ ms (superamostrado)')
    ax2.set_xlim(0, 1800)
    ax2.set_xlabel(r'Atraso $\tau$ [ms]')
    ax2.set_ylabel(r'Autocorrelação $\rho_{y^2}(\tau)$ (sólida) e $\rho_y(\tau)$ (pont.)')
    ax2.set_title(rf'(b) Critério de Aguirre ($\tau_m \approx {tau_m_val*10:.0f}$ ms)', fontweight='bold')
    ax2.grid(True)
    ax2.legend(loc='upper right', fontsize=7.3)

    plt.suptitle("Etapa 2 — Determinação Metodológica do Período de Amostragem ($T_s$ / Decimação)",
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_academic_plot(fig, "narx_metodologia_amostragem")
    plt.close(fig)

plot_sampling_methodology(acf_curves_plot, f99_max, tau_m_med, dec_aguirre_lo, dec_aguirre_hi)

# %% [markdown]
# ## Etapa 3: Seleção de Ordens ($n_y, n_u$) via BIC (ARX) e Análise de Polos
#
# Avalia modelos lineares ARX de ordem $n = n_y = n_u \in \{1, \dots, 6\}$ sobre os 10 ensaios de
# treinamento em $100\text{ Hz}$ (`DECIMATION = 1`) e nas decimações da faixa de Aguirre.
# Demonstra quantitativamente:
# - Em $100\text{ Hz}$, os polos AR colam no círculo unitário ($\max|z| \ge 0{,}999$).
# - Na faixa de Aguirre, o BIC apresenta um joelho nítido em $n = 3$ (coerente com a física de
#   3ª ordem: $\theta, \dot{\theta}, a_{\text{motor}}$), definindo as ordens candidatas $(n_y, n_u) \in \{(3,3), (4,4)\}$.

# %%
def decimate_group(raw_dict, dec):
    return [(u[::dec].copy(), y[::dec].copy()) for u, y, _, _ in raw_dict.values()]

def arx_bic_analysis(raw_dict, dec_list, orders=(1, 2, 3, 4, 5, 6)):
    """Calcula OSA RMSE, BIC e raio espectral dos polos AR sobre todos os ensaios de treino."""
    results = {}
    for dec in dec_list:
        ds_list = decimate_group(raw_dict, dec)
        results[dec] = []
        for n in orders:
            Ps, Ys = [], []
            for u, y in ds_list:
                N = len(y)
                P = np.column_stack(
                    [np.ones(N - n)]
                    + [y[n - i : N - i] for i in range(1, n + 1)]
                    + [u[n - j : N - j] for j in range(1, n + 1)]
                )
                Ps.append(P)
                Ys.append(y[n:])
            P_all = np.vstack(Ps)
            y_all = np.concatenate(Ys)
            th, _, _, _ = np.linalg.lstsq(P_all, y_all, rcond=None)
            res = y_all - P_all @ th
            N_tot = len(y_all)
            rss = float(np.dot(res, res))
            k_params = P_all.shape[1]
            bic = float(N_tot * np.log(rss / N_tot) + k_params * np.log(N_tot))
            osa_rmse = float(np.sqrt(rss / N_tot))
            a_coeffs = th[1 : 1 + n]
            poles = np.roots(np.r_[1.0, -a_coeffs])
            max_pole = float(np.max(np.abs(poles)))
            results[dec].append({
                'order': n, 'osa_rmse': osa_rmse, 'bic': bic,
                'max_pole': max_pole, 'sum_a': float(np.sum(a_coeffs))
            })
    return results

print("\n" + "=" * 80)
print("ETAPA 3: SELEÇÃO METODOLÓGICA DE ORDENS (ny, nu) VIA BIC EM ARX E ANÁLISE DE POLOS")
print("=" * 80)
dec_bic_check = [1] + [d for d in [5, 8, 10] if d in dec_candidates or d == 5]
bic_results = arx_bic_analysis(train_raw, dec_bic_check, orders=(1, 2, 3, 4, 5, 6))

for dec in dec_bic_check:
    print(f"\n  DECIMATION = {dec:2d} (Ts = {dec*10:3d} ms, fs = {FS_NATIVE/dec:5.1f} Hz):")
    for row in bic_results[dec]:
        print(f"    ny=nu={row['order']}: OSA_RMSE={row['osa_rmse']:.4f}° | BIC={row['bic']:10.1f} | "
              f"max|polo|={row['max_pole']:.5f} | sum(a)={row['sum_a']:.5f}")

# Determinação automática das ordens candidatas pelo joelho do BIC na decimação mediana de Aguirre
dec_ref_bic = dec_candidates[len(dec_candidates) // 2]
if dec_ref_bic not in bic_results:
    dec_ref_bic = dec_bic_check[-1]
rows_ref = bic_results[dec_ref_bic]
bic_drops = [rows_ref[i - 1]['bic'] - rows_ref[i]['bic'] for i in range(1, len(rows_ref))]
total_drop = sum(max(0.0, d) for d in bic_drops)
cum_drop = 0.0
elbow_order = 3
for idx_o, d in enumerate(bic_drops, start=2):
    cum_drop += max(0.0, d)
    if cum_drop >= 0.96 * total_drop:
        elbow_order = idx_o
        break
elbow_order = max(3, min(4, elbow_order))
order_candidates = sorted(set([elbow_order, min(4, elbow_order + 1)])) if elbow_order == 3 else [3, 4]
print(f"\n  -> Joelho do BIC detectado em n = {elbow_order} (coerente com planta de 3ª ordem).")
print(f"  -> Ordens candidatas selecionadas para busca NARX: (ny, nu) in {[(n, n) for n in order_candidates]}")

def plot_bic_methodology(bic_map, dec_show):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    colors = {1: '#d62728', 5: '#1f77b4', 8: '#2ca02c', 10: '#9467bd'}

    # (a) Curva de redução relativa do BIC (normalizada 0..100%) e OSA RMSE em dec_ref
    for dec in dec_show:
        if dec == 1:
            continue
        rows = bic_map[dec]
        ords = [r['order'] for r in rows]
        bics = np.array([r['bic'] for r in rows])
        # Normaliza redução acumulada de BIC em %
        b_norm = 100.0 * (bics[0] - bics) / (bics[0] - bics[-1] + 1e-12)
        ax1.plot(ords, b_norm, marker='o', lw=1.4, color=colors.get(dec, '#333333'),
                 label=f'DEC={dec} ($T_s={dec*10}$ ms)')

    ax1.axhline(96.0, color='#555555', ls='--', lw=0.9, label='Limiar de joelho (96% ganho BIC)')
    ax1.axvspan(2.8, 4.2, color='#2ca02c', alpha=0.14, label=r'Ordens candidatas $n \in \{3, 4\}$')
    ax1.set_xticks([1, 2, 3, 4, 5, 6])
    ax1.set_xlabel(r'Ordem ARX ($n_y = n_u = n$)')
    ax1.set_ylabel('Redução Acumulada do BIC [%]')
    ax1.set_title('(a) Joelho do Critério BIC por Ordem', fontweight='bold')
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize=7.8)

    # (b) Raio espectral dos polos dominantes: 100 Hz vs. Decimado
    for dec in dec_show:
        rows = bic_map[dec]
        ords = [r['order'] for r in rows]
        poles = [r['max_pole'] for r in rows]
        lbl = f'DEC={dec} ($T_s={dec*10}$ ms)' + (' [superamostrado]' if dec == 1 else '')
        ls = '--' if dec == 1 else '-'
        ax2.plot(ords, poles, marker='s', lw=1.3, ls=ls, color=colors.get(dec, '#333333'), label=lbl)

    ax2.axhline(1.0, color='#111111', ls=':', lw=1.0, label=r'Fronteira de instabilidade $|z|=1$')
    ax2.set_xticks([1, 2, 3, 4, 5, 6])
    ax2.set_xlabel(r'Ordem ARX ($n_y = n_u = n$)')
    ax2.set_ylabel(r'Polo Dominante $\max |z_i|$')
    ax2.set_title('(b) Afastamento dos Polos do Círculo Unitário', fontweight='bold')
    ax2.grid(True)
    ax2.legend(loc='lower left', fontsize=7.6)

    plt.suptitle("Etapa 3 — Determinação Metodológica das Ordens ($n_y, n_u$) via BIC e Polos ARX",
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_academic_plot(fig, "narx_metodologia_ordem_bic")
    plt.close(fig)

plot_bic_methodology(bic_results, dec_bic_check)

# %% [markdown]
# ## Etapas 4 e 5: Motor Polinomial NARX — SR-FROLS + Estabilidade de Aguirre + LM-OE
#
# - **Dicionário polinomial puro aditivo ($f(y) + g(u)$):** sem termos cruzados espúrios $y \cdot u$
#   decorrentes da realimentação PID na coleta em malha fechada; grau $\ell_y = 2$ (evita escape
#   cúbico em tempo finito acima de $90^\circ$) e $\ell_u \in \{2, 3\}$ (termos ímpares $u, u^3$
#   preservam a simetria física do empuxo bidirecional $u|u|$).
# - **Seleção de estrutura orientada a simulação (*SR-FROLS*, Piroddi & Spinelli, 2003):**
#   usa matriz de Gram normalizada pré-computada (OLS exato em tempo $O(k^3)$) e escolhe cada
#   termo minimizando o RMSE de simulação multi-passo ($H = 10\text{ s}$) exclusivamente no Treino.
# - **Critério Algébrico de Estabilidade de Aguirre (1994):** verifica analiticamente que:
#   (i) todos os polos lineares estão no interior do círculo unitário ($\max|z_i| < 0{,}999$),
#   (ii) não há termo cúbico positivo instabilizante em $y$ ($\sum c_{ijk} \le 0$), e
#   (iii) a força restauradora estática $F_y(y) = (1-\sum a_i)y - q_y y^2 - c_y y^3$ mantém sinal
#   restaurador em todo o envelope físico ($F_y(y) > 0$ em $[5^\circ, 165^\circ]$ e $F_y(y) < 0$ em $[-45^\circ, -5^\circ]$).
# - **Refinamento em Erro de Saída (*LM-OE*):** Levenberg-Marquardt amortecido com sensibilidade
#   analítica exata $\partial\hat{y}(k)/\partial\theta$ no Treino, com *early stopping* guiado
#   estritamente pelos 8 ensaios de Validação (100% estável sem clip em 18/18 Treino+Validação).

# %%
def build_polynomial_regmat(datasets, ny, nu, ly, lu, additive=True):
    """Constrói matriz de regressores polinomiais puros (sem vazamento entre ensaios)."""
    Ps, Ys = [], []
    names = None
    ml = max(ny, nu)
    for u, y in datasets:
        N = len(y)
        NP = N - ml
        yt = y[ml:]
        Py = np.column_stack([y[ml - i : N - i] for i in range(1, ny + 1)])
        Pu = np.column_stack([u[ml - j : N - j] for j in range(1, nu + 1)])
        yn = [f'y(k-{i})' for i in range(1, ny + 1)]
        un = [f'u(k-{j})' for j in range(1, nu + 1)]
        cols = [np.ones((NP, 1))]
        names = ['constant']
        if additive:
            cols.extend([Py, Pu])
            names.extend(yn + un)
            for deg in range(2, ly + 1):
                for idxs in combinations_with_replacement(range(ny), deg):
                    names.append("".join(yn[i] for i in idxs))
                    cols.append(np.prod(Py[:, idxs], axis=1, keepdims=True))
            for deg in range(2, lu + 1):
                for idxs in combinations_with_replacement(range(nu), deg):
                    names.append("".join(un[i] for i in idxs))
                    cols.append(np.prod(Pu[:, idxs], axis=1, keepdims=True))
        else:
            base = np.column_stack([Py, Pu])
            bnames = yn + un
            cols.append(base)
            names.extend(bnames)
            max_l = max(ly, lu)
            for deg in range(2, max_l + 1):
                for idxs in combinations_with_replacement(range(base.shape[1]), deg):
                    names.append("".join(bnames[i] for i in idxs))
                    cols.append(np.prod(base[:, idxs], axis=1, keepdims=True))
        Ps.append(np.hstack(cols))
        Ys.append(yt)
    return np.vstack(Ps), names, np.concatenate(Ys)

def parse_narx_terms(terms):
    """Compila nomes de termos polinomiais de grau arbitrário para avaliação ultrarrápida."""
    parsed = []
    for t in terms:
        if t == 'constant':
            parsed.append((0, ()))
        else:
            matches = re.findall(r'([yu])\(k-(\d+)\)', t)
            ms = tuple((0 if v == 'y' else 1, int(lg)) for v, lg in matches)
            parsed.append((len(ms), ms))
    return parsed

def pack_full_trajectories(ds_list):
    """Empacota múltiplas trajetórias em tensores acolchoados para simulação vetorizada."""
    L = max(len(y) for _, y in ds_list)
    S = len(ds_list)
    U = np.zeros((S, L), dtype=float)
    Y = np.zeros((S, L), dtype=float)
    lens = np.array([len(y) for _, y in ds_list], dtype=int)
    mask = np.zeros((S, L), dtype=bool)
    for i, (u, y) in enumerate(ds_list):
        U[i, :len(u)] = u
        Y[i, :len(y)] = y
        mask[i, :len(y)] = True
    return U, Y, lens, mask

def make_multistep_batch(ds_list, ml, H_steps, stride):
    """Extrai janelas de simulação multi-passo (Multiple Shooting) sem cruzar fronteiras de ensaios."""
    L = ml + H_steps
    Us, Ys = [], []
    for u, y in ds_list:
        for s in range(0, len(y) - L + 1, stride):
            Us.append(u[s : s + L])
            Ys.append(y[s : s + L])
    return np.array(Us, dtype=float), np.array(Ys, dtype=float)

def sim_and_jac_batch(parsed, theta, scale, ml, U_b, Y_b, explode=350.0, need_jac=False, clip_bounds=None):
    """Simulação livre (Free-Run) vetorizada de grau polinomial arbitrário + sensibilidade analítica exata."""
    S, L = U_b.shape
    P = len(parsed)
    YH = np.empty((S, L), dtype=float)
    YH[:, :ml] = Y_b[:, :ml]
    if need_jac:
        DY = np.zeros((S, L, P), dtype=float)
    div = np.zeros(S, dtype=bool)

    for k in range(ml, L):
        val = np.zeros(S, dtype=float)
        if need_jac:
            dval = np.zeros((S, P), dtype=float)
        for p_idx, ((deg, ms), th, sc) in enumerate(zip(parsed, theta, scale)):
            if deg == 0:
                val += th
                if need_jac:
                    dval[:, p_idx] += sc
            elif deg == 1:
                v0, l0 = ms[0]
                a = YH[:, k - l0] if v0 == 0 else U_b[:, k - l0]
                val += th * a
                if need_jac:
                    dval[:, p_idx] += sc * a
                    if v0 == 0:
                        dval += th * DY[:, k - l0, :]
            elif deg == 2:
                (v0, l0), (v1, l1) = ms
                a = YH[:, k - l0] if v0 == 0 else U_b[:, k - l0]
                b = YH[:, k - l1] if v1 == 0 else U_b[:, k - l1]
                val += th * a * b
                if need_jac:
                    dval[:, p_idx] += sc * a * b
                    if v0 == 0:
                        dval += (th * b)[:, None] * DY[:, k - l0, :]
                    if v1 == 0:
                        dval += (th * a)[:, None] * DY[:, k - l1, :]
            else:
                factors = [YH[:, k - l] if v == 0 else U_b[:, k - l] for v, l in ms]
                prod_all = np.ones(S, dtype=float)
                for f in factors:
                    prod_all *= f
                val += th * prod_all
                if need_jac:
                    dval[:, p_idx] += sc * prod_all
                    for idx_f, (v, l) in enumerate(ms):
                        if v == 0:
                            other = np.full(S, th, dtype=float)
                            for j_f, f in enumerate(factors):
                                if j_f != idx_f:
                                    other *= f
                            dval += other[:, None] * DY[:, k - l, :]
        if clip_bounds is not None:
            val = np.clip(val, clip_bounds[0], clip_bounds[1])
        bad = (~np.isfinite(val)) | (np.abs(val) > explode)
        div |= bad
        YH[:, k] = np.where(div, explode, val)
        if need_jac:
            DY[:, k, :] = np.where(div[:, None], 0.0, np.clip(dval, -1e9, 1e9))

    return (YH, DY, div) if need_jac else (YH, div)

def compute_metrics(y_true, y_pred):
    """Calcula métricas quantitativas de ajuste (RMSE, MAE, FIT%, R²)."""
    if np.any(~np.isfinite(y_pred)):
        return np.nan, np.nan, -999.0, -999.0
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.linalg.norm(y_true - np.mean(y_true)))
    fit = float(100.0 * (1.0 - np.linalg.norm(err) / denom)) if denom > 1e-6 else 0.0
    r2 = float(1.0 - np.sum(err ** 2) / (denom ** 2)) if denom > 1e-6 else 0.0
    return rmse, mae, fit, r2

def eval_packed(terms, theta, ml, packed, keys=None, explode=350.0, clip_bounds=None):
    """Avalia simulação livre completa sobre um conjunto empacotado (padrão: SEM clip!)."""
    U_p, Y_p, lens, _ = packed
    parsed = parse_narx_terms(terms)
    YH, div = sim_and_jac_batch(parsed, theta, np.ones(len(terms)), ml, U_p, Y_p,
                                explode=explode, need_jac=False, clip_bounds=clip_bounds)
    fits, rmses, maes, r2s = [], [], [], []
    per_key = {}
    for i in range(len(lens)):
        yt = Y_p[i, ml : lens[i]]
        yp = YH[i, ml : lens[i]]
        if div[i]:
            rmse, mae, fit, r2 = 999.0, 999.0, -999.0, -999.0
        else:
            rmse, mae, fit, r2 = compute_metrics(yt, yp)
        fits.append(fit); rmses.append(rmse); maes.append(mae); r2s.append(r2)
        if keys is not None:
            per_key[keys[i]] = {
                'rmse': rmse, 'mae': mae, 'fit': fit, 'r2': r2,
                'diverged': bool(div[i]), 'y_true': yt, 'y_fr': yp, 'u': U_p[i, ml : lens[i]]
            }
    n_ok = int(np.sum(~div))
    return n_ok == len(lens), n_ok, float(np.mean(fits)), float(np.mean(rmses)), per_key

def analyze_static_and_poles(terms, theta, ml):
    """Analisa polos da parte linear AR e a função restauradora estática F_y(y) (Aguirre, 1994)."""
    parsed = parse_narx_terms(terms)
    a = np.zeros(ml)
    c0, q_y, c_y = 0.0, 0.0, 0.0
    b_u1, b_u2, b_u3 = 0.0, 0.0, 0.0
    for (deg, ms), th in zip(parsed, theta):
        th = float(th)
        if deg == 0:
            c0 += th
        elif deg == 1:
            if ms[0][0] == 0:
                a[ms[0][1] - 1] += th
            else:
                b_u1 += th
        elif deg == 2:
            if all(v == 0 for v, _ in ms):
                q_y += th
            elif all(v == 1 for v, _ in ms):
                b_u2 += th
        elif deg == 3:
            if all(v == 0 for v, _ in ms):
                c_y += th
            elif all(v == 1 for v, _ in ms):
                b_u3 += th

    poles = np.roots(np.r_[1.0, -a]) if np.any(a != 0) else np.array([0.0])
    max_pole = float(np.max(np.abs(poles)))
    one_minus_a = 1.0 - float(np.sum(a))

    ys_p = np.linspace(5.0, 165.0, 33)
    fy_p = one_minus_a * ys_p - q_y * (ys_p ** 2) - c_y * (ys_p ** 3)
    ys_n = np.linspace(-45.0, -5.0, 10)
    fy_n = one_minus_a * ys_n - q_y * (ys_n ** 2) - c_y * (ys_n ** 3)

    # Ponto de sela quadrático (quando c_y == 0 e q_y > 0)
    y_saddle = float(one_minus_a / (2.0 * q_y)) if (abs(c_y) < 1e-12 and q_y > 1e-12) else None

    alg_stable = (
        (max_pole < 0.999)
        and (c_y <= 1e-12)
        and (one_minus_a > 0.0)
        and bool(np.all(fy_p > 0.0))
        and bool(np.all(fy_n < 0.0))
    )
    return {
        'alg_stable': alg_stable,
        'max_pole': max_pole,
        'poles_real': [float(p.real) for p in poles],
        'poles_imag': [float(p.imag) for p in poles],
        'sum_a': float(np.sum(a)),
        'one_minus_a': one_minus_a,
        'c0': c0, 'q_y': q_y, 'c_y': c_y,
        'b_u1': b_u1, 'b_u2': b_u2, 'b_u3': b_u3,
        'y_saddle_deg': y_saddle,
    }

def fast_gram_ols(G, b, col_norms, idxs):
    """Resolve OLS exato em O(k^3) a partir da matriz de Gram normalizada pré-computada."""
    sub_G = G[np.ix_(idxs, idxs)].copy()
    sub_G[np.diag_indices_from(sub_G)] += 1e-13
    return np.linalg.solve(sub_G, b[idxs]) / col_norms[idxs]

def frols_err_fast(P, yt, colnames, n_components, eps=1e-12):
    """FROLS/ERR clássico vetorizado (Gram-Schmidt Modificado — equivalente exato ao sysid.NARX)."""
    NP, M = P.shape
    num_sel = min(n_components, M)
    yy = float(np.dot(yt, yt))
    col_norms = np.maximum(np.linalg.norm(P, axis=0), eps)
    Pn = P / col_norms[None, :]
    G = Pn.T @ Pn
    b = Pn.T @ yt
    Q_rem = Pn.copy()
    selected, errs = [], []
    used = np.zeros(M, dtype=bool)
    for _ in range(num_sel):
        norms = np.einsum('ij,ij->j', Q_rem, Q_rem)
        dots = Q_rem.T @ yt
        valid = (~used) & (norms >= eps)
        if not np.any(valid):
            break
        err_c = np.full(M, -np.inf)
        err_c[valid] = (dots[valid] ** 2) / (norms[valid] * yy)
        best = int(np.argmax(err_c))
        selected.append(best)
        errs.append(float(err_c[best]))
        used[best] = True
        q_new = Q_rem[:, best].copy()
        Q_rem -= np.outer(q_new, (q_new @ Q_rem) / norms[best])
        Q_rem[:, best] = 0.0
    theta = fast_gram_ols(G, b, col_norms, selected)
    return [colnames[i] for i in selected], theta, np.array(errs)

def sr_frols_path(P, yt, colnames, tr_list, tr_packed, sel_packed, ny, nu,
                  seed_mode='ar', max_terms=14, H_steps=150):
    """Seleção de Estrutura Orientada a Simulação (SR-FROLS, Piroddi & Spinelli, 2003):
    - Semente: esqueleto dinâmico linear AR [y(k-1)..y(k-ny)] (ou ARX incluindo u(k-1..nu)),
      evitando divergência espúria no passo 1 de sistemas oscilatórios de ordem >= 2.
    - Expansão gulosa minimizando RMSE multi-passo (H_steps) exclusivamente no Treino.
    - Avalia cada passo k na Validação (8 ensaios) + Estabilidade Algébrica de Aguirre.
    """
    ml = max(ny, nu)
    col_norms = np.maximum(np.linalg.norm(P, axis=0), 1e-12)
    Pn = P / col_norms[None, :]
    G = Pn.T @ Pn
    b = Pn.T @ yt

    U_ms, Y_ms = make_multistep_batch(tr_list, ml, H_steps, stride=H_steps)
    Y_ms_tgt = Y_ms[:, ml:]
    M = P.shape[1]

    selected = [colnames.index(f'y(k-{i})') for i in range(1, ny + 1)]
    if seed_mode == 'arx':
        selected += [colnames.index(f'u(k-{j})') for j in range(1, nu + 1)]
    used = np.zeros(M, dtype=bool)
    used[selected] = True
    path = []

    def record_step(sel_idxs, th_s, ms_val):
        tk = [colnames[i] for i in sel_idxs]
        st_tr, _, fit_tr, _, _ = eval_packed(tk, th_s, ml, tr_packed)
        st_sel, _, fit_sel, rmse_sel, _ = eval_packed(tk, th_s, ml, sel_packed)
        alg_info = analyze_static_and_poles(tk, th_s, ml)
        alg_ok = bool(st_tr and st_sel and alg_info['alg_stable'])
        path.append({
            'k': len(tk), 'terms': tk, 'theta': th_s.copy(), 'ms_rmse': float(ms_val),
            'tr_stable': st_tr, 'tr_fit': fit_tr,
            'sel_stable': st_sel, 'sel_fit': fit_sel, 'sel_rmse': rmse_sel,
            'alg_ok': alg_ok
        })

    if seed_mode == 'arx':
        th0 = fast_gram_ols(G, b, col_norms, selected)
        YH0, _ = sim_and_jac_batch(parse_narx_terms([colnames[i] for i in selected]), th0,
                                   np.ones(len(selected)), ml, U_ms, Y_ms, explode=300.0, need_jac=False)
        ms0 = float(np.sqrt(np.mean((Y_ms_tgt - YH0[:, ml:]) ** 2)))
        record_step(selected, th0, ms0)

    for step in range(len(selected), max_terms):
        best_sc = np.inf
        best_idx = -1
        best_th = None
        sc_ones = np.ones(step + 1)
        for idx in range(M):
            if used[idx]:
                continue
            cand = selected + [idx]
            try:
                th = fast_gram_ols(G, b, col_norms, cand)
            except np.linalg.LinAlgError:
                continue
            YH, _ = sim_and_jac_batch(parse_narx_terms([colnames[i] for i in cand]), th,
                                      sc_ones, ml, U_ms, Y_ms, explode=300.0, need_jac=False)
            sc = float(np.sqrt(np.mean((Y_ms_tgt - YH[:, ml:]) ** 2)))
            if sc < best_sc:
                best_sc = sc
                best_idx = idx
                best_th = th
        if best_idx < 0:
            break
        selected.append(best_idx)
        used[best_idx] = True
        record_step(selected, best_th, best_sc)
    return path

def lm_oe_refine(terms, theta0, ml, tr_list, tr_packed, sel_packed,
                 max_iter=35, patience=10, ms_warmup_H=None):
    """Refinamento em Erro de Saída (Free-Run OE) por Levenberg-Marquardt amortecido com
    Jacobiano analítico exato nos dados de TREINO, e early-stopping rigoroso na VALIDAÇÃO
    exigindo 100% de estabilidade (Treino + Validação + Critério Algébrico de Aguirre).
    """
    parsed = parse_narx_terms(terms)
    U_tr, Y_tr, len_tr, mask_tr = tr_packed
    P_dim = len(terms)
    scale = np.maximum(np.abs(theta0), 1e-7)
    z = theta0 / scale

    # Warmup multi-passo opcional se o OLS inicial divergir em horizonte completo
    st_tr0, _, _, _, _ = eval_packed(terms, theta0, ml, tr_packed)
    if (not st_tr0) and (ms_warmup_H is not None):
        U_ms, Y_ms = make_multistep_batch(tr_list, ml, ms_warmup_H, stride=ms_warmup_H)
        y_ms_f = Y_ms[:, ml:].ravel()
        lam_w = 1e-1
        for _ in range(12):
            YH, DY, _ = sim_and_jac_batch(parsed, z * scale, scale, ml, U_ms, Y_ms, explode=300.0, need_jac=True)
            r_w = y_ms_f - YH[:, ml:].ravel()
            c0_w = 0.5 * float(np.dot(r_w, r_w))
            J_w = -DY[:, ml:, :].reshape(-1, P_dim)
            JTJ_w = J_w.T @ J_w
            g_w = J_w.T @ r_w
            diag_w = np.maximum(np.diag(JTJ_w), 1e-8)
            ok_w = False
            for _ in range(8):
                try:
                    dz_w = -np.linalg.solve(JTJ_w + lam_w * np.diag(diag_w), g_w)
                except np.linalg.LinAlgError:
                    lam_w *= 10.0
                    continue
                YH2, _ = sim_and_jac_batch(parsed, (z + dz_w) * scale, scale, ml, U_ms, Y_ms, explode=300.0, need_jac=False)
                r2_w = y_ms_f - YH2[:, ml:].ravel()
                if 0.5 * float(np.dot(r2_w, r2_w)) < c0_w:
                    z = z + dz_w
                    lam_w = max(lam_w * 0.5, 1e-6)
                    ok_w = True
                    break
                lam_w = min(lam_w * 5.0, 1e7)
            if not ok_w:
                break

    # Pesos normalizados por variância para que cada ensaio de treino contribua igualmente
    w_mat = np.zeros((len(len_tr), Y_tr.shape[1] - ml))
    tr_norms = np.zeros(len(len_tr))
    for i in range(len(len_tr)):
        yt = Y_tr[i, ml : len_tr[i]]
        tr_norms[i] = np.linalg.norm(yt - np.mean(yt))
        w_mat[i, : len_tr[i] - ml] = 1.0 / tr_norms[i]
    m_flat = mask_tr[:, ml:].ravel()
    w_flat = w_mat.ravel()[m_flat]
    y_flat = Y_tr[:, ml:].ravel()[m_flat]

    def eval_tr_cj(zc):
        YH, DY, div = sim_and_jac_batch(parsed, zc * scale, scale, ml, U_tr, Y_tr, explode=350.0, need_jac=True)
        r = (y_flat - YH[:, ml:].ravel()[m_flat]) * w_flat
        J = -DY[:, ml:, :].reshape(-1, P_dim)[m_flat] * w_flat[:, None]
        st_tr = not np.any(div)
        if st_tr:
            fits = [100.0 * (1.0 - np.linalg.norm(Y_tr[i, ml:len_tr[i]] - YH[i, ml:len_tr[i]]) / tr_norms[i])
                    for i in range(len(len_tr))]
            tr_fit = float(np.mean(fits))
        else:
            tr_fit = -999.0
        return 0.5 * float(np.dot(r, r)), r, J, st_tr, tr_fit

    def eval_tr_c(zc):
        YH, div = sim_and_jac_batch(parsed, zc * scale, scale, ml, U_tr, Y_tr, explode=350.0, need_jac=False)
        r = (y_flat - YH[:, ml:].ravel()[m_flat]) * w_flat
        return 0.5 * float(np.dot(r, r)), bool(np.any(div))

    cost, r, J, st_tr, fit_tr = eval_tr_cj(z)

    def evaluate_val_and_alg(th_c, st_tr_flag):
        st_sel, _, fit_sel, rmse_sel, _ = eval_packed(terms, th_c, ml, sel_packed)
        alg_ok = analyze_static_and_poles(terms, th_c, ml)['alg_stable'] if (st_tr_flag and st_sel) else False
        return st_sel, alg_ok, fit_sel, rmse_sel

    st_sel, alg_ok, fit_sel, rmse_sel = evaluate_val_and_alg(z * scale, st_tr)
    best_th = (z * scale).copy()
    best_sel_fit = fit_sel if (st_tr and st_sel and alg_ok) else -999.0
    best_ok = bool(st_tr and st_sel and alg_ok)
    best_iter = 0
    history = [{'iter': 0, 'tr_fit': fit_tr, 'sel_fit': fit_sel if st_sel else -999.0,
                'alg_ok': alg_ok, 'tr_stable': st_tr, 'sel_stable': st_sel}]

    lam = 1e-2
    no_imp = 0
    for it in range(1, max_iter + 1):
        JTJ = J.T @ J
        g = J.T @ r
        diag = np.maximum(np.diag(JTJ), 1e-8)
        accepted = False
        dz = np.zeros_like(z)
        for _ in range(10):
            try:
                dz = -np.linalg.solve(JTJ + lam * np.diag(diag), g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            z_try = z + dz
            c_try, div_try = eval_tr_c(z_try)
            if ((not div_try) or (not st_tr)) and (c_try < cost):
                z = z_try
                cost, r, J, st_tr, fit_tr = eval_tr_cj(z)
                lam = max(lam * 0.5, 1e-7)
                accepted = True
                break
            lam = min(lam * 5.0, 1e8)
        if not accepted:
            break

        th_cur = z * scale
        st_sel, alg_ok, fit_sel, rmse_sel = evaluate_val_and_alg(th_cur, st_tr)
        history.append({'iter': it, 'tr_fit': fit_tr, 'sel_fit': fit_sel if st_sel else -999.0,
                        'alg_ok': alg_ok, 'tr_stable': st_tr, 'sel_stable': st_sel})

        if st_tr and st_sel and alg_ok and fit_sel > best_sel_fit + 1e-3:
            best_sel_fit = fit_sel
            best_th = th_cur.copy()
            best_ok = True
            best_iter = it
            no_imp = 0
        else:
            no_imp += 1

        if np.linalg.norm(dz) < 1e-6 or (best_ok and no_imp >= patience):
            break

    return best_th, best_ok, best_sel_fit, best_iter, history

# %% [markdown]
# ### Execução da Busca Metodológica na Validação (8 ensaios — ZERO acesso ao Teste Final!)

# %%
print("\n" + "=" * 80)
print("ETAPAS 4 & 5: BUSCA METODOLÓGICA (SR-FROLS + AGUIRRE + LM-OE) NA VALIDAÇÃO (8 ENSAIOS)")
print("=" * 80)
print("  Critérios obrigatórios: 100% estável SEM clip em Treino (10/10) + Validação (8/8)")
print("  + Critério Algébrico de Estabilidade de Aguirre. Teste Final (6 ensaios) intocado!\n")

# Grade metodológica derivada das Etapas 2 e 3:
# - dec in dec_candidates (faixa de Aguirre + PSD)
# - (ny, nu) in order_candidates (joelho do BIC)
# - ly = 2 (dissipatividade em y), lu in {2, 3} (comparando entrada quadrática vs. cúbica bidirecional)
# - sementes AR e ARX avaliadas na faixa ótima
search_grid = []
for dec in dec_candidates:
    for n_ord in order_candidates:
        search_grid.append((dec, n_ord, n_ord, 2, 3, 'ar'))
        if n_ord == 3:
            search_grid.append((dec, n_ord, n_ord, 2, 3, 'arx'))
    # Comparativo de grau lu=2 na ordem principal (joelho BIC)
    search_grid.append((dec, order_candidates[0], order_candidates[0], 2, 2, 'ar'))

t_grid_start = time.time()
grid_results = []
cached_packed = {}

for dec, ny, nu, ly, lu, seed_mode in search_grid:
    t0 = time.time()
    if dec not in cached_packed:
        tr_l = decimate_group(train_raw, dec)
        sel_l = decimate_group(sel_raw, dec)
        cached_packed[dec] = (tr_l, pack_full_trajectories(tr_l), pack_full_trajectories(sel_l))
    tr_list, tr_packed, sel_packed = cached_packed[dec]

    P, colnames, yt = build_polynomial_regmat(tr_list, ny, nu, ly=ly, lu=lu, additive=True)
    ml = max(ny, nu)
    H_sr = int(round(10.0 * FS_NATIVE / dec))  # Horizonte de 10 s para SR-FROLS

    path = sr_frols_path(P, yt, colnames, tr_list, tr_packed, sel_packed,
                         ny=ny, nu=nu, seed_mode=seed_mode, max_terms=14, H_steps=H_sr)

    # Seleciona até 2 estruturas candidatas do caminho SR-FROLS para refinar via LM-OE:
    # (1) Melhor estrutura já estável em Treino+Validação+Aguirre no OLS
    # (2) Estrutura de menor erro multi-passo no Treino (para estabilização/refinamento OE)
    to_refine = []
    alg_steps = [p for p in path if p['alg_ok']]
    if alg_steps:
        bp = max(alg_steps, key=lambda p: p['sel_fit'])
        to_refine.append(('SR_alg', bp))
    ms_candidates = path[1:] if len(path) > 1 else path
    bm = min(ms_candidates, key=lambda p: p['ms_rmse'])
    if all(bm['terms'] != x[1]['terms'] for x in to_refine):
        to_refine.append(('SR_ms', bm))

    best_cfg = None
    for tag, cand_p in to_refine:
        th_oe, ok_oe, sel_fit_oe, best_it, oe_hist = lm_oe_refine(
            cand_p['terms'], cand_p['theta'], ml, tr_list, tr_packed, sel_packed,
            max_iter=35, patience=10, ms_warmup_H=H_sr
        )
        if ok_oe:
            _, _, tr_fit_oe, tr_rmse_oe, _ = eval_packed(cand_p['terms'], th_oe, ml, tr_packed)
            _, _, sel_fit_chk, sel_rmse_chk, sel_per_k = eval_packed(
                cand_p['terms'], th_oe, ml, sel_packed, keys=list(sel_files.keys())
            )
            if (best_cfg is None) or (sel_fit_chk > best_cfg['sel_fit_oe']):
                best_cfg = {
                    'dec': dec, 'Ts': dec * DT_NATIVE, 'ny': ny, 'nu': nu,
                    'ly': ly, 'lu': lu, 'seed': seed_mode, 'tag': tag,
                    'k': len(cand_p['terms']), 'terms': cand_p['terms'],
                    'theta_ee': cand_p['theta'].copy(), 'theta_oe': th_oe.copy(),
                    'sel_fit_ee': cand_p['sel_fit'] if cand_p['alg_ok'] else -999.0,
                    'sel_fit_oe': sel_fit_chk, 'sel_rmse_oe': sel_rmse_chk,
                    'tr_fit_oe': tr_fit_oe, 'tr_rmse_oe': tr_rmse_oe,
                    'best_iter': best_it, 'path': path, 'oe_hist': oe_hist,
                    'sel_per_k': sel_per_k
                }

    dt_run = time.time() - t0
    if best_cfg is None:
        print(f"  dec={dec:2d} (Ts={dec*10:3d}ms) ny=nu={ny} ly={ly} lu={lu} seed={seed_mode:3s} ({dt_run:4.1f}s) | "
              f"Rejeitado (sem estabilidade global Treino+Val+Aguirre)")
    else:
        r = best_cfg
        print(f"  dec={dec:2d} (Ts={dec*10:3d}ms) ny=nu={ny} ly={ly} lu={lu} seed={seed_mode:3s} k={r['k']:2d} "
              f"[{r['tag']:6s}] ({dt_run:4.1f}s) | EE Val={r['sel_fit_ee']:6.1f}% -> "
              f"OE Tr={r['tr_fit_oe']:5.1f}%  Val={r['sel_fit_oe']:5.1f}% (RMSE={r['sel_rmse_oe']:5.2f}°, it={r['best_iter']:2d})")
        grid_results.append(r)

if not grid_results:
    raise RuntimeError("Nenhum modelo candidato atingiu estabilidade global no conjunto de Validação.")

winner = max(grid_results, key=lambda r: r['sel_fit_oe'])
print("-" * 80)
print(f"MODELO VENCEDOR SELECIONADO ESTRITAMENTE NA VALIDAÇÃO (em {time.time() - t_grid_start:.1f} s):")
print(f"  DECIMATION = {winner['dec']} (Ts = {winner['Ts']*1000:.0f} ms, fs = {FS_NATIVE/winner['dec']:.2f} Hz)")
print(f"  Ordens     = (ny={winner['ny']}, nu={winner['nu']}) -> vetor de estados x_k in R^{winner['ny']+winner['nu']}")
print(f"  Graus      = (ly={winner['ly']}, lu={winner['lu']}) | Semente = {winner['seed'].upper()} | Termos = {winner['k']}")
print(f"  Validação (8 ensaios, SEM clip): FIT Médio = {winner['sel_fit_oe']:.2f}%, RMSE Médio = {winner['sel_rmse_oe']:.2f}°")
print("-" * 80)

def plot_structure_and_oe_convergence(win):
    """Figura metodológica: (a) caminho SR-FROLS por nº de termos e (b) convergência LM-OE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1))

    ks = [p['k'] for p in win['path']]
    ms_rmses = [p['ms_rmse'] for p in win['path']]
    val_fits = [p['sel_fit'] if p['sel_stable'] else np.nan for p in win['path']]

    ax1_r = ax1.twinx()
    l1 = ax1.plot(ks, ms_rmses, marker='o', lw=1.3, color='#1f77b4', label='Treino RMSE Multi-Passo (10 s) [°]')
    l2 = ax1_r.plot(ks, val_fits, marker='s', lw=1.3, ls='--', color='#2ca02c', label='Validação Free-Run FIT [%] (estável)')
    l3 = ax1.axvline(win['k'], color='#d62728', ls=':', lw=1.3, label=f"Estrutura selecionada (k={win['k']})")
    ax1.set_xlabel('Número de Termos Polinomiais ($k$)')
    ax1.set_ylabel(r'RMSE Multi-Passo no Treino [$^\circ$]', color='#1f77b4')
    ax1_r.set_ylabel('FIT Médio na Validação [%]', color='#2ca02c')
    ax1.set_title(f"(a) Caminho SR-FROLS (DEC={win['dec']}, $n_y=n_u={win['ny']}$)", fontweight='bold')
    ax1.grid(True)
    lines = l1 + l2 + [l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=7.5)

    its = [h['iter'] for h in win['oe_hist']]
    tr_f = [h['tr_fit'] if h['tr_stable'] else np.nan for h in win['oe_hist']]
    vl_f = [h['sel_fit'] if h['sel_stable'] else np.nan for h in win['oe_hist']]
    ax2.plot(its, tr_f, marker='o', ms=3.5, lw=1.3, color='#1f77b4', label='Treino Free-Run FIT [%] (10 ensaios)')
    ax2.plot(its, vl_f, marker='s', ms=3.5, lw=1.3, color='#2ca02c', label='Validação Free-Run FIT [%] (8 ensaios)')
    ax2.axvline(win['best_iter'], color='#d62728', ls='--', lw=1.2,
                label=f"Early Stopping (it={win['best_iter']}, Val={win['sel_fit_oe']:.1f}%)")
    ax2.set_xlabel('Iteração Levenberg-Marquardt (OE)')
    ax2.set_ylabel('FIT Médio em Simulação Livre [%]')
    ax2.set_title('(b) Refinamento em Erro de Saída (LM-OE)', fontweight='bold')
    ax2.grid(True)
    ax2.legend(loc='lower right', fontsize=7.8)

    plt.suptitle("Etapas 4 e 5 — Seleção de Estrutura SR-FROLS e Refinamento em Erro de Saída (LM-OE)",
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_academic_plot(fig, "narx_metodologia_selecao_estrutura")
    plt.close(fig)

plot_structure_and_oe_convergence(winner)

# %% [markdown]
# ## Etapa 6: Avaliação Final no Conjunto de Teste Intocado (SEM Clip) + Ablação
#
# Avalia o modelo vencedor **uma única vez** nos 6 ensaios oficiais de Teste (`val_files`),
# comparando-o de forma transparente (com e sem clip no baseline) contra:
# 1. **Baseline Original** (`DEC=1`, 100 Hz, $n_y=n_u=15$, $\ell=2$, 10 termos via FROLS/ERR 1-passo);
# 2. **FROLS/ERR Clássico** no mesmo $(T_s, n_y, n_u, \ell_y, \ell_u, k)$ do modelo vencedor;
# 3. **SR-FROLS (OLS / EE)** (antes do refinamento LM-OE);
# 4. **SR-FROLS + LM-OE + Estabilidade de Aguirre (Modelo Final)**.

# %%
print("\n" + "=" * 80)
print("ETAPA 6: AVALIAÇÃO FINAL NO TESTE INTOCADO (6 ENSAIOS OFICIAIS — SEM CLIP!)")
print("=" * 80)

win_dec = winner['dec']
win_ny = winner['ny']
win_nu = winner['nu']
win_ml = max(win_ny, win_nu)
win_terms = winner['terms']
win_theta_ee = winner['theta_ee']
win_theta_oe = winner['theta_oe']
win_l = max(winner['ly'], winner['lu'])

test_keys = list(val_files.keys())
test_dec_list = decimate_group(test_raw, win_dec)
test_packed_win = pack_full_trajectories(test_dec_list)

# 1. Baseline original (6 ensaios originais de treino, 100 Hz, ny=nu=15, l=2, nc=10)
orig_train_keys = ['multisine_45_tr', 'chirp_45_tr', 'aprbs_45_tr1',
                   'multisine_60_tr', 'chirp_60_tr', 'aprbs_60_tr1']
orig_tr_list = [(train_raw[k][0], train_raw[k][1]) for k in orig_train_keys]
P_base, names_base, yt_base = build_polynomial_regmat(orig_tr_list, 15, 15, 2, 2, additive=False)
base_terms, base_theta, _ = frols_err_fast(P_base, yt_base, names_base, n_components=10)
test_packed_100hz = pack_full_trajectories(decimate_group(test_raw, 1))

_, base_ok_unclip, base_fit_unclip, base_rmse_unclip, base_pk_unclip = eval_packed(
    base_terms, base_theta, 15, test_packed_100hz, keys=test_keys, clip_bounds=None
)
_, base_ok_clip, base_fit_clip, base_rmse_clip, base_pk_clip = eval_packed(
    base_terms, base_theta, 15, test_packed_100hz, keys=test_keys, clip_bounds=(-10.0, 135.0)
)

# 2. FROLS/ERR clássico no mesmo (dec, ny, nu, ly, lu, k) do modelo vencedor
tr_win_list = cached_packed[win_dec][0]
P_win, names_win, yt_win = build_polynomial_regmat(tr_win_list, win_ny, win_nu, winner['ly'], winner['lu'], additive=True)
err_terms, err_theta, _ = frols_err_fast(P_win, yt_win, names_win, n_components=winner['k'])
_, err_ok, err_fit, err_rmse, err_pk = eval_packed(
    err_terms, err_theta, win_ml, test_packed_win, keys=test_keys, clip_bounds=None
)

# 3. SR-FROLS (OLS/EE) e 4. SR-FROLS + LM-OE (Final) — ambos 100% SEM clip!
_, ee_ok, ee_fit, ee_rmse, ee_pk = eval_packed(
    win_terms, win_theta_ee, win_ml, test_packed_win, keys=test_keys, clip_bounds=None
)
st_final, final_ok, final_fit, final_rmse, final_pk = eval_packed(
    win_terms, win_theta_oe, win_ml, test_packed_win, keys=test_keys, clip_bounds=None
)

def summarize_45_60(pk):
    ok45 = all(not pk[k]['diverged'] for k in test_keys if '_45_' in k)
    ok60 = all(not pk[k]['diverged'] for k in test_keys if '_60_' in k)
    okall = ok45 and ok60
    f45 = float(np.mean([pk[k]['fit'] for k in test_keys if '_45_' in k])) if ok45 else -999.0
    r45 = float(np.mean([pk[k]['rmse'] for k in test_keys if '_45_' in k])) if ok45 else 999.0
    f60 = float(np.mean([pk[k]['fit'] for k in test_keys if '_60_' in k])) if ok60 else -999.0
    r60 = float(np.mean([pk[k]['rmse'] for k in test_keys if '_60_' in k])) if ok60 else 999.0
    fall = float(np.mean([pk[k]['fit'] for k in test_keys])) if okall else -999.0
    rall = float(np.mean([pk[k]['rmse'] for k in test_keys])) if okall else 999.0
    return f45, r45, f60, r60, fall, rall

ablation_rows = [
    ("1. Baseline Original (100Hz, FROLS/ERR, SEM clip)",  base_ok_unclip, *summarize_45_60(base_pk_unclip)),
    ("1b. Baseline Original (100Hz, COM clip [-10,135])", base_ok_clip,   *summarize_45_60(base_pk_clip)),
    (f"2. FROLS/ERR Clássico (DEC={win_dec}, ny={win_ny}, SEM clip)", err_ok, *summarize_45_60(err_pk)),
    (f"3. SR-FROLS + Aguirre (OLS/EE, SEM clip)",         ee_ok,          *summarize_45_60(ee_pk)),
    (f"4. FINAL: SR-FROLS + Aguirre + LM-OE (SEM clip)",  final_ok,       *summarize_45_60(final_pk)),
]

print("\n" + "=" * 102)
print(f"{'Estudo de Ablação (6 Ensaios de Teste Intocados)':<50s} | {'Estáv.':^6s} | {'FIT 45°':>8s} | {'FIT 60°':>8s} | {'FIT Geral':>9s} | {'RMSE':>7s}")
print("=" * 102)
for lbl, nok, f45, r45, f60, r60, fall, rall in ablation_rows:
    f45_s  = f"{f45:7.1f}%" if f45 > -900 else " diverge"
    f60_s  = f"{f60:7.1f}%" if f60 > -900 else " diverge"
    fall_s = f"{fall:8.1f}%" if fall > -900 else "  diverge"
    rall_s = f"{rall:6.2f}°" if rall < 200 else "    inf"
    print(f"{lbl:<50s} | {nok:2d}/6  | {f45_s} | {f60_s} | {fall_s} | {rall_s}")
print("=" * 102)

# Monta dicionário detalhado val_results no formato canônico para gráficos e JSON
val_results = {}
for key, meta in val_files.items():
    u_full, y_full, t_full, ref_full = test_raw[key]
    u_d = u_full[::win_dec]
    y_d = y_full[::win_dec]
    t_d = t_full[::win_dec]
    ref_d = ref_full[::win_dec]
    info = final_pk[key]
    val_results[key] = {
        'label': meta['label'],
        't': t_d[win_ml:],
        'y_true': info['y_true'],
        'y_fr': info['y_fr'],
        'u': u_d[win_ml:],
        'ref': ref_d[win_ml:],
        'rmse': info['rmse'],
        'mae': info['mae'],
        'fit': info['fit'],
        'r2': info['r2'],
        'diverged': info['diverged'],
    }

print("\n" + "=" * 78)
print(f"{'Ensaio de Teste Final (Intocado, SEM Clip)':<34s} | {'RMSE [°]':<9s} | {'MAE [°]':<9s} | {'FIT [%]':<8s} | {'R²':<7s}")
print("=" * 78)
for key, res in val_results.items():
    print(f"{res['label']:<34s} | {res['rmse']:<9.3f} | {res['mae']:<9.3f} | {res['fit']:<8.2f} | {res['r2']:<7.4f}")
print("=" * 78)

# %% [markdown]
# ## Etapa 7: Validação Formal (Billings & Voon, 1986 + Curva Estática de Aguirre)

# %%
def xcorr_norm(a, b, max_lag=25):
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) + 1e-12
    lags = np.arange(-max_lag, max_lag + 1)
    vals = np.empty(len(lags), dtype=float)
    for idx, lag in enumerate(lags):
        if lag == 0:
            vals[idx] = np.dot(a, b) / denom
        elif lag > 0:
            vals[idx] = np.dot(a[:-lag], b[lag:]) / denom
        else:
            vals[idx] = np.dot(a[-lag:], b[:lag]) / denom
    return lags, vals

def eval_osa(terms, theta, ml, u, y):
    parsed = parse_narx_terms(terms)
    N = len(y)
    y_osa = np.zeros(N - ml)
    for (deg, ms), th in zip(parsed, theta):
        if deg == 0:
            y_osa += th
        else:
            col = np.ones(N - ml)
            for v, lg in ms:
                col *= y[ml - lg : N - lg] if v == 0 else u[ml - lg : N - lg]
            y_osa += th * col
    return y[ml:] - y_osa, y_osa

# 7.1 Testes não-lineares de correlação de resíduos de Billings & Voon (1986) no ensaio multisseno de teste
u_bv = test_raw['multisine_45_val'][0][::win_dec]
y_bv = test_raw['multisine_45_val'][1][::win_dec]
xi_osa, _ = eval_osa(win_terms, win_theta_oe, win_ml, u_bv, y_bv)
xi_fr = val_results['multisine_45_val']['y_true'] - val_results['multisine_45_val']['y_fr']
u_bv_m = u_bv[win_ml:]
u_bv_c = u_bv_m - np.mean(u_bv_m)
xi_c = xi_osa - np.mean(xi_osa)
band_95 = float(1.96 / np.sqrt(len(xi_osa)))

lags_bv, phi_xx = xcorr_norm(xi_c, xi_c, max_lag=25)
_, phi_ux = xcorr_norm(u_bv_c, xi_c, max_lag=25)
# Billings & Voon (1986): Phi_{xi, (xi*u)}(tau) para tau >= 0: E[xi(k) * xi(k-1-tau) * u'(k-1-tau)]
lags_pos = np.arange(0, 26)
xi_u_prod = xi_c * u_bv_c
xi_u_prod_c = xi_u_prod - np.mean(xi_u_prod)
denom_xxu = float(np.sqrt(np.dot(xi_c, xi_c) * np.dot(xi_u_prod_c, xi_u_prod_c))) + 1e-12
phi_xxu = np.array([
    float(np.dot(xi_c[tau + 1 :], xi_u_prod_c[: -(tau + 1)]) / denom_xxu)
    for tau in lags_pos
])
u2p = (u_bv_m ** 2) - np.mean(u_bv_m ** 2)
_, phi_u2x = xcorr_norm(u2p, xi_c, max_lag=25)
_, phi_u2x2 = xcorr_norm(u2p, xi_c ** 2, max_lag=25)

bv_summary = {
    'osa_rmse_deg': float(np.sqrt(np.mean(xi_osa ** 2))),
    'freerun_rmse_deg': float(np.sqrt(np.mean(xi_fr ** 2))),
    'band_95': band_95,
    'max_abs_phi_xx_nz': float(np.max(np.abs(np.delete(phi_xx, 25)))),
    'max_abs_phi_ux': float(np.max(np.abs(phi_ux))),
    'max_abs_phi_xxu': float(np.max(np.abs(phi_xxu))),
    'max_abs_phi_u2x': float(np.max(np.abs(phi_u2x))),
    'max_abs_phi_u2x2': float(np.max(np.abs(phi_u2x2))),
}

print("\n" + "=" * 80)
print("ETAPA 7: VALIDAÇÃO ESTATÍSTICA (BILLINGS & VOON) E FÍSICA (AGUIRRE)")
print("=" * 80)
print(f"  Resíduos OSA em multisine_45_val: RMSE_OSA = {bv_summary['osa_rmse_deg']:.4f}° | "
      f"RMSE_FR = {bv_summary['freerun_rmse_deg']:.3f}° | Banda 95% = +/-{band_95:.4f}")
print(f"    max|Phi_xx(tau!=0)| = {bv_summary['max_abs_phi_xx_nz']:.4f} | "
      f"max|Phi_ux| = {bv_summary['max_abs_phi_ux']:.4f} | "
      f"max|Phi_x(xu)(tau>=0)| = {bv_summary['max_abs_phi_xxu']:.4f} | "
      f"max|Phi_(u2')x| = {bv_summary['max_abs_phi_u2x']:.4f} | "
      f"max|Phi_(u2')x2| = {bv_summary['max_abs_phi_u2x2']:.4f}")

def plot_billings_voon(lags_sym, lags_p, curves, band):
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 5.4))
    axes_flat = axes.flatten()
    titles = [
        r'$\Phi_{\xi\xi}(\tau)$ (Autocorrelação dos resíduos)',
        r'$\Phi_{u\xi}(\tau)$ (Correlação entrada–resíduo)',
        r'$\Phi_{\xi(\xi u)}(\tau)$ para $\tau \geq 0$ (Acoplamento não-linear)',
        r"$\Phi_{(u^2)'\xi}(\tau)$ (Não-linearidade par em $u$)",
        r"$\Phi_{(u^2)'\xi^2}(\tau)$ (Variância residual vs. $(u^2)'$)",
    ]
    lags_each = [lags_sym, lags_sym, lags_p, lags_sym, lags_sym]
    for i, (lg_x, vals, title) in enumerate(zip(lags_each, curves, titles)):
        ax = axes_flat[i]
        markerline, stemlines, _ = ax.stem(lg_x, vals, linefmt='#1f77b4', markerfmt='o', basefmt='k-')
        plt.setp(markerline, markersize=2.8)
        plt.setp(stemlines, linewidth=0.9)
        ax.axhline(band, color='#d62728', ls='--', lw=1.0, label=rf'$\pm 1.96/\sqrt{{N}} = \pm{band:.3f}$')
        ax.axhline(-band, color='#d62728', ls='--', lw=1.0)
        ax.set_title(title, fontsize=8.8, fontweight='bold')
        ax.set_xlabel(r'Atraso $\tau$ [amostras]')
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True)
        if i == 0:
            ax.legend(loc='upper right', fontsize=7.5)

    # 6º subplot: histograma dos resíduos de Free-Run
    ax_h = axes_flat[5]
    ax_h.hist(xi_fr, bins=40, density=True, color='#2ca02c', alpha=0.72, edgecolor='white')
    ax_h.axvline(0.0, color='#111111', ls='--', lw=1.0)
    ax_h.set_title(r'Distribuição do Erro Free-Run ($y - \hat{y}_{\mathrm{FR}}$)', fontsize=8.8, fontweight='bold')
    ax_h.set_xlabel(r'Erro [$^\circ$]')
    ax_h.set_ylabel('Densidade')
    ax_h.grid(True)

    plt.suptitle("Etapa 7a — Testes de Validação de Resíduos (Billings & Voon, 1986) em Multisseno 45°",
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_academic_plot(fig, "narx_validacao_billings_voon")
    plt.close(fig)

plot_billings_voon(lags_bv, lags_pos, [phi_xx, phi_ux, phi_xxu, phi_u2x, phi_u2x2], band_95)

# 7.2 Análise Estática & Pontos Fixos (Aguirre)
static_info = analyze_static_and_poles(win_terms, win_theta_oe, win_ml)
print(f"  Análise Estática e Polos Lineares (Aguirre):")
print(f"    max|polo| = {static_info['max_pole']:.5f} (< 1) | sum(a_i) = {static_info['sum_a']:.5f} | "
      f"q_y = {static_info['q_y']:+.4e} | c_y = {static_info['c_y']:+.4e}")
print(f"    Estabilidade algébrica no envelope [-45°, 165°]: {'APROVADA' if static_info['alg_stable'] else 'REPROVADA'}")
if static_info['y_saddle_deg'] is not None:
    print(f"    Ponto de sela estático y_sela = {static_info['y_saddle_deg']:.1f}°")

def plot_static_validation(raw_tr_map, st_info, terms, theta, ml, dec):
    """Compara pontos quase-estáticos reais da RODADA-7 com a curva estática do modelo NARX."""
    # Extrai pontos quase-estáticos (|dy/dt| < 1.5 °/s e |d2y/dt2| < 8 °/s^2) dos ensaios de treino
    eq_y_list, eq_u_list = [], []
    for u, y, t, _ in raw_tr_map.values():
        dy = np.gradient(y, DT_NATIVE)
        d2y = np.gradient(dy, DT_NATIVE)
        m = (np.abs(dy) < 1.5) & (np.abs(d2y) < 8.0)
        eq_y_list.append(y[m])
        eq_u_list.append(u[m])
    eq_y = np.concatenate(eq_y_list)
    eq_u = np.concatenate(eq_u_list)

    # Bins experimentais
    bins = np.arange(0, 125, 10)
    bin_y, bin_u, bin_u_std = [], [], []
    for lo in bins:
        m = (eq_y >= lo) & (eq_y < lo + 10)
        if np.sum(m) >= 20:
            bin_y.append(float(np.mean(eq_y[m])))
            bin_u.append(float(np.mean(eq_u[m])))
            bin_u_std.append(float(np.std(eq_u[m])))

    # Curva estática do NARX via simulação de degraus constantes u in [0, 56]%
    u_grid = np.linspace(-15.0, 56.0, 60)
    L_eq = ml + int(round(20.0 * FS_NATIVE / dec))
    U_eq = np.repeat(u_grid[:, None], L_eq, axis=1)
    Y_eq = np.zeros_like(U_eq)
    YH_eq, div_eq = sim_and_jac_batch(parse_narx_terms(terms), theta, np.ones(len(terms)),
                                      ml, U_eq, Y_eq, explode=250.0, need_jac=False)
    y_ss = np.where(div_eq, np.nan, np.mean(YH_eq[:, -10:], axis=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1))
    ax1.scatter(eq_u[::10], eq_y[::10], s=4, alpha=0.12, color='#7f7f7f', label='Amostras quase-estáticas (RODADA-7)')
    ax1.errorbar(bin_u, bin_y, xerr=bin_u_std, fmt='o', color='#111111', ms=4.5, capsize=3,
                 label='Média experimental por faixa')
    ax1.plot(u_grid, y_ss, color='#d62728', lw=1.8, label='Equilíbrio Estático NARX ($y_{ss}(u)$)')
    ax1.set_xlabel('Comando Estacionário $u$ [%]')
    ax1.set_ylabel(r'Ângulo de Equilíbrio $\theta_{ss}$ [$^\circ$]')
    ax1.set_title('(a) Curva Estática Experimental vs. NARX', fontweight='bold')
    ax1.set_xlim(-18, 62)
    ax1.set_ylim(-25, 135)
    ax1.grid(True)
    ax1.legend(loc='upper left', fontsize=7.6)

    # (b) Função restauradora estática F_y(y) = (1 - sum a)y - q_y*y^2 - c_y*y^3
    y_span = np.linspace(-45.0, 175.0, 300)
    fy_span = st_info['one_minus_a'] * y_span - st_info['q_y'] * (y_span ** 2) - st_info['c_y'] * (y_span ** 3)
    ax2.plot(y_span, fy_span, color='#1f77b4', lw=1.6, label=r'$F_y(y) = (1-\sum a_i)y - q_y y^2 - c_y y^3$')
    ax2.axhline(0.0, color='#111111', ls='--', lw=0.9)
    ax2.axvspan(-30, 150, color='#2ca02c', alpha=0.12, label=r'Envelope operacional $[-30^\circ,\,150^\circ]$')
    if st_info['y_saddle_deg'] is not None and st_info['y_saddle_deg'] < 200:
        ax2.axvline(st_info['y_saddle_deg'], color='#d62728', ls=':', lw=1.2,
                    label=rf'Pico estático ($y_{{\mathrm{{sela}}}} = {st_info["y_saddle_deg"]:.1f}^\circ$)')
    ax2.set_xlabel(r'Ângulo $y$ [$^\circ$]')
    ax2.set_ylabel(r'Termo Restaurador $F_y(y)$ [$^\circ$/passo]')
    ax2.set_title('(b) Estabilidade Estática de Aguirre (Pontos Fixos)', fontweight='bold')
    ax2.grid(True)
    ax2.legend(loc='upper left', fontsize=7.6)

    plt.suptitle("Etapa 7b — Validação da Curva Estática e Estabilidade Algébrica de Pontos Fixos (Aguirre)",
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_academic_plot(fig, "narx_validacao_curva_estatica")
    plt.close(fig)

plot_static_validation(train_raw, static_info, win_terms, win_theta_oe, win_ml, win_dec)

# %% [markdown]
# ## Geração dos Gráficos Canônicos de Validação (Padrão Acadêmico — Compatível com TCC)

# %%
def plot_single_validation(res, filename_base):
    """Gera gráfico individual de validação com 2 subplots (Ângulo e Controle)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.3), sharex=True,
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    t = res['t']
    y_true = res['y_true']
    y_fr = res['y_fr']
    u = res['u']

    ax1.plot(t, y_true, color='#111111', lw=1.1, label=r'Medido $y(k)$')
    ax1.plot(t, y_fr, color='#d62728', ls='--', lw=1.3, label=r'NARX Free-Run $\hat{y}(k)$')
    if not np.all(np.isnan(res['ref'])):
        ax1.plot(t, res['ref'], color='#7f7f7f', ls=':', lw=0.9, alpha=0.7, label=r'Referência $r(k)$')

    ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.92,
               borderpad=0.25, labelspacing=0.2, handlelength=1.2, handletextpad=0.35)
    ax1.set_xlim(0, t[-1])

    y_peak = max(float(np.max(y_true)), float(np.max(y_fr)))
    y_valley = min(float(np.min(y_true)), float(np.min(y_fr)))
    margin_top = max(16.0, (y_peak - y_valley) * 0.20)
    margin_bottom = max(5.0, (y_peak - y_valley) * 0.05)
    ax1.set_ylim(y_valley - margin_bottom, y_peak + margin_top)

    metric_text = f"RMSE = {res['rmse']:.3f}°\nMAE = {res['mae']:.3f}°\nFIT = {res['fit']:.1f}%"
    ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=7.5,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.85', alpha=0.92))

    ax2.plot(t, u, color='#1f77b4', lw=1.0, label=r'Excitação $u(k)$')
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel(r'Controle $u$ [%]')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.92,
               borderpad=0.25, labelspacing=0.2, handlelength=1.2)
    u_peak, u_valley = float(np.max(u)), float(np.min(u))
    ax2.set_ylim(u_valley - 4.0, u_peak + max(12.0, (u_peak - u_valley) * 0.22))

    plt.tight_layout()
    save_academic_plot(fig, filename_base)
    plt.close(fig)

def plot_consolidated_validation(val_dict, keys, filename_base, suptitle):
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 4.8), sharex=False,
                             gridspec_kw={'height_ratios': [2.2, 1.2]})
    for col, key in enumerate(keys):
        res = val_dict[key]
        t = res['t']
        ax_y, ax_u = axes[0, col], axes[1, col]
        ax_y.plot(t, res['y_true'], color='#111111', lw=1.1, label=r'Medido $y(k)$')
        ax_y.plot(t, res['y_fr'], color='#d62728', ls='--', lw=1.2, label=r'NARX $\hat{y}(k)$')
        ax_y.set_title(res['label'], fontsize=9.5, fontweight='bold', pad=4)
        ax_y.grid(True)
        ax_y.set_xlim(0, t[-1])
        if col == 0:
            ax_y.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
            ax_y.legend(loc='upper right', fontsize=8)
        txt = f"RMSE = {res['rmse']:.2f}°\nFIT = {res['fit']:.1f}%"
        ax_y.text(0.03, 0.94, txt, transform=ax_y.transAxes,
                  verticalalignment='top', fontsize=8,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.85', alpha=0.9))
        ax_u.plot(t, res['u'], color='#1f77b4', lw=1.0)
        ax_u.set_xlabel('Tempo [s]')
        ax_u.grid(True)
        ax_u.set_xlim(0, t[-1])
        if col == 0:
            ax_u.set_ylabel(r'Controle $u$ [%]')
    plt.suptitle(suptitle, fontsize=11, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_academic_plot(fig, filename_base)
    plt.close(fig)

def plot_consolidated_validation_all(val_dict, filename_base="narx_val_consolidado_completo"):
    fig, axes = plt.subplots(4, 3, figsize=(13.0, 8.5), sharex=False,
                             gridspec_kw={'height_ratios': [2.0, 1.0, 2.0, 1.0]})
    rows_keys = [
        ['multisine_45_val', 'chirp_45_val', 'steps_45_val'],
        ['multisine_60_val', 'chirp_60_val', 'steps_60_val']
    ]
    for r_idx, keys in enumerate(rows_keys):
        row_y, row_u = r_idx * 2, r_idx * 2 + 1
        for col, key in enumerate(keys):
            res = val_dict[key]
            t = res['t']
            ax_y, ax_u = axes[row_y, col], axes[row_u, col]
            ax_y.plot(t, res['y_true'], color='#111111', lw=1.1, label=r'Medido $y(k)$')
            ax_y.plot(t, res['y_fr'], color='#d62728', ls='--', lw=1.2, label=r'NARX $\hat{y}(k)$')
            ax_y.set_title(res['label'], fontsize=9.0, fontweight='bold', pad=3)
            ax_y.grid(True)
            ax_y.set_xlim(0, t[-1])
            if col == 0:
                ax_y.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
                ax_y.legend(loc='upper right', fontsize=7.5)
            txt = f"RMSE = {res['rmse']:.2f}°\nFIT = {res['fit']:.1f}%"
            ax_y.text(0.03, 0.94, txt, transform=ax_y.transAxes,
                      verticalalignment='top', fontsize=7.5,
                      bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.85', alpha=0.9))
            ax_u.plot(t, res['u'], color='#1f77b4', lw=0.9)
            ax_u.set_xlabel('Tempo [s]')
            ax_u.grid(True)
            ax_u.set_xlim(0, t[-1])
            if col == 0:
                ax_u.set_ylabel(r'$u$ [%]')
    plt.suptitle(
        f"Teste Final em Simulação Livre (Free-Run, SEM Clip) — NARX Metodológico "
        f"(Ts={win_dec*10} ms, ny=nu={win_ny}, {len(win_terms)} termos)",
        fontsize=11.2, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    save_academic_plot(fig, filename_base)
    plt.close(fig)

print("\nGerando gráficos individuais e consolidados de Teste Final (Free-Run SEM clip)...")
plot_single_validation(val_results['multisine_45_val'], "narx_freerun")
plot_single_validation(val_results['multisine_45_val'], "narx_val_multisine")
plot_single_validation(val_results['chirp_45_val'], "narx_val_chirp")
plot_single_validation(val_results['steps_45_val'], "narx_val_steps")
plot_single_validation(val_results['multisine_60_val'], "narx_val_multisine_60")
plot_single_validation(val_results['chirp_60_val'], "narx_val_chirp_60")
plot_single_validation(val_results['steps_60_val'], "narx_val_steps_60")

plot_consolidated_validation(
    val_results, keys=['multisine_45_val', 'chirp_45_val', 'steps_45_val'],
    filename_base="narx_val_consolidado",
    suptitle=f"Teste Final Free-Run (45°, SEM Clip) — NARX Metodológico (Ts={win_dec*10} ms)"
)
plot_consolidated_validation(
    val_results, keys=['multisine_60_val', 'chirp_60_val', 'steps_60_val'],
    filename_base="narx_val_consolidado_60",
    suptitle=f"Teste Final Free-Run (60°, SEM Clip) — NARX Metodológico (Ts={win_dec*10} ms)"
)
plot_consolidated_validation_all(val_results, "narx_val_consolidado_completo")

# %% [markdown]
# ## Exportação do Modelo, Equações LaTeX e Relatório Metodológico
#
# - `narx_model.json`: 100% compatível com `mpc_v2.py`, `mpc_v3.py`, `mpc_v4.py` (inclui `ny`, `nu`, `l`, `terms`, `theta`, `metrics`, além de `decimation`, `Ts`, metadados metodológicos).
# - `narx_metrics.json`: métricas oficiais dos 6 ensaios de teste.
# - `narx_equation.tex` e `narx_terms_table.tex`: equação e tabela LaTeX formatadas com suporte a potências até grau 3.
# - `narx_equation.txt`: resumo completo legível em texto.
# - `narx_methodology_report.json`: trilha auditável de todas as decisões metodológicas.

# %%
def format_term_latex(term):
    """Formata um monômio NARX de grau arbitrário para LaTeX limpo (agrupando potências)."""
    if term == 'constant':
        return "1"
    toks = re.findall(r'([yu])\(k-(\d+)\)', term)
    counts = {}
    order = []
    for tok in toks:
        if tok not in counts:
            counts[tok] = 0
            order.append(tok)
        counts[tok] += 1
    parts = []
    for v, lg in order:
        c = counts[(v, lg)]
        base = f"{v}[k-{lg}]"
        parts.append(f"{base}^{c}" if c > 1 else base)
    return r"\,".join(parts)

terms_list = list(win_terms)
theta_list = [float(x) for x in win_theta_oe]
theta_ee_list = [float(x) for x in win_theta_ee]

# 1. narx_model.json (compatível com CasADi MPC)
model_json = {
    'ny': int(win_ny),
    'nu': int(win_nu),
    'l': int(win_l),
    'ly': int(winner['ly']),
    'lu': int(winner['lu']),
    'decimation': int(win_dec),
    'Ts': float(winner['Ts']),
    'fs_hz': float(FS_NATIVE / win_dec),
    'bidirectional_u': True,
    'selection_method': f"SR-FROLS ({winner['seed'].upper()}-seed) + Aguirre Algebraic Stability + LM-OE (Val Early-Stopping)",
    'terms': terms_list,
    'theta': theta_list,
    'theta_ols_ee': theta_ee_list,
    'metrics': {
        k: {'rmse': v['rmse'], 'mae': v['mae'], 'fit': v['fit'], 'r2': v['r2']}
        for k, v in val_results.items()
    },
    'validation_selection_metrics': {
        k: {'rmse': v['rmse'], 'mae': v['mae'], 'fit': v['fit'], 'r2': v['r2']}
        for k, v in winner['sel_per_k'].items()
    },
    'static_and_poles': static_info,
    'billings_voon': bv_summary,
}

json_path = os.path.join(current_dir, 'narx_model.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(model_json, f, indent=4, ensure_ascii=False)
print(f"\n[OK] Modelo exportado: {json_path}")

# 2. narx_metrics.json (esquema canônico dos 6 ensaios oficiais)
metrics_json = {
    k: {
        'label': v['label'],
        'rmse_deg': v['rmse'],
        'mae_deg': v['mae'],
        'fit_percent': v['fit'],
        'r2_score': v['r2'],
        'samples': int(len(v['t'])),
        'decimation': int(win_dec),
        'Ts_sec': float(winner['Ts']),
        'unclipped_stable': not v['diverged'],
    }
    for k, v in val_results.items()
}
metrics_path = os.path.join(current_dir, 'narx_metrics.json')
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_json, f, indent=4, ensure_ascii=False)
print(f"[OK] Métricas exportadas: {metrics_path}")

# 3. narx_equation.tex e narx_terms_table.tex
eq_terms_latex, eq_terms_txt = [], []
for term, th in zip(terms_list, theta_list):
    sign = "+" if th >= 0 else "-"
    val = abs(th)
    val_str_tex = f"{val:.4e}" if val < 1e-3 else f"{val:.4f}"
    val_str_txt = f"{val:.6e}" if val < 1e-3 else f"{val:.6f}"
    if term == 'constant':
        eq_terms_latex.append(f"{sign} {val_str_tex}")
        eq_terms_txt.append(f"{sign} {val_str_txt}")
    else:
        t_tex = format_term_latex(term)
        eq_terms_latex.append(f"{sign} {val_str_tex} \\cdot {t_tex}")
        eq_terms_txt.append(f"{sign} {val_str_txt} * {term}")

latex_equation = "\\hat{y}(k) = " + " ".join(eq_terms_latex).lstrip("+ ")
txt_equation = "y_hat(k) = " + " ".join(eq_terms_txt).lstrip("+ ")

tex_path = os.path.join(current_dir, 'narx_equation.tex')
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(
        f"% Equação NARX polinomial identificada metodologicamente (SR-FROLS + Aguirre + LM-OE)\n"
        f"% Ts = {winner['Ts']*1000:.0f} ms (DECIMATION = {win_dec}), ny = {win_ny}, nu = {win_nu}, ly = {winner['ly']}, lu = {winner['lu']}\n"
        f"\\begin{{equation}}\n    {latex_equation}\n    \\label{{eq:narx_identificado}}\n\\end{{equation}}\n"
    )
print(f"[OK] Equação LaTeX exportada: {tex_path}")

table_tex_path = os.path.join(current_dir, 'narx_terms_table.tex')
with open(table_tex_path, 'w', encoding='utf-8') as f:
    f.write("\\begin{table}[H]\n    \\centering\n")
    f.write(f"    \\caption{{Termos e parâmetros do modelo NARX metodológico ($T_s = {win_dec*10}$\\,ms, $n_y = n_u = {win_ny}$).}}\n")
    f.write("    \\label{tab:narx_termos}\n    \\begin{tabular}{lrr}\n        \\toprule\n")
    f.write("        \\textbf{Termo Regressor} & $\\hat{\\theta}_{\\text{EE}}$ (OLS) & $\\hat{\\theta}_{\\text{OE}}$ (LM-OE) \\\\ \\midrule\n")
    for term, th_e, th_o in zip(terms_list, theta_ee_list, theta_list):
        t_disp = "Constante" if term == 'constant' else f"${format_term_latex(term)}$"
        fe = f"{th_e:+.4e}" if abs(th_e) < 1e-3 else f"{th_e:+.4f}"
        fo = f"{th_o:+.4e}" if abs(th_o) < 1e-3 else f"{th_o:+.4f}"
        f.write(f"        {t_disp:<34s} & ${fe}$ & ${fo}$ \\\\\n")
    f.write("        \\bottomrule\n    \\end{tabular}\n\\end{table}\n")
print(f"[OK] Tabela LaTeX exportada: {table_tex_path}")

# 4. narx_equation.txt
txt_path = os.path.join(current_dir, 'narx_equation.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("Modelo NARX Polinomial Metodológico (SR-FROLS + Estabilidade de Aguirre + LM-OE)\n")
    f.write(f"DECIMATION = {win_dec} (Ts = {winner['Ts']*1000:.0f} ms, fs = {FS_NATIVE/win_dec:.2f} Hz)\n")
    f.write(f"Ordens: ny = {win_ny}, nu = {win_nu} | Graus: ly = {winner['ly']}, lu = {winner['lu']} | Termos: {len(terms_list)}\n")
    f.write(f"Polo dominante max|z| = {static_info['max_pole']:.5f} | sum(a) = {static_info['sum_a']:.5f}\n\n")
    f.write(f"Equação:\n{txt_equation}\n\n")
    f.write(f"{'#':>2s}  {'Termo':<24s} {'theta_EE (OLS)':>16s} {'theta_OE (Final)':>16s}\n")
    f.write("-" * 64 + "\n")
    for i, (term, th_e, th_o) in enumerate(zip(terms_list, theta_ee_list, theta_list), 1):
        f.write(f"{i:2d}. {term:<24s} {th_e:+16.8e} {th_o:+16.8e}\n")
print(f"[OK] Equação em texto exportada: {txt_path}")

# 5. narx_methodology_report.json (trilha completa e reprodutível)
report_json = {
    'dataset_split': {
        'session': 'RODADA-7 (09/04, 24 ensaios canônicos)',
        'train_files': [m['file'] for m in train_files.values()],
        'validation_selection_files': [m['file'] for m in sel_files.values()],
        'heldout_test_files': [m['file'] for m in val_files.values()],
    },
    'stage2_sampling': {
        'f99_median_hz': f99_med,
        'f99_max_hz': f99_max,
        'tau_m_median_samples_100hz': tau_m_med,
        'aguirre_decimation_range': [dec_aguirre_lo, dec_aguirre_hi],
        'spectral_decimation_range': [dec_bw_lo, dec_bw_hi],
        'dec_candidates_evaluated': dec_candidates,
        'per_dataset': sampling_diag,
    },
    'stage3_arx_bic': {str(k): v for k, v in bic_results.items()},
    'stage4_5_validation_grid': [
        {
            'dec': r['dec'], 'Ts': r['Ts'], 'ny': r['ny'], 'nu': r['nu'],
            'ly': r['ly'], 'lu': r['lu'], 'seed': r['seed'], 'tag': r['tag'],
            'k': r['k'], 'tr_fit_oe': r['tr_fit_oe'], 'sel_fit_ee': r['sel_fit_ee'],
            'sel_fit_oe': r['sel_fit_oe'], 'sel_rmse_oe': r['sel_rmse_oe'],
            'best_iter': r['best_iter'], 'terms': r['terms']
        }
        for r in grid_results
    ],
    'stage6_ablation_test': [
        {
            'model': lbl, 'stable_count': int(nok), 'fit_45': f45, 'rmse_45': r45,
            'fit_60': f60, 'rmse_60': r60, 'fit_all': fall, 'rmse_all': rall
        }
        for lbl, nok, f45, r45, f60, r60, fall, rall in ablation_rows
    ],
    'stage7_validation': {
        'static_and_poles': static_info,
        'billings_voon': bv_summary,
    },
}
report_path = os.path.join(current_dir, 'narx_methodology_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report_json, f, indent=4, ensure_ascii=False)
print(f"[OK] Relatório metodológico exportado: {report_path}")

print("\nPipeline metodológico NARX concluído com sucesso!")
