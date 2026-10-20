# %% [markdown]
# # Identificação do Modelo Não-Linear NARX (Aeropêndulo)
# 
# Modelo NARX (*Nonlinear AutoRegressive with eXogenous inputs*) polinomial para a planta
# de 1 GDL (aeropêndulo) utilizando o algoritmo FROLS (*Forward Regression Orthogonal Least Squares*)
# com critério de seleção por ERR (*Error Reduction Ratio*).
#
# Gera gráficos e relatórios no padrão acadêmico (IEEE / PUCPR / ABNT), salvando automaticamente
# as figuras em alta resolução (.png 300 DPI e .pdf vetorial) para uso no TCC / artigo.

# %%
import os
import sys
import json
import re
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS E IMPORTAÇÕES
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Diretórios de saída para salvar os gráficos
plots_dirs = [
    os.path.join(current_dir, 'plots')
]

tcc_img_dir = os.path.join(root_dir, 'tcc', 'img', 'resultados')
if os.path.exists(tcc_img_dir):
    plots_dirs.append(tcc_img_dir)

overleaf_img_dir = os.path.join(root_dir, 'img', 'resultados')
if os.path.exists(overleaf_img_dir):
    plots_dirs.append(overleaf_img_dir)

for d in plots_dirs:
    os.makedirs(d, exist_ok=True)

# Importa aerodata
try:
    from aerodata import readData
except ImportError:
    sys.path.append(root_dir)
    from aerodata import readData

# Instala/importa a biblioteca de identificação sysid
try:
    from sysid import NARX
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/helonayala/sysid.git'])
    from sysid import NARX

# ==============================================================================
# PADRONIZAÇÃO VISUAL ACADÊMICA (IEEE / ABNT / PUCPR)
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
    """Salva a figura simultaneamente em PNG (300 DPI) e PDF vetorial em todas as pastas alvo."""
    for d in dirs:
        png_path = os.path.join(d, f"{filename_base}.png")
        pdf_path = os.path.join(d, f"{filename_base}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  -> Gráfico salvo: {filename_base} (.png e .pdf) em {len(dirs)} pastas.")

# %% [markdown]
# ## Carregamento e Pré-Processamento dos Datasets
# 
# ### Metodologia de Coleta e Seleção dos Sinais Experimentais:
# - **Riqueza espectral dos sinais:** Necessária para identificar a dinâmica não-linear da planta
#   (empuxo quadrático do rotor $\propto u^2$, atraso de transporte, amortecimento viscoso e torque gravitacional $\propto \sin\theta$).
# - **Frequência máxima definida:** $f_{\max} = 0{,}50$\,Hz, estabelecida com base na largura de banda mecânica
#   do aeropêndulo e na dinâmica do conjunto motor-hélice, evitando vibrações estruturais parasitas.
# - **Dois níveis de estabilização em malha fechada:** Operação em torno de $45^\circ$ e $60^\circ$, estabilizada
#   pelo controlador PID preliminar embarcado no STM32 para garantia de integridade mecânica.
# - **Envelope operacional de referência:** Faixa total definida de $0^\circ$ a $120^\circ$.
# - **Critério de divisão treino vs. validação:**
#   - **Treinamento (6 ensaios):** Foram selecionados os sinais com maior frequência ($f = 0{,}50$\,Hz) e
#     maior amplitude (Chirp $\pm 45^\circ$ em $45^\circ$, Chirp $\pm 60^\circ$ em $60^\circ$, e sequências APRBS),
#     garantindo máxima cobertura do espaço de estados e das não-linearidades.
#   - **Validação e Free-Run (6 ensaios):** Os sinais restantes de menor frequência ($0{,}30$\,Hz), menor amplitude
#     ($\pm 35^\circ$ em $45^\circ$, $\pm 40^\circ$ em $60^\circ$) e múltiplos degraus foram reservados para teste
#     independente de interpolação e predição recursiva livre sem realimentação.

# %%
DECIMATION = 1          # Decimação (1 = 100 Hz original)
TRIM_START_SEC = 10.0   # Descarta primeiros 10 segundos (transitório de partida)
TRIM_END_SEC = 13.0     # Descarta últimos 13 segundos (parada da telemetria)

def load_processed(fname, trim_start=TRIM_START_SEC, trim_end=TRIM_END_SEC, decimation=DECIMATION):
    """Carrega dataset via aerodata, corta margens transitórias e normaliza o vetor de tempo."""
    y, u, t, ref = readData(dataset_name=fname, return_ref=True)
    
    t_start_target = t[0] + trim_start
    t_end_target = t[-1] - trim_end
    
    idx_start = np.searchsorted(t, t_start_target)
    idx_end = np.searchsorted(t, t_end_target)
    
    if idx_start >= idx_end:
        idx_start, idx_end = 0, len(t)
        
    sl = slice(idx_start, idx_end, decimation)
    u_c, y_c, t_c = u[sl], y[sl], t[sl]
    
    # Reseta o tempo para iniciar exatamente em t = 0
    t_c = t_c - t_c[0]
    
    if len(ref) > 0:
        ref_c = ref[sl]
    else:
        ref_c = np.full_like(t_c, np.nan)
        
    return u_c, y_c, t_c, ref_c

# Arquivos de Treinamento (6 ensaios: máxima frequência 0.50 Hz e maior amplitude em 45° e 60°)
train_files = {
    'multisine_45_train': {
        'file': 'RODADA-7/multi-seno-45-050Hz_0904_20-29.csv',
        'label': 'Multisseno (45°, 0.50 Hz)'
    },
    'chirp_45_train': {
        'file': 'RODADA-7/chirp-45-amp45_0904_20-15.csv',
        'label': 'Chirp (45°, Amp ±45°)'
    },
    'steps_45_train': {
        'file': 'RODADA-7/aprbs-45-1_0904_19-54.csv',
        'label': 'APRBS (45°)'
    },
    'multisine_60_train': {
        'file': 'RODADA-7/multi-seno-60-050Hz_0904_20-38.csv',
        'label': 'Multisseno (60°, 0.50 Hz)'
    },
    'chirp_60_train': {
        'file': 'RODADA-7/chirp-60-amp60_0904_20-20.csv',
        'label': 'Chirp (60°, Amp ±60°)'
    },
    'steps_60_train': {
        'file': 'RODADA-7/aprbs-60-1_0904_20-04.csv',
        'label': 'APRBS (60°)'
    }
}

train_datasets = {
    key: load_processed(meta['file'])
    for key, meta in train_files.items()
}

# Arquivos de Validação e Testes Free-Run (Sinais restantes: menor frequência e menor amplitude em 45° e 60°)
val_files = {
    'multisine_45_val': {
        'file': 'RODADA-7/multi-seno-45-030Hz_0904_20-23.csv',
        'label': 'Multisseno (45°, 0.30 Hz)'
    },
    'chirp_45_val': {
        'file': 'RODADA-7/chirp-45-amp35_0904_20-13.csv',
        'label': 'Chirp (45°, Amp ±35°)'
    },
    'steps_45_val': {
        'file': 'RODADA-7/seq-degraus-45-2_0904_20-45.csv',
        'label': 'Sequência de Degraus (45°)'
    },
    'multisine_60_val': {
        'file': 'RODADA-7/multi-seno-60-030Hz_0904_20-32.csv',
        'label': 'Multisseno (60°, 0.30 Hz)'
    },
    'chirp_60_val': {
        'file': 'RODADA-7/chirp-60-amp40_0904_20-16.csv',
        'label': 'Chirp (60°, Amp ±40°)'
    },
    'steps_60_val': {
        'file': 'RODADA-7/seq-degraus-60-2_0904_20-54.csv',
        'label': 'Sequência de Degraus (60°)'
    }
}

val_datasets = {
    key: load_processed(meta['file'])
    for key, meta in val_files.items()
}

# %% [markdown]
# ## Gráfico Padronizado dos Sinais de Treinamento
# 
# Painel exibindo a riqueza de excitação dos 6 ensaios selecionados para treinar o modelo NARX
# (frequência máxima de 0.50 Hz e maior excursão angular em 45° e 60°).

# %%
def plot_training_datasets_grid(train_ds, meta_dict):
    """Gera um painel com os 6 conjuntos de treino (Ângulo em cima, Controle em baixo)."""
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 7.5), sharex=False)
    axes_flat = axes.flatten()
    
    for i, (key, meta) in enumerate(meta_dict.items()):
        ax = axes_flat[i]
        u, y, t, ref = train_ds[key]
        
        # Subplot duplo via twinx ou dois eixos
        ax.plot(t, y, color='#1f77b4', lw=1.1, label=r'Ângulo $\theta$')
        if not np.all(np.isnan(ref)):
            ax.plot(t, ref, color='#d62728', ls='--', lw=1.0, alpha=0.8, label=r'Ref $r$')
        ax.set_ylabel(r'$\theta$ [$^\circ$]', color='#1f77b4')
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        ax.set_title(meta['label'], fontsize=9.5, fontweight='bold', pad=4)
        ax.grid(True)
        
        ax_u = ax.twinx()
        ax_u.plot(t, u, color='#2ca02c', lw=0.9, alpha=0.75, label=r'Controle $u$')
        ax_u.set_ylabel(r'$u$ [\%]', color='#2ca02c')
        ax_u.tick_params(axis='y', labelcolor='#2ca02c')
        ax.set_xlabel('Tempo [s]')
        
        if i == 0:
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax_u.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=8)

    plt.suptitle("Conjunto de Ensaios Experimentais para Treinamento do Modelo NARX (0.50 Hz / Máx. Amplitude)", fontsize=11, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_academic_plot(fig, "narx_train_datasets")
    plt.close(fig)

print("Gerando gráfico dos sinais de treinamento...")
plot_training_datasets_grid(train_datasets, train_files)

# %% [markdown]
# ## Treinamento do Modelo NARX (FROLS / ERR)
# 
# - Ordens de memória: $n_y = 15$, $n_u = 15$ (150 ms)
# - Grau polinomial: $\ell = 2$
# - Critério de parada: 10 componentes selecionados

# %%
train_data = []
for name, d in train_datasets.items():
    u, y, t, ref = d
    train_data.append((u, y))

ny_model = 15
nu_model = 15
poly_order_model = 2
n_components = 10

print(f"\nTreinando modelo NARX (ny={ny_model}, nu={nu_model}, grau={poly_order_model}, componentes={n_components})...")
narx_model = NARX(nu=nu_model, ny=ny_model, poly_order_l=poly_order_model, n_components=n_components)
narx_model.fit(train_data)
narx_model.print()

# %% [markdown]
# ## Validação em Simulação Livre (*Free-Run*)
# 
# Avaliação recursiva sem realimentação dos estados medidos para verificar estabilidade e fidelidade dinâmica.

# %%
ml = narx_model._max_lag_internal_

def compute_metrics(y_true, y_pred):
    """Calcula métricas quantitativas de ajuste (RMSE, MAE, FIT%)."""
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    denom = np.linalg.norm(y_true - np.mean(y_true))
    fit = 100.0 * (1.0 - np.linalg.norm(err) / denom) if denom > 1e-6 else 0.0
    r2 = 1.0 - (np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)) if denom > 1e-6 else 0.0
    return rmse, mae, fit, r2

def compile_fast_evaluator(model):
    """Compila os termos selecionados do modelo NARX para avaliação ultrarrápida (O(1))."""
    import re
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

def simulate_freerun_physical(model, u_in, y_init, bounds=(-10.0, 135.0)):
    """Simulação livre (free-run) recursiva ultra-otimizada com saturação física [-10°, 135°]."""
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
        y_hat[k] = val
        
    return y_hat[ml:]

# Executa as simulações em free-run e compila os resultados
val_results = {}
for key, meta in val_files.items():
    u, y, t, ref = val_datasets[key]
    print(f"Simulando Free-Run para '{meta['label']}' ({len(t) - ml} passos)...")
    y_fr = simulate_freerun_physical(narx_model, u, y, bounds=(-10.0, 135.0))
    
    t_val = t[ml:]
    y_true = y[ml:]
    u_val = u[ml:]
    ref_val = ref[ml:]
    
    rmse, mae, fit, r2 = compute_metrics(y_true, y_fr)
    val_results[key] = {
        'label': meta['label'],
        't': t_val,
        'y_true': y_true,
        'y_fr': y_fr,
        'u': u_val,
        'ref': ref_val,
        'rmse': float(rmse),
        'mae': float(mae),
        'fit': float(fit),
        'r2': float(r2)
    }

# Tabela no console
print("\n" + "="*70)
print(f"{'Ensaio de Validação':<32} | {'RMSE [°]':<10} | {'MAE [°]':<10} | {'FIT [%]':<10}")
print("="*70)
for key, res in val_results.items():
    print(f"{res['label']:<32} | {res['rmse']:<10.3f} | {res['mae']:<10.3f} | {res['fit']:<10.2f}")
print("="*70)

# %% [markdown]
# ## Geração dos Gráficos de Validação em Padrão Acadêmico

# %%
def plot_single_validation(res, filename_base):
    """Gera gráfico individual de validação com 2 subplots (Ângulo e Controle)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.3), sharex=True,
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    t = res['t']
    y_true = res['y_true']
    y_fr = res['y_fr']
    u = res['u']
    
    # Subplot Superior: Ângulo
    ax1.plot(t, y_true, color='#111111', lw=1.1, label=r'Medido $y(k)$')
    ax1.plot(t, y_fr, color='#d62728', ls='--', lw=1.3, label=r'NARX Free-Run $\hat{y}(k)$')
    if not np.all(np.isnan(res['ref'])):
        ax1.plot(t, res['ref'], color='#7f7f7f', ls=':', lw=0.9, alpha=0.7, label=r'Referência $r(k)$')
        
    ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
    ax1.grid(True)
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.set_xlim(0, t[-1])
    
    # Caixa com métricas
    metric_text = (f"RMSE = {res['rmse']:.3f}" + r"$^\circ$" + "\n"
                   f"MAE = {res['mae']:.3f}" + r"$^\circ$" + "\n"
                   f"FIT = {res['fit']:.1f}\\%")
    ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.8', alpha=0.92))
    
    # Subplot Inferior: Controle
    ax2.plot(t, u, color='#1f77b4', lw=1.0, label=r'Excitação $u(k)$')
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel(r'Controle $u$ [\%]')
    ax2.grid(True)
    ax2.legend(loc='upper right', framealpha=0.95)
    
    plt.tight_layout()
    save_academic_plot(fig, filename_base)
    plt.close(fig)

# Gera os plots individuais para todos os ensaios de validação
plot_single_validation(val_results['multisine_45_val'], "narx_freerun")
plot_single_validation(val_results['multisine_45_val'], "narx_val_multisine")
plot_single_validation(val_results['chirp_45_val'], "narx_val_chirp")
plot_single_validation(val_results['steps_45_val'], "narx_val_steps")

if 'multisine_60_val' in val_results:
    plot_single_validation(val_results['multisine_60_val'], "narx_val_multisine_60")
    plot_single_validation(val_results['chirp_60_val'], "narx_val_chirp_60")
    plot_single_validation(val_results['steps_60_val'], "narx_val_steps_60")

# %% [markdown]
# ## Painéis Consolidados de Validação
# 
# Figuras unificadas em padrão acadêmico para o TCC / Monografia.

# %%
def plot_consolidated_validation(val_dict, keys, filename_base, suptitle):
    """Gera figura unificada com 3 colunas comparando ensaios de validação."""
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 4.8), sharex=False,
                             gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    for col, key in enumerate(keys):
        res = val_dict[key]
        t = res['t']
        
        ax_y = axes[0, col]
        ax_u = axes[1, col]
        
        # Superior: Ângulo
        ax_y.plot(t, res['y_true'], color='#111111', lw=1.1, label='Medido $y(k)$')
        ax_y.plot(t, res['y_fr'], color='#d62728', ls='--', lw=1.2, label=r'NARX $\hat{y}(k)$')
        ax_y.set_title(res['label'], fontsize=9.5, fontweight='bold', pad=4)
        ax_y.grid(True)
        ax_y.set_xlim(0, t[-1])
        
        if col == 0:
            ax_y.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
            ax_y.legend(loc='upper right', fontsize=8)
            
        # Métricas na coluna
        txt = f"RMSE = {res['rmse']:.2f}" + r"$^\circ$" + f"\nFIT = {res['fit']:.1f}\\%"
        ax_y.text(0.03, 0.94, txt, transform=ax_y.transAxes,
                  verticalalignment='top', fontsize=8,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.85', alpha=0.9))
        
        # Inferior: Controle
        ax_u.plot(t, res['u'], color='#1f77b4', lw=1.0, label='Controle $u(k)$')
        ax_u.set_xlabel('Tempo [s]')
        ax_u.grid(True)
        ax_u.set_xlim(0, t[-1])
        if col == 0:
            ax_u.set_ylabel(r'Controle $u$ [\%]')
            
    plt.suptitle(suptitle, fontsize=11, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_academic_plot(fig, filename_base)
    plt.close(fig)

def plot_consolidated_validation_all(val_dict, filename_base="narx_val_consolidado_completo"):
    """Gera painel completo (2 linhas x 3 colunas) comparando 45° e 60°."""
    fig, axes = plt.subplots(4, 3, figsize=(13.0, 8.5), sharex=False,
                             gridspec_kw={'height_ratios': [2.0, 1.0, 2.0, 1.0]})
    
    rows_keys = [
        ['multisine_45_val', 'chirp_45_val', 'steps_45_val'],
        ['multisine_60_val', 'chirp_60_val', 'steps_60_val']
    ]
    
    for r_idx, keys in enumerate(rows_keys):
        row_y = r_idx * 2
        row_u = r_idx * 2 + 1
        for col, key in enumerate(keys):
            res = val_dict[key]
            t = res['t']
            
            ax_y = axes[row_y, col]
            ax_u = axes[row_u, col]
            
            ax_y.plot(t, res['y_true'], color='#111111', lw=1.1, label='Medido $y(k)$')
            ax_y.plot(t, res['y_fr'], color='#d62728', ls='--', lw=1.2, label=r'NARX $\hat{y}(k)$')
            ax_y.set_title(res['label'], fontsize=9.0, fontweight='bold', pad=3)
            ax_y.grid(True)
            ax_y.set_xlim(0, t[-1])
            
            if col == 0:
                ax_y.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
                ax_y.legend(loc='upper right', fontsize=7.5)
                
            txt = f"RMSE = {res['rmse']:.2f}" + r"$^\circ$" + f"\nFIT = {res['fit']:.1f}\\%"
            ax_y.text(0.03, 0.94, txt, transform=ax_y.transAxes,
                      verticalalignment='top', fontsize=7.5,
                      bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.85', alpha=0.9))
            
            ax_u.plot(t, res['u'], color='#1f77b4', lw=0.9, label='Controle $u(k)$')
            ax_u.set_xlabel('Tempo [s]')
            ax_u.grid(True)
            ax_u.set_xlim(0, t[-1])
            if col == 0:
                ax_u.set_ylabel(r'$u$ [\%]')
                
    plt.suptitle("Validação em Simulação Livre (Free-Run) nos Dois Pontos de Operação (45° e 60°)",
                 fontsize=11.5, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_academic_plot(fig, filename_base)
    plt.close(fig)

print("Gerando painel consolidado de validação (45°)...")
plot_consolidated_validation(val_results,
                             keys=['multisine_45_val', 'chirp_45_val', 'steps_45_val'],
                             filename_base="narx_val_consolidado",
                             suptitle="Validação em Simulação Livre (Free-Run) - Ponto de Operação 45°")

if 'multisine_60_val' in val_results:
    print("Gerando painel consolidado de validação (60°)...")
    plot_consolidated_validation(val_results,
                                 keys=['multisine_60_val', 'chirp_60_val', 'steps_60_val'],
                                 filename_base="narx_val_consolidado_60",
                                 suptitle="Validação em Simulação Livre (Free-Run) - Ponto de Operação 60°")
    print("Gerando painel consolidado completo (45° e 60°)...")
    plot_consolidated_validation_all(val_results, "narx_val_consolidado_completo")

# %% [markdown]
# ## Exportação do Modelo e das Equações
# 
# Salva:
# - `narx_model.json`: compatível com mpc_v2, mpc_v3, mpc_v4 e outros scripts.
# - `narx_metrics.json`: métricas quantitativas de cada ensaio.
# - `narx_equation.tex`: equação matemática formatada em LaTeX para a Monografia.
# - `narx_equation.txt`: equação em formato texto legível.

# %%
# 1. Exporta narx_model.json
terms_list = list(narx_model.selected_P_colnames_)
theta_list = [float(t) for t in narx_model.theta_]

model_json = {
    'ny': int(narx_model.ny),
    'nu': int(narx_model.nu),
    'l': int(narx_model.poly_order_l),
    'terms': terms_list,
    'theta': theta_list,
    'metrics': {
        k: {'rmse': v['rmse'], 'mae': v['mae'], 'fit': v['fit'], 'r2': v['r2']}
        for k, v in val_results.items()
    }
}

json_path = os.path.join(current_dir, 'narx_model.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(model_json, f, indent=4)
print(f"\n[OK] Modelo exportado: {json_path}")

# 2. Exporta narx_metrics.json
metrics_json = {
    k: {
        'label': v['label'],
        'rmse_deg': v['rmse'],
        'mae_deg': v['mae'],
        'fit_percent': v['fit'],
        'r2_score': v['r2'],
        'samples': len(v['t'])
    }
    for k, v in val_results.items()
}
metrics_path = os.path.join(current_dir, 'narx_metrics.json')
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_json, f, indent=4)
print(f"[OK] Métricas exportadas: {metrics_path}")

# 3. Exporta narx_equation.tex e narx_equation.txt
eq_terms_latex = []
eq_terms_txt = []

for term, th in zip(terms_list, theta_list):
    sign_tex = "+" if th >= 0 else "-"
    val = abs(th)
    
    if term == 'constant':
        eq_terms_latex.append(f"{sign_tex} {val:.4f}")
        eq_terms_txt.append(f"{sign_tex} {val:.6f}")
    else:
        # Formata índices: y(k-1) -> y[k-1] ou y_{k-1}
        t_tex = term.replace('y(k-', 'y[k-').replace('u(k-', 'u[k-').replace(')', ']')
        eq_terms_latex.append(f"{sign_tex} {val:.4e} \\cdot {t_tex}" if val < 1e-3 else f"{sign_tex} {val:.4f} \\cdot {t_tex}")
        eq_terms_txt.append(f"{sign_tex} {val:.6e} * {term}" if val < 1e-3 else f"{sign_tex} {val:.6f} * {term}")

latex_equation = "\\hat{y}(k) = " + " ".join(eq_terms_latex).lstrip("+ ")
txt_equation = "y_hat(k) = " + " ".join(eq_terms_txt).lstrip("+ ")

tex_path = os.path.join(current_dir, 'narx_equation.tex')
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(f"% Equação NARX identificada via FROLS/ERR\n\\begin{{equation}}\n    {latex_equation}\n    \\label{{eq:narx_identificado}}\n\\end{{equation}}\n")
print(f"[OK] Equação LaTeX exportada: {tex_path}")

txt_path = os.path.join(current_dir, 'narx_equation.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(f"Modelo NARX Identificado:\n{txt_equation}\n\nTermos e Parâmetros:\n")
    for i, (term, th) in enumerate(zip(terms_list, theta_list), 1):
        f.write(f"{i:2d}. {term:<20s} : {th:+.8e}\n")
print(f"[OK] Equação em texto exportada: {txt_path}")

print("\nProcesso de identificação, validação e salvamento de gráficos concluído com sucesso!")
