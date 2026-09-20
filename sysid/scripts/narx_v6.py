# %% [markdown]
# # NARX v6 — Estudo Comparativo de Tipos de Excitação
#
# Objetivo: replicar o protocolo do `node_v6.py` no modelo NARX Multi-Step BPTT
# para comparar exatamente quais excitações generalizam melhor.
#
# Protocolo idêntico ao v6:
#   TREINO   → arquivos da raiz /experimentos/ (Rodada 5, 27/08)
#   VAL      → RODADA-3/chirp-1_0807_16-32.csv  (early stopping free-run)
#   TESTE 🔒 → Testes da Rodada 2 e 4 (cross-session)
#

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime, os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import decimate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Seed deterministica
torch.manual_seed(0)
np.random.seed(0)

# ──────────────────────────────────────────────────────────────────────
# 1. DEFINIÇÃO DOS EXPERIMENTOS
# ──────────────────────────────────────────────────────────────────────
BASE = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"

EXPERIMENTOS = {
    "A_APRBS": [
        "aprbs-1_0827_17-19.csv",
        "aprbs-2_0827_17-25.csv",
        "aprbs-3_0827_17-28.csv",
        "aprbs-4_0827_17-31.csv",
    ],
    "B_MultiSeno": [
        "multi-seno-1_0827_17-34.csv",
        "multi-seno-2_0827_17-37.csv",
        "multi-seno-3_0827_17-40.csv",
        "multi-seno-4_0827_17-43.csv",
    ],
    "C_SweptSine": [
        "swept-sine-1_0827_17-58.csv",
        "swept-sine-4_0827_18-00.csv",
    ],
    "D_Degraus": [
        "seq-degraus-1_0827_17-46.csv",
        "seq-degraus-2_0827_17-49.csv",
        "seq-degraus-3_0827_17-52.csv",
        "seq-degraus-4_0827_17-55.csv",
    ],
    "E_Chirp": [
        "RODADA-3/chirp-1_0807_16-34.csv",
        "RODADA-3/chirp-2_0807_17-07.csv",
        "RODADA-3/chirp-2_0807_17-09.csv",
    ],
}
all_files = [f for files in EXPERIMENTOS.values() for f in files]
EXPERIMENTOS["F_Mix"] = list(dict.fromkeys(all_files))

VAL_FILES = [
    "RODADA-3/chirp-1_0807_16-32.csv",
]

TEST_FILES_BY_TYPE = {
    "APRBS":     "RODADA-4/aprbs-2_0819_18-51.csv",
    "MultiSeno": "RODADA-2/multi-seno-1_0804_19-06.csv",
    "SweptSine": "RODADA-4/swept-sine-1_0819_19-02.csv",
    "Degraus":   "RODADA-2/seq-degraus-2_0804_19-38.csv",
    "Chirp":     "RODADA-2/chirp-1_0804_19-19.csv",
}

# ──────────────────────────────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────────────────────────────
DECIMACAO = 2
START_IDX = 200
END_IDX   = -200

# Scalers globais baseados nas faixas operacionais 
scaler_u = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_u.fit(np.array([[-100], [100]]))
scaler_y.fit(np.array([[-180], [180]]))

def carregar_experimento(url, decimacao=DECIMACAO, start_idx=START_IDX, end_idx=END_IDX):
    df = pd.read_csv(url)
    if 'referencia' in df.columns:
        df = df[df['referencia'] > 0]
        
    u_full = df['u_pct'].values.astype(np.float64) if 'u_pct' in df.columns else df['motor_percent'].values.astype(np.float64)
    y_full = df['angulo_deg'].values.astype(np.float64)
    
    if decimacao > 1:
        y_raw = decimate(y_full, decimacao, ftype='iir', zero_phase=True)
        u_raw = u_full[::decimacao]
    else:
        u_raw, y_raw = u_full, y_full
        
    dt = 0.010 * decimacao
    t_raw = np.arange(len(y_raw)) * dt
    min_l = min(len(y_raw), len(u_raw))
    t_raw, u_raw, y_raw = t_raw[:min_l], u_raw[:min_l], y_raw[:min_l]
    
    if start_idx is not None and end_idx is not None:
        t_raw = t_raw[start_idx:end_idx]
        u_raw = u_raw[start_idx:end_idx]
        y_raw = y_raw[start_idx:end_idx]
        
    t_raw = t_raw - t_raw[0]
    return t_raw, u_raw, y_raw

def processar_dataset(t_raw, u_raw, y_raw):
    # Para NARX, mantemos a normalização em [-1, 1] do notebook original
    u_norm = scaler_u.transform(u_raw.reshape(-1, 1)).flatten()
    y_norm = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()
    return t_raw, u_norm, y_norm, u_raw, y_raw

def carregar_lista(file_list):
    datasets = []
    for f in file_list:
        t, u_raw, y_raw = carregar_experimento(BASE + f)
        t, u_norm, y_norm, _, _ = processar_dataset(t, u_raw, y_raw)
        datasets.append({
            'name': os.path.basename(f),
            't': t, 'u_norm': u_norm, 'y_norm': y_norm,
            'u_raw': u_raw, 'y_raw': y_raw
        })
    return datasets

# ──────────────────────────────────────────────────────────────────────
# 3. MODELO NARX
# ──────────────────────────────────────────────────────────────────────
class NARXModel(nn.Module):
    def __init__(self, ny, nu, hidden_dim1=256, hidden_dim2=128, dropout=0.1):
        super().__init__()
        self.ny = ny
        self.nu = nu
        self.linear_bypass = nn.Linear(ny + nu, 1)
        self.net = nn.Sequential(
            nn.Linear(ny + nu, hidden_dim1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward(self, x):
        return self.linear_bypass(x) + self.net(x)
        
    def multi_step_forward(self, y_past, u_seq, H):
        y_preds = []
        y_buffer = y_past.clone()
        for h in range(H):
            y_in = y_buffer[:, -self.ny:]
            u_in = u_seq[:, h : h + self.nu]
            x = torch.cat([y_in, u_in], dim=1)
            pred = self.forward(x)
            y_preds.append(pred)
            y_buffer = torch.cat([y_buffer, pred], dim=1)
        return torch.cat(y_preds, dim=1)
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

def make_bptt_windows(y, u, ny, nu, H):
    p_start = max(ny, nu)
    p_end = len(y) - H + 1
    Y_past_list, Y_target_list, U_seq_list = [], [], []
    for p in range(p_start, p_end):
        Y_past_list.append(y[p - ny : p])
        Y_target_list.append(y[p : p + H])
        U_seq_list.append(u[p - nu : p + H - 1])
    return np.array(Y_past_list), np.array(Y_target_list), np.array(U_seq_list)

def matReg(y, u, ny, nu):
    p = max(ny, nu) + 1
    N = len(y)
    Nu = len(u)
    if N != Nu: return None, None
    Phi = np.zeros((N - p + 1, ny + nu))
    for i in range(ny):
        Phi[:, i] = y[p - i - 2 : N - i - 1]
    for i in range(nu):
        Phi[:, i + ny] = u[p - i - 2 : N - i - 1]
    return y[p-1:N], Phi

def freeRun_numpy(model, y, u, ny, nu):
    p = max(ny, nu) + 1
    N = len(y)
    yhat = np.zeros(N)
    yhat[:p-1] = y[:p-1] 
    
    # Processo iterativo (sample-by-sample) para full free-run
    # Mais confiavel/flexivel que o forward multi-step dinamico para trajetorias muito longas
    for k in range(p, N+1):
        auxY = np.concatenate((yhat[(k-p):(k-1)], (0,)), axis=0)
        auxU = np.concatenate((u[(k-p):(k-1)], (0,)), axis=0)
        _, fr_input = matReg(auxY, auxU, ny, nu)
        tensor_input = torch.tensor(fr_input, dtype=torch.float32).to(device)
        yhat[k-1] = model.predict(tensor_input).item()
        
    return yhat

def avalia_free_run(model, datasets, ny, nu):
    model.eval()
    resultados = []
    for ds in datasets:
        y_norm = ds['y_norm']
        u_norm = ds['u_norm']
        
        y_pred_norm = freeRun_numpy(model, y_norm, u_norm, ny, nu)
        
        # O RMSE final deve ser em Graus (como no v6)
        y_real = ds['y_raw']
        y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        
        # Corta o inicio (onde nao ha predicao valida p)
        p = max(ny, nu) + 1
        y_real_valid = y_real[p-1:]
        y_pred_valid = y_pred[p-1:]
        t_valid = ds['t'][p-1:]
        
        rmse = float(np.sqrt(np.mean((y_pred_valid - y_real_valid) ** 2)))
        r2 = float(r2_score(y_real_valid, y_pred_valid))
        fit = float((1 - np.linalg.norm(y_pred_valid - y_real_valid) / (np.linalg.norm(y_real_valid - np.mean(y_real_valid)) + 1e-12)) * 100)
        
        resultados.append({
            'name': ds['name'], 'rmse': rmse, 'r2': r2, 'fit': fit,
            'y_real': y_real_valid, 'y_pred': y_pred_valid, 't': t_valid
        })
    return resultados

def val_rmse(model, val_datasets, ny, nu):
    res = avalia_free_run(model, val_datasets, ny, nu)
    return np.mean([r['rmse'] for r in res])

# ──────────────────────────────────────────────────────────────────────
# 4. TREINAMENTO
# ──────────────────────────────────────────────────────────────────────
def train_narx(model, name, train_datasets, val_datasets, ny, nu, H=10, epochs=1000, lr=1e-3):
    print(f"\n{'─'*60}")
    print(f"  Experimento: {name}  |  {len(train_datasets)} dataset(s) de treino")
    print(f"{'─'*60}")
    
    Y_past_all, Y_target_all, U_seq_all = [], [], []
    for ds in train_datasets:
        yp, yt, us = make_bptt_windows(ds['y_norm'], ds['u_norm'], ny, nu, H)
        if len(yp) > 0:
            Y_past_all.append(yp)
            Y_target_all.append(yt)
            U_seq_all.append(us)
            
    Y_past_t = torch.tensor(np.concatenate(Y_past_all, axis=0), dtype=torch.float32).to(device)
    Y_target_t = torch.tensor(np.concatenate(Y_target_all, axis=0), dtype=torch.float32).to(device)
    U_seq_t = torch.tensor(np.concatenate(U_seq_all, axis=0), dtype=torch.float32).to(device)
    
    dataset = TensorDataset(Y_past_t, U_seq_t, Y_target_t)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    patience_limit = 25  # Se passar 25 avaliacoes (125 epochs) sem melhorar freerun, parar.
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for y_past, u_seq, targets in dataloader:
            optimizer.zero_grad()
            outputs = model.multi_step_forward(y_past, u_seq, H)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(dataloader)
        
        # Validacao freerun a cada 5 épocas
        if epoch % 5 == 0 or epoch == 1:
            val = val_rmse(model, val_datasets, ny, nu)
            scheduler.step(val)
            
            if val < best_val_rmse:
                best_val_rmse = val
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 50 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:4d} | LR={lr_now:.5e} | Train={epoch_loss:.5f} | Best Val RMSE={best_val_rmse:.3f}deg")
                
            if patience_counter >= patience_limit:
                print(f"  -> Early stopping na epoch {epoch}.")
                break

    print(f"  -> Melhor Val RMSE: {best_val_rmse:.3f}deg — restaurando checkpoint.")
    model.load_state_dict(best_sd)
    return model, best_val_rmse

# ──────────────────────────────────────────────────────────────────────
# 5. PLOTS AUXILIARES
# ──────────────────────────────────────────────────────────────────────
def plot_datasets(datasets, title, out_path=None):
    n = len(datasets)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows * 2, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle(title, fontsize=13)
    axs = np.array(axs).reshape(rows * 2, cols)

    for i, ds in enumerate(datasets):
        r, c = (i // cols) * 2, i % cols
        t_np, y_deg, u_pct = ds['t'], ds['y_raw'], ds['u_raw']

        axs[r, c].plot(t_np, y_deg, color='steelblue', lw=0.8)
        axs[r, c].set_title(ds['name'], fontsize=8)
        axs[r, c].set_ylabel('Angulo (deg)', fontsize=7)
        axs[r, c].grid(True, lw=0.4)

        axs[r+1, c].plot(t_np, u_pct, color='darkorange', lw=0.8)
        axs[r+1, c].set_ylabel('u (%)', fontsize=7)
        axs[r+1, c].set_xlabel('Tempo (s)', fontsize=7)
        axs[r+1, c].grid(True, lw=0.4)

    for i in range(n, rows * cols):
        r, c = (i // cols) * 2, i % cols
        axs[r, c].axis('off')
        axs[r+1, c].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if out_path:
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()

def plot_free_run(resultado, exp_name, out_path=None):
    n = len(resultado)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    fig.suptitle(f'Free-Run NARX — {exp_name}', fontsize=13)
    axs = np.array(axs).reshape(rows, cols) if n > 1 else np.array([[axs]])

    for i, r in enumerate(resultado):
        ax = axs[i // cols, i % cols]
        ax.plot(r['t'], r['y_real'], 'k',   lw=1.2, label='Real')
        ax.plot(r['t'], r['y_pred'], 'r--', lw=1.0, label='Pred')
        ax.set_title(f"{r['name']}\nRMSE={r['rmse']:.2f}deg R2={r['r2']:.3f} FIT={r['fit']:.1f}%", fontsize=8)
        ax.set_xlabel('Tempo (s)', fontsize=7)
        ax.set_ylabel('Angulo (deg)', fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.4)

    for i in range(n, rows * cols):
        axs[i // cols, i % cols].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if out_path:
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()

def plot_heatmap(matriz, metrica, out_path=None):
    treinos = list(matriz.keys())
    tipos   = list(next(iter(matriz.values())).keys())
    dados   = np.array([[matriz[t][tp][metrica] for tp in tipos] for t in treinos])

    fig, ax = plt.subplots(figsize=(len(tipos) * 1.6 + 1, len(treinos) * 0.9 + 1.5))
    cmap = 'RdYlGn' if metrica != 'rmse' else 'RdYlGn_r'
    im = ax.imshow(dados, aspect='auto', cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(tipos)))
    ax.set_xticklabels(tipos, fontsize=9)
    ax.set_yticks(range(len(treinos)))
    ax.set_yticklabels(treinos, fontsize=9)
    ax.set_xlabel('Tipo de Teste', fontsize=10)
    ax.set_ylabel('Tipo de Treino', fontsize=10)
    label = {'rmse': 'RMSE (deg)', 'r2': 'R2', 'fit': 'FIT%'}[metrica]
    ax.set_title(f'Matriz de Generalizacao NARX — {label}', fontsize=12)
    for i in range(len(treinos)):
        for j in range(len(tipos)):
            ax.text(j, i, f"{dados[i,j]:.2f}", ha='center', va='center', fontsize=8, color='black')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()

def plot_comparativo(resultados_todos, out_path=None):
    exps = list(resultados_todos.keys())
    rmses = [np.mean([r['rmse'] for r in v]) for v in resultados_todos.values()]
    r2s = [np.mean([r['r2'] for r in v]) for v in resultados_todos.values()]
    fits = [np.mean([r['fit'] for r in v]) for v in resultados_todos.values()]

    x = np.arange(len(exps))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Comparativo NARX — Media Geral (Todos os Tipos de Teste)', fontsize=13)

    for ax, vals, ylabel, color in zip(axes, [rmses, r2s, fits], ['RMSE (deg)', 'R2', 'FIT (%)'], ['steelblue', 'seagreen', 'darkorange']):
        bars = ax.bar(x, vals, color=color, alpha=0.85, edgecolor='black', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(exps, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', lw=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * (max(vals) - min(vals) + 1e-9), f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()

# ──────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"resultados_narx_v6_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Resultados NARX em: {out_dir}/\n")

    print("Carregando Val e conjuntos de Teste por tipo...")
    val_datasets = carregar_lista(VAL_FILES)
    test_por_tipo = {tipo: carregar_lista([arquivo]) for tipo, arquivo in TEST_FILES_BY_TYPE.items()}
    print(f"  Val  : {[d['name'] for d in val_datasets]}")
    for tipo, dsl in test_por_tipo.items():
        print(f"  Teste {tipo:<10}: {[d['name'] for d in dsl]}")

    ny = 30
    nu = 50
    H = 10

    matriz = {}
    resumo = {}

    for exp_nome, train_files in EXPERIMENTOS.items():
        print(f"\n{'='*60}")
        print(f"  EXPERIMENTO NARX {exp_nome}")
        print(f"{'='*60}")

        train_datasets = carregar_lista(train_files)
        plot_datasets(train_datasets, f'Treino NARX — {exp_nome}', out_path=f"{out_dir}/treino_{exp_nome}.png")

        torch.manual_seed(0)
        np.random.seed(0)
        model = NARXModel(ny=ny, nu=nu, hidden_dim1=256, hidden_dim2=128).to(device)

        model, best_val = train_narx(
            model, exp_nome, train_datasets, val_datasets,
            ny=ny, nu=nu, H=H, epochs=1000, lr=1e-3
        )

        torch.save(model.state_dict(), f"{out_dir}/model_narx_{exp_nome}.pth")

        matriz[exp_nome] = {}
        todos_resultados = []
        for tipo, dsl in test_por_tipo.items():
            res = avalia_free_run(model, dsl, ny, nu)
            todos_resultados.extend(res)
            rmse_m = np.mean([r['rmse'] for r in res])
            r2_m = np.mean([r['r2'] for r in res])
            fit_m = np.mean([r['fit'] for r in res])
            matriz[exp_nome][tipo] = {'rmse': rmse_m, 'r2': r2_m, 'fit': fit_m}
            print(f"  [TESTE {tipo:<10}] RMSE={rmse_m:.3f}deg  R2={r2_m:.4f}  FIT={fit_m:.1f}%")

        plot_free_run(todos_resultados, exp_nome, out_path=f"{out_dir}/freerun_{exp_nome}.png")

        rmse_geral = np.mean([v['rmse'] for v in matriz[exp_nome].values()])
        r2_geral = np.mean([v['r2'] for v in matriz[exp_nome].values()])
        fit_geral = np.mean([v['fit'] for v in matriz[exp_nome].values()])
        resumo[exp_nome] = {
            'por_tipo': {t: {k: round(v,4) for k,v in m.items()} for t, m in matriz[exp_nome].items()},
            'media_geral': {'RMSE_deg': round(rmse_geral,4), 'R2': round(r2_geral,4), 'FIT_pct': round(fit_geral,2)},
            'best_val_rmse': round(best_val, 4),
            'params': {'ny': ny, 'nu': nu, 'H': H}
        }
        print(f"  [MEDIA GERAL] RMSE={rmse_geral:.3f}deg  R2={r2_geral:.4f}  FIT={fit_geral:.1f}%")

    for metrica in ['rmse', 'r2', 'fit']:
        plot_heatmap(matriz, metrica, out_path=f"{out_dir}/heatmap_{metrica}.png")

    resultados_para_barchart = {
        exp: [{'rmse': m['media_geral']['RMSE_deg'], 'r2': m['media_geral']['R2'], 'fit': m['media_geral']['FIT_pct']}]
        for exp, m in resumo.items()
    }
    plot_comparativo(resultados_para_barchart, out_path=f"{out_dir}/comparativo_geral.png")

    tipos = list(TEST_FILES_BY_TYPE.keys())
    col_w = 10
    header = f"{'Treino':<14}" + "".join(f" {'RMSE_'+t:>{col_w}}" for t in tipos) + f" {'MEDIA':>{col_w}}"
    print(f"\n{'='*len(header)}")
    print("  RESUMO NARX — RMSE (deg) por tipo de teste")
    print(f"{'='*len(header)}")
    print(header)
    print('─' * len(header))
    for exp, m in resumo.items():
        linha = f"{exp:<14}"
        for t in tipos:
            linha += f" {m['por_tipo'][t]['rmse']:>{col_w}.3f}"
        linha += f" {m['media_geral']['RMSE_deg']:>{col_w}.3f}"
        print(linha)

    with open(f"{out_dir}/resumo_narx.json", 'w') as fp:
        json.dump(resumo, fp, indent=2)
    print(f"\nResultados NARX em: {out_dir}/")
    print("[V6 NARX] Estudo comparativo concluido.")
