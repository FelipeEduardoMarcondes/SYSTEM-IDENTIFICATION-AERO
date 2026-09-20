# %% [markdown]
# # NODE v6 — Estudo Comparativo de Tipos de Excitação
#
# Objetivo: identificar qual tipo de sinal de excitação no treino
# produz o modelo NODE mais generalizável, usando chirp como
# avaliador universal (free-run completo).
#
# Protocolo:
#   TREINO   → arquivos da raiz /experimentos/ (Rodada 5, 27/08)
#   VAL      → RODADA-3/chirp-1_0807_16-32.csv  (early stopping)
#   TESTE 🔒 → RODADA-2/chirp-1_0804_19-19.csv  (avaliação final única)
#
# Experimentos (mesmo modelo, mesmo hiperparâmetro, treino diferente):
#   A  — só APRBS
#   B  — só Multi-seno
#   C  — só Varredura (swept-sine + chirp unificados)
#   D  — só Seq-degraus
#   E  — mix de todos  <- controle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # backend sem GUI para rodar em loop
import matplotlib.pyplot as plt
import datetime, os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from scipy.signal import savgol_filter, decimate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(0)
np.random.seed(0)

# ──────────────────────────────────────────────────────────────────────
# 1. PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────────────────────────────
BASE = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"

DECIMACAO  = 2
START_IDX  = 200    # amostras cortadas no inicio (apos decimacao)
END_IDX    = -200   # amostras cortadas no fim    (apos decimacao)


def carregar_experimento(url, decimacao=DECIMACAO, start_idx=START_IDX, end_idx=END_IDX):
    df = pd.read_csv(url)
    if 'referencia' in df.columns:
        df = df[df['referencia'] > 0]

    u_full = df['u_pct'].values.astype(np.float64) if 'u_pct' in df.columns \
             else df['motor_percent'].values.astype(np.float64)
    y_full = df['angulo_deg'].values.astype(np.float64)

    if decimacao > 1:
        y_raw = decimate(y_full, decimacao, ftype='iir', zero_phase=True)
        u_raw = u_full[::decimacao]
    else:
        u_raw, y_raw = u_full, y_full

    dt    = 0.010 * decimacao
    t_raw = np.arange(len(y_raw)) * dt
    min_l = min(len(y_raw), len(u_raw))
    t_raw, u_raw, y_raw = t_raw[:min_l], u_raw[:min_l], y_raw[:min_l]

    if start_idx is not None and end_idx is not None:
        t_raw = t_raw[start_idx:end_idx]
        u_raw = u_raw[start_idx:end_idx]
        y_raw = y_raw[start_idx:end_idx]

    return t_raw, u_raw, y_raw


def processar_dataset(t_raw, u_raw, y_raw):
    t_raw  = t_raw - t_raw[0]
    y_rad  = y_raw * (np.pi / 180.0)
    u_norm = np.clip(u_raw / 100.0, -1.0, 1.0)
    dt_m   = np.mean(np.diff(t_raw))
    v_rad  = savgol_filter(y_rad, 11, 3, deriv=1, delta=dt_m)
    x_mat  = np.vstack((y_rad, v_rad)).T
    return (
        torch.tensor(t_raw, dtype=torch.float32),
        torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1),
        torch.tensor(x_mat,  dtype=torch.float32),
        y_rad, v_rad, u_norm,
    )


def carregar_lista(file_list):
    datasets = []
    for f in file_list:
        t, u, y = carregar_experimento(BASE + f)
        t_ten, u_ten, x_ten, *_ = processar_dataset(t, u, y)
        datasets.append({'name': os.path.basename(f), 't': t_ten, 'u': u_ten, 'x': x_ten})
    return datasets


# ──────────────────────────────────────────────────────────────────────
# 2. DEFINIÇÃO DOS EXPERIMENTOS
# ──────────────────────────────────────────────────────────────────────
# Todos os arquivos de treino estao na raiz (Rodada 5, 27/08).
# Chirp e swept-sine sao ambos sinais de varredura de frequencia
# (linear vs logaritmico), unificados no Exp C_Varredura.

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
    "C_Varredura": [
        "swept-sine-1_0827_17-58.csv",
        "swept-sine-4_0827_18-00.csv",
        "RODADA-3/chirp-1_0807_16-34.csv",
        "RODADA-3/chirp-2_0807_17-07.csv",
        "RODADA-3/chirp-2_0807_17-09.csv",
    ],
    "D_Degraus": [
        "seq-degraus-1_0827_17-46.csv",
        "seq-degraus-2_0827_17-49.csv",
        "seq-degraus-3_0827_17-52.csv",
        "seq-degraus-4_0827_17-55.csv",
    ],
}
# Exp E: mix de todos os anteriores (sem duplicatas)
all_files = [f for files in EXPERIMENTOS.values() for f in files]
EXPERIMENTOS["E_Mix"] = list(dict.fromkeys(all_files))

# Val: guia o early stopping via free-run RMSE (nunca entra no treino)
VAL_FILES = [
    "RODADA-3/chirp-1_0807_16-32.csv",
]

# Teste final por tipo (bloqueado, cross-session, arquivos por tipo):
# - APRBS    : RODADA-4 (nao usada no treino do v6)
# - MultiSeno: RODADA-2 (nao usada no treino do v6)
# - Varredura: RODADA-4 + RODADA-2 (nao usadas no treino do v6)
# - Degraus  : RODADA-2 (nao usada no treino do v6)
TEST_FILES_BY_TYPE = {
    "APRBS":     ["RODADA-4/aprbs-2_0819_18-51.csv"],
    "MultiSeno": ["RODADA-2/multi-seno-1_0804_19-06.csv"],
    "Varredura": ["RODADA-4/swept-sine-1_0819_19-02.csv",
                   "RODADA-2/chirp-1_0804_19-19.csv"],
    "Degraus":   ["RODADA-2/seq-degraus-2_0804_19-38.csv"],
}


# ──────────────────────────────────────────────────────────────────────
# 3. MODELO — PhysicsODE Assimetrico (fixo para todos os experimentos)
# ──────────────────────────────────────────────────────────────────────
class BaseODE(nn.Module):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0):
        super().__init__()
        self.m1, self.L1 = 0.122, 0.39
        self.m2, self.L2 = 0.055, 0.347
        self.g = 9.81
        self.log_J  = nn.Parameter(torch.log(torch.tensor(float(J0))))
        self.log_b  = nn.Parameter(torch.log(torch.tensor(float(b0))))
        self.log_Gu = nn.Parameter(torch.log(torch.tensor(float(Gu0))))
        self.u_series = None
        self.t_series = None
        self.batch_start_times = None

    def get_params(self):
        return torch.exp(self.log_J), torch.exp(self.log_b), torch.exp(self.log_Gu)

    def _get_u_t(self, t, x):
        if self.batch_start_times is not None:
            t_abs = self.batch_start_times + t
        else:
            t_abs = t * torch.ones_like(x[:, 0:1])
        k = torch.searchsorted(self.t_series, t_abs.reshape(-1), right=True)
        k = torch.clamp(k, 1, len(self.t_series) - 1)
        t1, t2 = self.t_series[k-1].unsqueeze(1), self.t_series[k].unsqueeze(1)
        u1, u2 = self.u_series[k-1], self.u_series[k]
        denom = (t2 - t1)
        denom[denom < 1e-6] = 1.0
        alpha = (t_abs - t1) / denom
        return u1 + alpha * (u2 - u1)


class PhysicsODE_Asymmetric(BaseODE):
    def __init__(self, J0=1.0, b_pos0=np.exp(-1.0), b_neg0=np.exp(-1.0),
                 Gu_pos0=1.0, Gu_neg0=1.0):
        super().__init__(J0, b_pos0, Gu_pos0)
        self.log_b_neg  = nn.Parameter(torch.log(torch.tensor(float(b_neg0))))
        self.log_Gu_neg = nn.Parameter(torch.log(torch.tensor(float(Gu_neg0))))

    def forward(self, t, x):
        J      = torch.clamp(torch.exp(self.log_J),       min=0.01, max=1.0)
        b_pos  = torch.clamp(torch.exp(self.log_b),       min=0.001, max=1.0)
        b_neg  = torch.clamp(torch.exp(self.log_b_neg),   min=0.001, max=1.0)
        Gu_pos = torch.clamp(torch.exp(self.log_Gu),      min=0.01, max=10.0)
        Gu_neg = torch.clamp(torch.exp(self.log_Gu_neg),  min=0.01, max=10.0)
        u_t    = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]

        sigma_b  = torch.sigmoid(50.0 * theta_dot)
        b        = sigma_b * b_pos + (1.0 - sigma_b) * b_neg
        sigma_Gu = torch.sigmoid(50.0 * u_t)
        Gu       = sigma_Gu * Gu_pos + (1.0 - sigma_Gu) * Gu_neg

        motor_torque    = Gu * u_t * torch.abs(u_t)
        gravity_torque  = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot
        theta_ddot      = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)


# ──────────────────────────────────────────────────────────────────────
# 4. TREINAMENTO
# ──────────────────────────────────────────────────────────────────────
def avalia_free_run(model, datasets, integrator='rk4'):
    """Simula free-run completo (cond. inicial real, u real, sem correcao)."""
    model.eval()
    resultados = []
    with torch.no_grad():
        for ds in datasets:
            t_t = ds['t'].to(device)
            u_t = ds['u'].to(device)
            x_t = ds['x'].to(device)
            model.t_series          = t_t
            model.u_series          = u_t
            model.batch_start_times = torch.zeros(1, 1, device=device)
            x0   = x_t[0].unsqueeze(0)
            pred = odeint(model, x0, t_t, method=integrator).squeeze(1)

            y_real = x_t[:, 0].cpu().numpy() * (180 / np.pi)
            y_pred = pred[:, 0].cpu().numpy() * (180 / np.pi)

            rmse = float(np.sqrt(np.mean((y_pred - y_real) ** 2)))
            ss_res = np.sum((y_real - y_pred) ** 2)
            ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
            r2   = float(1 - ss_res / (ss_tot + 1e-12))
            fit  = float((1 - np.linalg.norm(y_pred - y_real) /
                          (np.linalg.norm(y_real - np.mean(y_real)) + 1e-12)) * 100)
            resultados.append({
                'name': ds['name'], 'rmse': rmse, 'r2': r2, 'fit': fit,
                'y_real': y_real, 'y_pred': y_pred, 't': ds['t'].numpy(),
            })
    return resultados


def val_rmse(model, val_datasets):
    res = avalia_free_run(model, val_datasets)
    return np.mean([r['rmse'] for r in res])


def train_node(model, name, train_datasets, val_datasets,
               epochs=2000, lr=0.015,
               k_min=20, k_max=400, curriculum_stage_epochs=400,
               base_batch_size=1024, integrator='rk4'):
    print(f"\n{'─'*60}")
    print(f"  Experimento: {name}  |  {len(train_datasets)} dataset(s) de treino")
    print(f"{'─'*60}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    all_pos   = np.concatenate([d['x'][:, 0].numpy() for d in train_datasets])
    all_vel   = np.concatenate([d['x'][:, 1].numpy() for d in train_datasets])
    state_std = torch.tensor([float(np.std(all_pos)), float(np.std(all_vel))],
                              dtype=torch.float32, device=device)

    dt = (train_datasets[0]['t'][1] - train_datasets[0]['t'][0]).item()

    best_val = val_rmse(model, val_datasets)
    best_sd  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        stage   = epoch // curriculum_stage_epochs
        k_steps = int(min(k_min * (2 ** stage), k_max))
        t_eval  = torch.arange(0, k_steps * dt, dt, device=device)[:k_steps]

        batch_size     = max(64, int(base_batch_size * (k_min / k_steps)))
        samples_per_ds = max(1, batch_size // len(train_datasets))

        optimizer.zero_grad()
        total_loss = 0

        for ds in train_datasets:
            t_ds, u_ds, x_ds = ds['t'].to(device), ds['u'].to(device), ds['x'].to(device)
            model.t_series, model.u_series = t_ds, u_ds

            start_idx = np.random.randint(0, len(t_ds) - k_steps, size=samples_per_ds)
            x0 = x_ds[start_idx]
            model.batch_start_times = t_ds[start_idx].reshape(-1, 1)

            pred_state    = odeint(model, x0, t_eval, method=integrator)
            batch_targets = torch.stack([x_ds[i:i + k_steps] for i in start_idx], dim=1)
            loss = torch.mean(((pred_state - batch_targets) / state_std) ** 2)
            total_loss += loss

        total_loss = total_loss / len(train_datasets)
        # Regularizacao L2 nos parametros assimetricos para evitar divergencia
        reg_loss = 0.01 * (model.log_Gu_neg ** 2 + model.log_b_neg ** 2)
        total_loss = total_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == epochs:
            val = val_rmse(model, val_datasets)
            if val < best_val:
                best_val = val
                best_sd  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 200 == 0 or epoch == epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d} | k={k_steps:3d} | LR={lr_now:.5f} | "
                  f"Train={total_loss.item():.4f} | Best Val RMSE={best_val:.3f}deg")

    print(f"  -> Melhor Val RMSE: {best_val:.3f}deg  — restaurando checkpoint.")
    model.load_state_dict(best_sd)
    return model, best_val


# ──────────────────────────────────────────────────────────────────────
# 5. PLOTS AUXILIARES
# ──────────────────────────────────────────────────────────────────────
def plot_datasets(datasets, title, out_path=None):
    n    = len(datasets)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows * 2, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle(title, fontsize=13)
    axs = np.array(axs).reshape(rows * 2, cols)

    for i, ds in enumerate(datasets):
        r, c  = (i // cols) * 2, i % cols
        t_np  = ds['t'].numpy()
        y_deg = ds['x'][:, 0].numpy() * (180 / np.pi)
        u_pct = ds['u'][:, 0].numpy() * 100.0

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
    n    = len(resultado)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    fig.suptitle(f'Free-Run — {exp_name}', fontsize=13)
    axs = np.array(axs).reshape(rows, cols) if n > 1 else np.array([[axs]])

    for i, r in enumerate(resultado):
        ax = axs[i // cols, i % cols]
        ax.plot(r['t'], r['y_real'], 'k',   lw=1.2, label='Real')
        ax.plot(r['t'], r['y_pred'], 'r--', lw=1.0, label='Pred')
        ax.set_title(
            f"{r['name']}\nRMSE={r['rmse']:.2f}deg  R2={r['r2']:.3f}  FIT={r['fit']:.1f}%",
            fontsize=8)
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
    """Heatmap: linhas=tipo de treino, colunas=tipo de teste."""
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
    ax.set_title(f'Matriz de Generalizacao — {label}', fontsize=12)
    for i in range(len(treinos)):
        for j in range(len(tipos)):
            ax.text(j, i, f"{dados[i,j]:.2f}", ha='center', va='center',
                    fontsize=8, color='black')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap salvo: {out_path}")


def plot_comparativo(resultados_todos, out_path=None):
    exps  = list(resultados_todos.keys())
    rmses = [np.mean([r['rmse'] for r in v]) for v in resultados_todos.values()]
    r2s   = [np.mean([r['r2']   for r in v]) for v in resultados_todos.values()]
    fits  = [np.mean([r['fit']  for r in v]) for v in resultados_todos.values()]

    x   = np.arange(len(exps))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Comparativo — Media Geral (Todos os Tipos de Teste)', fontsize=13)


    for ax, vals, ylabel, color in zip(
            axes,
            [rmses, r2s, fits],
            ['RMSE (deg)', 'R2', 'FIT (%)'],
            ['steelblue', 'seagreen', 'darkorange']):
        bars = ax.bar(x, vals, color=color, alpha=0.85, edgecolor='black', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(exps, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', lw=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * (max(vals) - min(vals) + 1e-9),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  Barchart salvo: {out_path}")


# ──────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = f"resultados_v6_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Resultados em: {out_dir}/\n")

    # Carregar val e todos os conjuntos de teste
    print("Carregando Val e conjuntos de Teste por tipo...")
    val_datasets = carregar_lista(VAL_FILES)
    test_por_tipo = {
        tipo: carregar_lista(arquivos)
        for tipo, arquivos in TEST_FILES_BY_TYPE.items()
    }
    print(f"  Val  : {[d['name'] for d in val_datasets]}")
    for tipo, dsl in test_por_tipo.items():
        print(f"  Teste {tipo:<10}: {[d['name'] for d in dsl]}")

    # matriz[exp_nome][tipo_teste] = {'rmse', 'r2', 'fit'}
    matriz   = {}
    resumo   = {}

    for exp_nome, train_files in EXPERIMENTOS.items():
        print(f"\n{'='*60}")
        print(f"  EXPERIMENTO {exp_nome}")
        print(f"{'='*60}")

        train_datasets = carregar_lista(train_files)
        plot_datasets(train_datasets,
                      f'Treino — {exp_nome}',
                      out_path=f"{out_dir}/treino_{exp_nome}.png")
        print(f"  {len(train_datasets)} dataset(s) de treino carregados.")

        torch.manual_seed(0)
        np.random.seed(0)
        model = PhysicsODE_Asymmetric()

        model, best_val = train_node(
            model, exp_nome, train_datasets, val_datasets,
            epochs=2000, lr=0.015,
            k_min=20, k_max=400, curriculum_stage_epochs=400,
        )

        torch.save(model.state_dict(), f"{out_dir}/model_{exp_nome}.pth")

        # Parâmetros fisicos
        J, b, Gu = [p.item() for p in model.get_params()]
        b_neg    = torch.exp(model.log_b_neg).item()
        Gu_neg   = torch.exp(model.log_Gu_neg).item()
        print(f"  Params -> J={J:.4f}  b+={b:.4f}  b-={b_neg:.4f}  "
              f"Gu+={Gu:.4f}  Gu-={Gu_neg:.4f}")

        # Avaliacao em TODOS os tipos de teste
        matriz[exp_nome] = {}
        todos_resultados = []
        for tipo, dsl in test_por_tipo.items():
            res = avalia_free_run(model, dsl)
            todos_resultados.extend(res)
            rmse_m = np.mean([r['rmse'] for r in res])
            r2_m   = np.mean([r['r2']   for r in res])
            fit_m  = np.mean([r['fit']  for r in res])
            matriz[exp_nome][tipo] = {'rmse': rmse_m, 'r2': r2_m, 'fit': fit_m}
            print(f"  [TESTE {tipo:<10}] RMSE={rmse_m:.3f}deg  R2={r2_m:.4f}  FIT={fit_m:.1f}%")

        # Plot free-run de todos os tipos juntos
        plot_free_run(todos_resultados, exp_nome,
                      out_path=f"{out_dir}/freerun_{exp_nome}.png")

        # Media geral (todos os tipos)
        rmse_geral = np.mean([v['rmse'] for v in matriz[exp_nome].values()])
        r2_geral   = np.mean([v['r2']   for v in matriz[exp_nome].values()])
        fit_geral  = np.mean([v['fit']  for v in matriz[exp_nome].values()])
        resumo[exp_nome] = {
            'por_tipo': {t: {k: round(v,4) for k,v in m.items()}
                         for t, m in matriz[exp_nome].items()},
            'media_geral': {'RMSE_deg': round(rmse_geral,4),
                            'R2': round(r2_geral,4),
                            'FIT_pct': round(fit_geral,2)},
            'best_val_rmse': round(best_val, 4),
            'params': {'J': round(J,4), 'b+': round(b,4), 'b-': round(b_neg,4),
                       'Gu+': round(Gu,4), 'Gu-': round(Gu_neg,4)},
        }
        print(f"  [MEDIA GERAL] RMSE={rmse_geral:.3f}deg  R2={r2_geral:.4f}  FIT={fit_geral:.1f}%")

    # Heatmaps por metrica
    for metrica in ['rmse', 'r2', 'fit']:
        plot_heatmap(matriz, metrica,
                     out_path=f"{out_dir}/heatmap_{metrica}.png")

    # Barchart de media geral
    resultados_para_barchart = {
        exp: [{'rmse': m['media_geral']['RMSE_deg'],
               'r2':   m['media_geral']['R2'],
               'fit':  m['media_geral']['FIT_pct']}]
        for exp, m in resumo.items()
    }
    plot_comparativo(resultados_para_barchart,
                     out_path=f"{out_dir}/comparativo_geral.png")

    # Tabela de resumo no console
    tipos = list(TEST_FILES_BY_TYPE.keys())
    col_w = 10
    header = f"{'Treino':<14}" + "".join(f" {'RMSE_'+t:>{col_w}}" for t in tipos) + f" {'MEDIA':>{col_w}}"
    print(f"\n{'='*len(header)}")
    print("  RESUMO — RMSE (deg) por tipo de teste")
    print(f"{'='*len(header)}")
    print(header)
    print('─' * len(header))
    for exp, m in resumo.items():
        linha = f"{exp:<14}"
        for t in tipos:
            linha += f" {m['por_tipo'][t]['rmse']:>{col_w}.3f}"
        linha += f" {m['media_geral']['RMSE_deg']:>{col_w}.3f}"
        print(linha)

    with open(f"{out_dir}/resumo.json", 'w') as fp:
        json.dump(resumo, fp, indent=2)
    print(f"\nResultados em: {out_dir}/")
    print("[V6] Estudo comparativo concluido.")
