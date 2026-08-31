# %% [markdown]
# # NODE v7 — Design Fatorial: Modelo × Excitação
#
# Estudo comparativo cruzando 5 arquiteturas de modelo de atrito
# com 5 tipos de excitação = 25 treinamentos.
# Cada modelo avaliado em 4 tipos de teste cross-session.
#
# Modelos:
#   Baseline     — viscoso simétrico (3 params)
#   Coulomb      — viscoso + Coulomb seco (4 params)
#   Tustin       — viscoso + Stribeck (6 params)
#   Assimétrico  — viscoso + ganho direção-dependentes (5 params)
#   Híbrido      — baseline + MLP residual (3 + MLP)
#
# Excitações:
#   APRBS, Multi-seno, Varredura, Degraus, Mix
#
# Validação (early stopping):
#   Chirp broadband da Rodada 3 — mesmo critério para todos os 25 runs.
#   Justificativa: chirp testa ampla faixa de frequências,
#   e early stopping é regularização, não critério final.
#   Patience de 500 épocas para economia de tempo computacional.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime, os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from scipy.signal import savgol_filter, decimate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ──────────────────────────────────────────────────────────────────────
# 1. PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────────────────────────────
BASE = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"

DECIMACAO  = 2
START_IDX  = 200
END_IDX    = -150


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
# 2. MODELOS — 5 arquiteturas de atrito, todas com clamping
# ──────────────────────────────────────────────────────────────────────
class BaseODE(nn.Module):
    """Infraestrutura compartilhada: gravidade, interpolação de u, clamping."""
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

    def _gravity(self, theta):
        return (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)

    def _motor(self, Gu, u_t):
        return Gu * u_t * torch.abs(u_t)

    def reg_loss(self):
        """Override nas subclasses para regularizar parâmetros difíceis."""
        return torch.tensor(0.0, device=self.log_J.device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhysicsODE_Baseline(BaseODE):
    """Atrito viscoso simétrico. 3 params: J, b, Gu."""
    def forward(self, t, x):
        J  = torch.clamp(torch.exp(self.log_J),  min=0.01, max=1.0)
        b  = torch.clamp(torch.exp(self.log_b),  min=0.001, max=1.0)
        Gu = torch.clamp(torch.exp(self.log_Gu), min=0.01, max=10.0)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        theta_ddot = (self._motor(Gu, u_t) - self._gravity(theta) - b * theta_dot) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)


class PhysicsODE_Coulomb(BaseODE):
    """Viscoso + Coulomb seco. 4 params: J, b, Gu, Tc."""
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0, Tc0=0.01):
        super().__init__(J0, b0, Gu0)
        self.log_Tc = nn.Parameter(torch.log(torch.tensor(float(Tc0))))

    def forward(self, t, x):
        J  = torch.clamp(torch.exp(self.log_J),  min=0.01, max=1.0)
        b  = torch.clamp(torch.exp(self.log_b),  min=0.001, max=1.0)
        Gu = torch.clamp(torch.exp(self.log_Gu), min=0.01, max=10.0)
        Tc = torch.clamp(torch.exp(self.log_Tc), min=0.001, max=1.0)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        tau_f = b * theta_dot + Tc * torch.tanh(50.0 * theta_dot)
        theta_ddot = (self._motor(Gu, u_t) - self._gravity(theta) - tau_f) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)

    def reg_loss(self):
        return 0.005 * self.log_Tc ** 2


class PhysicsODE_Tustin(BaseODE):
    """Viscoso + Stribeck completo. 6 params: J, b, Gu, Tc, Ts, vs."""
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0,
                 Tc0=0.01, Ts0=0.02, vs0=0.1):
        super().__init__(J0, b0, Gu0)
        self.log_Tc = nn.Parameter(torch.log(torch.tensor(float(Tc0))))
        self.log_Ts = nn.Parameter(torch.log(torch.tensor(float(Ts0))))
        self.log_vs = nn.Parameter(torch.log(torch.tensor(float(vs0))))

    def forward(self, t, x):
        J  = torch.clamp(torch.exp(self.log_J),  min=0.01, max=1.0)
        b  = torch.clamp(torch.exp(self.log_b),  min=0.001, max=1.0)
        Gu = torch.clamp(torch.exp(self.log_Gu), min=0.01, max=10.0)
        Tc = torch.clamp(torch.exp(self.log_Tc), min=0.001, max=1.0)
        Ts = torch.clamp(torch.exp(self.log_Ts), min=0.001, max=1.0)
        vs = torch.clamp(torch.exp(self.log_vs), min=0.001, max=1.0)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        stribeck = Tc + (Ts - Tc) * torch.exp(-torch.abs(theta_dot / vs))
        tau_f = b * theta_dot + stribeck * torch.tanh(50.0 * theta_dot)
        theta_ddot = (self._motor(Gu, u_t) - self._gravity(theta) - tau_f) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)

    def reg_loss(self):
        return 0.005 * (self.log_Tc**2 + self.log_Ts**2 + self.log_vs**2)


class PhysicsODE_Asymmetric(BaseODE):
    """Viscoso + ganho assimétricos (sigmóide). 5 params: J, b+, b-, Gu+, Gu-."""
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
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]

        sigma_b  = torch.sigmoid(50.0 * theta_dot)
        b  = sigma_b * b_pos + (1.0 - sigma_b) * b_neg
        sigma_Gu = torch.sigmoid(50.0 * u_t)
        Gu = sigma_Gu * Gu_pos + (1.0 - sigma_Gu) * Gu_neg

        tau_f = b * theta_dot
        theta_ddot = (self._motor(Gu, u_t) - self._gravity(theta) - tau_f) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)

    def reg_loss(self):
        return 0.01 * (self.log_Gu_neg**2 + self.log_b_neg**2)


class PhysicsODE_Hybrid(BaseODE):
    """Baseline + MLP residual. 3 params físicos + MLP(3→16→1) = ~84 params."""
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0, hidden_dim=16):
        super().__init__(J0, b0, Gu0)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t, x):
        J  = torch.clamp(torch.exp(self.log_J),  min=0.01, max=1.0)
        b  = torch.clamp(torch.exp(self.log_b),  min=0.001, max=1.0)
        Gu = torch.clamp(torch.exp(self.log_Gu), min=0.01, max=10.0)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        residual = self.mlp(torch.cat([theta, theta_dot, u_t], dim=1))
        tau_f = b * theta_dot
        theta_ddot = (self._motor(Gu, u_t) - self._gravity(theta) - tau_f + residual) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)


# Dicionário: nome → classe (factory)
MODELOS = {
    "Baseline":    PhysicsODE_Baseline,
    "Coulomb":     PhysicsODE_Coulomb,
    "Tustin":      PhysicsODE_Tustin,
    "Assimetrico": PhysicsODE_Asymmetric,
    "Hibrido":     PhysicsODE_Hybrid,
}


# ──────────────────────────────────────────────────────────────────────
# 3. DEFINIÇÃO DOS EXPERIMENTOS
# ──────────────────────────────────────────────────────────────────────
EXCITACOES = {
    "APRBS": [
        "aprbs-1_0827_17-19.csv",
        "aprbs-2_0827_17-25.csv",
        "aprbs-3_0827_17-28.csv",
        "aprbs-4_0827_17-31.csv",
    ],
    "MultiSeno": [
        "multi-seno-1_0827_17-34.csv",
        "multi-seno-2_0827_17-37.csv",
        "multi-seno-3_0827_17-40.csv",
        "multi-seno-4_0827_17-43.csv",
    ],
    "Varredura": [
        "swept-sine-1_0827_17-58.csv",
        "swept-sine-4_0827_18-00.csv",
        "RODADA-3/chirp-1_0807_16-34.csv",
        "RODADA-3/chirp-2_0807_17-07.csv",
        "RODADA-3/chirp-2_0807_17-09.csv",
    ],
    "Degraus": [
        "seq-degraus-1_0827_17-46.csv",
        "seq-degraus-2_0827_17-49.csv",
        "seq-degraus-3_0827_17-52.csv",
        "seq-degraus-4_0827_17-55.csv",
    ],
}
all_files = [f for files in EXCITACOES.values() for f in files]
EXCITACOES["Mix"] = list(dict.fromkeys(all_files))

VAL_FILES = ["RODADA-3/chirp-1_0807_16-32.csv"]

TEST_FILES_BY_TYPE = {
    "APRBS":     ["RODADA-4/aprbs-2_0819_18-51.csv"],
    "MultiSeno": ["RODADA-2/multi-seno-1_0804_19-06.csv"],
    "Varredura": ["RODADA-4/swept-sine-1_0819_19-02.csv",
                   "RODADA-2/chirp-1_0804_19-19.csv"],
    "Degraus":   ["RODADA-2/seq-degraus-2_0804_19-38.csv"],
}


# ──────────────────────────────────────────────────────────────────────
# 4. TREINAMENTO E AVALIAÇÃO
# ──────────────────────────────────────────────────────────────────────
def avalia_free_run(model, datasets, integrator='rk4'):
    """Simulação livre completa: condição inicial real, u real, sem correção."""
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
               epochs=1500, lr=0.015,
               k_min=20, k_max=400, curriculum_stage_epochs=300,
               base_batch_size=1024, integrator='rk4', patience=500):
    print(f"  Treinando {name} ({model.n_params()} params, {epochs} épocas)...")
    model.to(device)

    # Optimizer: weight decay separado na MLP do Híbrido
    if hasattr(model, 'mlp'):
        mlp_p  = [p for n, p in model.named_parameters() if 'mlp' in n]
        phys_p = [p for n, p in model.named_parameters() if 'mlp' not in n]
        optimizer = optim.Adam([
            {'params': phys_p,  'weight_decay': 0.0},
            {'params': mlp_p,   'weight_decay': 1e-4},
        ], lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    all_pos   = np.concatenate([d['x'][:, 0].numpy() for d in train_datasets])
    all_vel   = np.concatenate([d['x'][:, 1].numpy() for d in train_datasets])
    state_std = torch.tensor([float(np.std(all_pos)), float(np.std(all_vel))],
                              dtype=torch.float32, device=device)

    dt = (train_datasets[0]['t'][1] - train_datasets[0]['t'][0]).item()

    best_val   = val_rmse(model, val_datasets)
    best_sd    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

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

        total_loss = total_loss / len(train_datasets) + model.reg_loss()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        # Val check a cada 50 épocas
        if epoch % 50 == 0 or epoch == epochs:
            val = val_rmse(model, val_datasets)
            if val < best_val:
                best_val   = val
                best_sd    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 50

        # Early stopping (não na 1ª stage — modelo precisa estabilizar)
        if no_improve >= patience and epoch > curriculum_stage_epochs:
            print(f"    Early stop epoch {epoch} (patience={patience})")
            break

        if epoch % 300 == 0 or epoch == epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch:4d} | k={k_steps:3d} | LR={lr_now:.5f} | "
                  f"Train={total_loss.item():.4f} | Best Val={best_val:.3f}°")

    print(f"    -> Melhor Val RMSE: {best_val:.3f}° — restaurando checkpoint.")
    model.load_state_dict(best_sd)
    return model, best_val


# ──────────────────────────────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ──────────────────────────────────────────────────────────────────────
def plot_heatmap(data_2d, row_labels, col_labels, metrica, title, out_path=None):
    """Heatmap genérico: linhas=modelos, colunas=excitações."""
    fig, ax = plt.subplots(figsize=(len(col_labels)*1.6 + 1.5,
                                    len(row_labels)*0.9 + 1.5))
    cmap = 'RdYlGn' if metrica != 'rmse' else 'RdYlGn_r'
    im = ax.imshow(data_2d, aspect='auto', cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel('Excitação', fontsize=10)
    ax.set_ylabel('Modelo', fontsize=10)
    ax.set_title(title, fontsize=12)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = data_2d[i, j]
            fmt = f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"
            ax.text(j, i, fmt, ha='center', va='center', fontsize=8,
                    color='white' if abs(v) > np.mean(np.abs(data_2d)) * 1.5 else 'black')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_barchart(names, values_dict, title, out_path=None):
    """Bar chart com múltiplas métricas lado a lado."""
    n = len(names)
    fig, axes = plt.subplots(1, len(values_dict), figsize=(5*len(values_dict), 4.5))
    if len(values_dict) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)
    colors = ['steelblue', 'seagreen', 'darkorange']
    for ax, (label, vals), color in zip(axes, values_dict.items(), colors):
        x = np.arange(n)
        bars = ax.bar(x, vals, color=color, alpha=0.85, edgecolor='black', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis='y', lw=0.4)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01*(max(vals)-min(vals)+1e-9),
                    fmt, ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


def plot_free_run(resultado, titulo, out_path=None):
    """Free-run: sobreposição real vs predito."""
    n    = len(resultado)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 3.5*rows))
    fig.suptitle(titulo, fontsize=13)
    axs = np.array(axs).reshape(rows, cols) if n > 1 else np.array([[axs]])

    for i, r in enumerate(resultado):
        ax = axs[i // cols, i % cols]
        ax.plot(r['t'], r['y_real'], 'k',   lw=1.2, label='Real')
        ax.plot(r['t'], r['y_pred'], 'r--', lw=1.0, label='Pred')
        ax.set_title(
            f"{r['name']}\nRMSE={r['rmse']:.2f}° R²={r['r2']:.3f} FIT={r['fit']:.1f}%",
            fontsize=8)
        ax.set_xlabel('Tempo (s)', fontsize=7)
        ax.set_ylabel('Ângulo (°)', fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.4)

    for i in range(n, rows*cols):
        axs[i // cols, i % cols].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if out_path:
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


def plot_datasets(datasets, title, out_path=None):
    """Plota sinais de cada dataset: ângulo e entrada vs tempo."""
    n    = len(datasets)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows * 2, cols, figsize=(6 * cols, 3.5 * rows))
    fig.suptitle(title, fontsize=13)
    if rows * 2 * cols == 1:
        axs = np.array([[axs]])
    axs = np.array(axs).reshape(rows * 2, cols)

    for i, ds in enumerate(datasets):
        r, c  = (i // cols) * 2, i % cols
        t_np  = ds['t'].numpy()
        y_deg = ds['x'][:, 0].numpy() * (180 / np.pi)
        u_pct = ds['u'][:, 0].numpy() * 100.0

        axs[r, c].plot(t_np, y_deg, color='steelblue', lw=0.8)
        axs[r, c].set_title(f"{ds['name']} ({len(t_np)} pts, {t_np[-1]:.1f}s)", fontsize=8)
        axs[r, c].set_ylabel('Ângulo (°)', fontsize=7)
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


# ──────────────────────────────────────────────────────────────────────
# 6. MAIN — Loop fatorial 5 × 5
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = f"resultados_v7_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Resultados em: {out_dir}/\n")

    # ── Carregar val e teste (uma única vez) ──
    print("Carregando Val e Teste...")
    val_datasets = carregar_lista(VAL_FILES)
    test_por_tipo = {tipo: carregar_lista(arqs)
                     for tipo, arqs in TEST_FILES_BY_TYPE.items()}

    # ── Pre-carregar dados de treino por excitação ──
    exc_datasets = {}
    for exc_nome, files in EXCITACOES.items():
        print(f"  Excitação {exc_nome}: {len(files)} arquivo(s)")
        exc_datasets[exc_nome] = carregar_lista(files)

    mod_nomes  = list(MODELOS.keys())
    exc_nomes  = list(EXCITACOES.keys())
    test_tipos = list(TEST_FILES_BY_TYPE.keys())

    # ── Plotar datasets de treino e teste (antes do loop) ──
    print("\nPlotando datasets de treino e teste...")
    for exc_nome, ds_list in exc_datasets.items():
        plot_datasets(ds_list, f'Treino — {exc_nome}',
                      out_path=f"{out_dir}/dados_treino_{exc_nome}.png")
        print(f"  dados_treino_{exc_nome}.png ({len(ds_list)} arquivos)")

    # Val
    plot_datasets(val_datasets, 'Validação (Early Stopping)',
                  out_path=f"{out_dir}/dados_val.png")
    print(f"  dados_val.png")

    # Teste por tipo
    all_test_ds = []
    for tipo, ds_list in test_por_tipo.items():
        all_test_ds.extend(ds_list)
    plot_datasets(all_test_ds, 'Teste (cross-session)',
                  out_path=f"{out_dir}/dados_teste.png")
    print(f"  dados_teste.png ({len(all_test_ds)} arquivos)")

    # resultados[mod][exc] = {'por_tipo': {...}, 'media': {...}, ...}
    resumo     = {}
    total_runs = len(mod_nomes) * len(exc_nomes)
    run_i      = 0
    t_start    = time.time()

    # ── Loop fatorial ──
    for mod_nome, ModelClass in MODELOS.items():
        resumo[mod_nome] = {}

        for exc_nome in exc_nomes:
            run_i += 1
            tag = f"{mod_nome}_{exc_nome}"
            elapsed = time.time() - t_start
            eta = (elapsed / run_i) * (total_runs - run_i) if run_i > 0 else 0

            print(f"\n{'='*60}")
            print(f"  [{run_i}/{total_runs}] {tag}  "
                  f"(elapsed {elapsed/60:.1f}min | ETA {eta/60:.1f}min)")
            print(f"{'='*60}")

            # Resetar seed para reprodutibilidade
            torch.manual_seed(0)
            np.random.seed(0)
            model = ModelClass()

            model, best_val = train_node(
                model, tag, exc_datasets[exc_nome], val_datasets,
                epochs=1500, lr=0.015,
                k_min=20, k_max=400, curriculum_stage_epochs=300,
            )

            torch.save(model.state_dict(), f"{out_dir}/model_{tag}.pth")

            # Avaliar em cada tipo de teste
            por_tipo = {}
            todos_res = []
            for test_tipo, test_ds in test_por_tipo.items():
                res = avalia_free_run(model, test_ds)
                todos_res.extend(res)
                rmse_m = np.mean([r['rmse'] for r in res])
                r2_m   = np.mean([r['r2']   for r in res])
                fit_m  = np.mean([r['fit']  for r in res])
                por_tipo[test_tipo] = {'rmse': round(rmse_m, 4),
                                       'r2':   round(r2_m, 4),
                                       'fit':  round(fit_m, 2)}
                print(f"  [TESTE {test_tipo:<10}] RMSE={rmse_m:.3f}° "
                      f"R²={r2_m:.4f} FIT={fit_m:.1f}%")

            # Média geral
            rmse_g = np.mean([v['rmse'] for v in por_tipo.values()])
            r2_g   = np.mean([v['r2']   for v in por_tipo.values()])
            fit_g  = np.mean([v['fit']  for v in por_tipo.values()])
            resumo[mod_nome][exc_nome] = {
                'por_tipo': por_tipo,
                'media': {'RMSE': round(rmse_g, 4),
                          'R2':   round(r2_g, 4),
                          'FIT':  round(fit_g, 2)},
                'best_val': round(best_val, 4),
                'n_params': model.n_params(),
            }
            print(f"  [MÉDIA GERAL] RMSE={rmse_g:.3f}° R²={r2_g:.4f} FIT={fit_g:.1f}%")

            # Free-run plot
            plot_free_run(todos_res, tag,
                          out_path=f"{out_dir}/freerun_{tag}.png")

            # Salvar JSON incremental (evita perder tudo se crashar)
            with open(f"{out_dir}/resumo.json", 'w', encoding='utf-8') as fp:
                json.dump(resumo, fp, indent=2, ensure_ascii=False)

            # Liberar memória
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ────────────────── VISUALIZAÇÕES FINAIS ──────────────────
    print(f"\n{'─'*60}")
    print("  Gerando visualizações finais...")
    print(f"{'─'*60}")

    # Heatmaps: média geral (modelo × excitação)
    for metrica, label in [('RMSE', 'RMSE (°)'), ('R2', 'R²'), ('FIT', 'FIT%')]:
        data = np.array([[resumo[m][e]['media'][metrica]
                          for e in exc_nomes] for m in mod_nomes])
        plot_heatmap(data, mod_nomes, exc_nomes,
                     'rmse' if metrica == 'RMSE' else metrica.lower(),
                     f'Média Geral — {label}',
                     out_path=f"{out_dir}/heatmap_media_{metrica.lower()}.png")
        print(f"  Heatmap salvo: heatmap_media_{metrica.lower()}.png")

    # Heatmaps: por tipo de teste (RMSE)
    for test_tipo in test_tipos:
        data = np.array([[resumo[m][e]['por_tipo'][test_tipo]['rmse']
                          for e in exc_nomes] for m in mod_nomes])
        plot_heatmap(data, mod_nomes, exc_nomes, 'rmse',
                     f'RMSE — Teste {test_tipo}',
                     out_path=f"{out_dir}/heatmap_{test_tipo}_rmse.png")
        print(f"  Heatmap salvo: heatmap_{test_tipo}_rmse.png")

    # Barchart: marginal por modelo (média sobre excitações)
    mod_rmse = [np.mean([resumo[m][e]['media']['RMSE'] for e in exc_nomes]) for m in mod_nomes]
    mod_r2   = [np.mean([resumo[m][e]['media']['R2']   for e in exc_nomes]) for m in mod_nomes]
    mod_fit  = [np.mean([resumo[m][e]['media']['FIT']  for e in exc_nomes]) for m in mod_nomes]
    plot_barchart(mod_nomes,
                  {'RMSE (°)': mod_rmse, 'R²': mod_r2, 'FIT%': mod_fit},
                  'Comparativo por Modelo (média sobre excitações)',
                  out_path=f"{out_dir}/barchart_por_modelo.png")
    print(f"  Barchart salvo: barchart_por_modelo.png")

    # Barchart: marginal por excitação (média sobre modelos)
    exc_rmse = [np.mean([resumo[m][e]['media']['RMSE'] for m in mod_nomes]) for e in exc_nomes]
    exc_r2   = [np.mean([resumo[m][e]['media']['R2']   for m in mod_nomes]) for e in exc_nomes]
    exc_fit  = [np.mean([resumo[m][e]['media']['FIT']  for m in mod_nomes]) for e in exc_nomes]
    plot_barchart(exc_nomes,
                  {'RMSE (°)': exc_rmse, 'R²': exc_r2, 'FIT%': exc_fit},
                  'Comparativo por Excitação (média sobre modelos)',
                  out_path=f"{out_dir}/barchart_por_excitacao.png")
    print(f"  Barchart salvo: barchart_por_excitacao.png")

    # ────────────────── TABELA CONSOLE ──────────────────
    print(f"\n{'='*80}")
    print("  RESUMO — RMSE MÉDIA (°) | Modelo × Excitação")
    print(f"{'='*80}")
    header = f"{'Modelo':<14}" + "".join(f" {e:>10}" for e in exc_nomes) + f" {'MÉDIA':>10}"
    print(header)
    print('─' * len(header))
    for m in mod_nomes:
        vals = [resumo[m][e]['media']['RMSE'] for e in exc_nomes]
        media = np.mean(vals)
        linha = f"{m:<14}" + "".join(f" {v:>10.3f}" for v in vals) + f" {media:>10.3f}"
        print(linha)

    # Melhor combinação
    best_combo = min(
        ((m, e) for m in mod_nomes for e in exc_nomes),
        key=lambda x: resumo[x[0]][x[1]]['media']['RMSE']
    )
    best_rmse = resumo[best_combo[0]][best_combo[1]]['media']['RMSE']
    total_time = (time.time() - t_start) / 60
    print(f"\n★ Melhor combo: {best_combo[0]} + {best_combo[1]} → RMSE={best_rmse:.3f}°")
    print(f"  Tempo total: {total_time:.1f} minutos")

    # Salvar JSON
    with open(f"{out_dir}/resumo.json", 'w', encoding='utf-8') as fp:
        json.dump(resumo, fp, indent=2, ensure_ascii=False)
    print(f"\nResultados em: {out_dir}/")
    print("[V7] Design fatorial concluído.")
