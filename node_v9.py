# %% [markdown]
# # NODE v9 — Física Aerodinâmica Assimétrica + Chunk Validation
#
# Avaliação de 4 arquiteturas físicas super-equipadas (arrasto, assimetria direcional) + Chunk Validation.
# Cada modelo avaliado em 4 tipos de teste cross-session.
#
# Modelos:
#   AeroBaseline — viscoso + arrasto quadrático assimétricos (7 params)
#   AeroCoulomb  — AeroBaseline + Coulomb assimétrico (9 params)
#   AeroTustin   — AeroCoulomb + Stribeck assimétrico (12 params)
#   AeroHibrido  — AeroBaseline + MLP (7 + 84 params)
#
# Excitações:
#   APRBS, Multi-seno, Varredura, Degraus, Mix
#
# Validação (early stopping):
#   Chirp broadband da Rodada 3 — mesmo critério para todos os 25 runs.
#   Justificativa: chirp testa ampla faixa de frequências,
#   e early stopping é regularização, não critério final.
#   Patience de 500 épocas para economia de tempo computacional.
#
# Dica de Performance (Paralelismo):
# Como os modelos ODE físicos são extremamente leves (7 a 84 params), o cálculo
# gera muito overhead de CPU, deixando a GPU (VRAM e Compute) ociosa (ex: <20% de uso).
# Portanto, você pode abrir 2 a 4 terminais rodando este script simultaneamente
# (ex: testando hiperparâmetros ou ablations) sem medo de estourar a VRAM!

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

DECIMACAO  = 1
START_IDX  = 150
END_IDX    = -1


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


def carregar_lista(file_list, to_device=None):
    datasets = []
    for f in file_list:
        t, u, y = carregar_experimento(BASE + f)
        t_ten, u_ten, x_ten, *_ = processar_dataset(t, u, y)
        if to_device is not None:
            datasets.append({'name': os.path.basename(f),
                             't': t_ten.to(to_device),
                             'u': u_ten.to(to_device),
                             'x': x_ten.to(to_device)})
        else:
            datasets.append({'name': os.path.basename(f), 't': t_ten, 'u': u_ten, 'x': x_ten})
    return datasets


# ──────────────────────────────────────────────────────────────────────
# 2. MODELOS — 4 arquiteturas físicas (Aerodinâmicas e Assimétricas)
# ──────────────────────────────────────────────────────────────────────
class BaseODE(nn.Module):
    """Infraestrutura compartilhada: gravidade, interpolação de u, clamping."""
    def __init__(self, J0=1.0, b_pos0=np.exp(-1.0), b_neg0=np.exp(-1.0), Gu_pos0=1.0, Gu_neg0=1.0, c_pos0=0.01, c_neg0=0.01):
        super().__init__()
        self.m1, self.L1 = 0.122, 0.39
        self.m2, self.L2 = 0.055, 0.347
        self.g = 9.81
        self.log_J  = nn.Parameter(torch.log(torch.tensor(float(J0))))
        self.log_b_pos  = nn.Parameter(torch.log(torch.tensor(float(b_pos0))))
        self.log_b_neg  = nn.Parameter(torch.log(torch.tensor(float(b_neg0))))
        self.log_Gu_pos = nn.Parameter(torch.log(torch.tensor(float(Gu_pos0))))
        self.log_Gu_neg = nn.Parameter(torch.log(torch.tensor(float(Gu_neg0))))
        self.log_c_pos  = nn.Parameter(torch.log(torch.tensor(float(c_pos0))))
        self.log_c_neg  = nn.Parameter(torch.log(torch.tensor(float(c_neg0))))
        self.u_series = None
        self.t_series = None
        self.batch_start_times = None

    def get_params(self):
        return torch.exp(self.log_J)

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

    def _motor(self, u_t):
        Gu_pos = torch.clamp(torch.exp(self.log_Gu_pos), min=0.01, max=10.0)
        Gu_neg = torch.clamp(torch.exp(self.log_Gu_neg), min=0.01, max=10.0)
        sigma_Gu = torch.sigmoid(50.0 * u_t)
        Gu = sigma_Gu * Gu_pos + (1.0 - sigma_Gu) * Gu_neg
        return Gu * u_t * torch.abs(u_t)

    def _aero_viscous(self, theta_dot):
        b_pos = torch.clamp(torch.exp(self.log_b_pos), min=0.001, max=1.0)
        b_neg = torch.clamp(torch.exp(self.log_b_neg), min=0.001, max=1.0)
        c_pos = torch.clamp(torch.exp(self.log_c_pos), min=0.0001, max=1.0)
        c_neg = torch.clamp(torch.exp(self.log_c_neg), min=0.0001, max=1.0)
        
        sigma_b = torch.sigmoid(50.0 * theta_dot)
        b = sigma_b * b_pos + (1.0 - sigma_b) * b_neg
        c = sigma_b * c_pos + (1.0 - sigma_b) * c_neg
        
        return b * theta_dot + c * theta_dot * torch.abs(theta_dot)

    def reg_loss(self):
        """Override nas subclasses para regularizar parâmetros difíceis."""
        return 0.01 * (self.log_b_neg**2 + self.log_Gu_neg**2 + self.log_c_neg**2 + self.log_c_pos**2)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhysicsODE_AeroBaseline(BaseODE):
    """Atrito viscoso assimétrico + Arrasto assimétrico. 7 params."""
    def forward(self, t, x):
        J  = torch.clamp(torch.exp(self.log_J),  min=0.01, max=1.0)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        tau_f = self._aero_viscous(theta_dot)
        theta_ddot = (self._motor(u_t) - self._gravity(theta) - tau_f) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)


class PhysicsODE_AeroCoulomb(BaseODE):
    """Baseline Aero + Coulomb assimétrico. 9 params."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_Tc_pos = nn.Parameter(torch.log(torch.tensor(0.01)))
        self.log_Tc_neg = nn.Parameter(torch.log(torch.tensor(0.01)))

    def forward(self, t, x):
        J  = torch.clamp(torch.exp(self.log_J),  min=0.01, max=1.0)
        Tc_pos = torch.clamp(torch.exp(self.log_Tc_pos), min=0.001, max=1.0)
        Tc_neg = torch.clamp(torch.exp(self.log_Tc_neg), min=0.001, max=1.0)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        
        sigma_Tc = torch.sigmoid(50.0 * theta_dot)
        Tc = sigma_Tc * Tc_pos + (1.0 - sigma_Tc) * Tc_neg
        
        tau_f = self._aero_viscous(theta_dot) + Tc * torch.tanh(50.0 * theta_dot)
        theta_ddot = (self._motor(u_t) - self._gravity(theta) - tau_f) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)

    def reg_loss(self):
        return super().reg_loss() + 0.005 * (self.log_Tc_pos**2 + self.log_Tc_neg**2)


class PhysicsODE_AeroTustin(BaseODE):
    """Baseline Aero + Stribeck assimétrico completo. 12 params."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_Tc_pos = nn.Parameter(torch.log(torch.tensor(0.01)))
        self.log_Tc_neg = nn.Parameter(torch.log(torch.tensor(0.01)))
        self.log_Ts_pos = nn.Parameter(torch.log(torch.tensor(0.02)))
        self.log_Ts_neg = nn.Parameter(torch.log(torch.tensor(0.02)))
        self.log_vs = nn.Parameter(torch.log(torch.tensor(0.1)))

    def forward(self, t, x):
        J  = torch.clamp(torch.exp(self.log_J),  min=0.01, max=1.0)
        Tc_pos = torch.clamp(torch.exp(self.log_Tc_pos), min=0.001, max=1.0)
        Tc_neg = torch.clamp(torch.exp(self.log_Tc_neg), min=0.001, max=1.0)
        Ts_pos = torch.clamp(torch.exp(self.log_Ts_pos), min=0.001, max=1.0)
        Ts_neg = torch.clamp(torch.exp(self.log_Ts_neg), min=0.001, max=1.0)
        vs = torch.clamp(torch.exp(self.log_vs), min=0.001, max=1.0)
        
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        
        sigma = torch.sigmoid(50.0 * theta_dot)
        Tc = sigma * Tc_pos + (1.0 - sigma) * Tc_neg
        Ts = sigma * Ts_pos + (1.0 - sigma) * Ts_neg
        
        stribeck = Tc + (Ts - Tc) * torch.exp(-torch.abs(theta_dot / vs))
        tau_f = self._aero_viscous(theta_dot) + stribeck * torch.tanh(50.0 * theta_dot)
        
        theta_ddot = (self._motor(u_t) - self._gravity(theta) - tau_f) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)

    def reg_loss(self):
        return super().reg_loss() + 0.005 * (self.log_Tc_pos**2 + self.log_Tc_neg**2 + self.log_Ts_pos**2 + self.log_Ts_neg**2 + self.log_vs**2)


class PhysicsODE_AeroHibrido(BaseODE):
    """AeroBaseline + MLP residual. 7 params físicos + MLP."""
    def __init__(self, hidden_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t, x):
        J  = torch.clamp(torch.exp(self.log_J),  min=0.01, max=1.0)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        residual = self.mlp(torch.cat([theta, theta_dot, u_t], dim=1))
        tau_f = self._aero_viscous(theta_dot)
        theta_ddot = (self._motor(u_t) - self._gravity(theta) - tau_f + residual) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)


# Dicionário: nome → classe (factory)
MODELOS = {
    "AeroBaseline": PhysicsODE_AeroBaseline,
    "AeroCoulomb":  PhysicsODE_AeroCoulomb,
    "AeroTustin":   PhysicsODE_AeroTustin,
    "AeroHibrido":  PhysicsODE_AeroHibrido,
}

# ──────────────────────────────────────────────────────────────────────
# 3. DEFINIÇÃO DOS EXPERIMENTOS
# ──────────────────────────────────────────────────────────────────────
ALL_TRAIN_FILES = [
    "RODADA-7/MIX_DC_45.csv",
    "RODADA-7/MIX_DC_60.csv"
]



VAL_FILES = [
    "RODADA-5/multi-seno-3_0827_17-40.csv",
]

TEST_FILES_BY_TYPE = {
    "APRBS":     ["RODADA-5/aprbs-1_0827_17-19.csv", "RODADA-5/aprbs-2_0827_17-25.csv"],
    "MultiSeno": ["RODADA-5/multi-seno-1_0827_17-34.csv", "RODADA-5/multi-seno-2_0827_17-37.csv"],
    "Varredura": ["RODADA-2/chirp-1_0804_19-17.csv", "RODADA-3/chirp-1_0807_16-34.csv"],
    "Degraus":   ["RODADA-5/seq-degraus-3_0827_17-52.csv", "RODADA-5/seq-degraus-4_0827_17-55.csv"],
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
            # Datasets já estão no device (GPU) — sem cópia extra
            t_t = ds['t']
            u_t = ds['u']
            x_t = ds['x']
            model.t_series          = t_t
            model.u_series          = u_t
            model.batch_start_times = torch.zeros(1, 1, device=t_t.device)
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
                'y_real': y_real, 'y_pred': y_pred, 't': ds['t'].cpu().numpy(),
            })
    return resultados


def avalia_chunks(model, datasets, k_steps=400, integrator='rk4'):
    """Avaliação por chunks: simula janelas de k_steps com CI real.
    
    Monta o sinal predito completo concatenando os chunks (sem sobreposição)
    para comparação direta com o sinal real de treino.
    """
    model.eval()
    resultados = []
    with torch.no_grad():
        for ds in datasets:
            t_ds = ds['t']
            u_ds = ds['u']
            x_ds = ds['x']
            model.t_series = t_ds
            model.u_series = u_ds

            dt = float(t_ds[1] - t_ds[0])
            N  = len(t_ds)

            # Chunks sem sobreposição, varrendo todo o sinal
            y_pred_full = np.full(N, np.nan)
            n_chunks = max(1, N // k_steps)
            chunk_starts = np.arange(0, N - k_steps + 1, k_steps)
            if len(chunk_starts) == 0:
                chunk_starts = np.array([0])

            for s in chunk_starts:
                k_actual = min(k_steps, N - s)
                t_eval = torch.arange(0, k_actual * dt, dt, device=t_ds.device)[:k_actual]
                x0 = x_ds[s].unsqueeze(0)
                model.batch_start_times = t_ds[s].reshape(1, 1)

                pred = odeint(model, x0, t_eval, method=integrator).squeeze(1)
                y_pred_full[s:s + k_actual] = pred[:, 0].cpu().numpy() * (180 / np.pi)

            # Preencher trecho final se sobrou
            last_end = chunk_starts[-1] + k_steps
            if last_end < N:
                k_tail = N - last_end
                t_eval = torch.arange(0, k_tail * dt, dt, device=t_ds.device)[:k_tail]
                x0 = x_ds[last_end].unsqueeze(0)
                model.batch_start_times = t_ds[last_end].reshape(1, 1)
                pred = odeint(model, x0, t_eval, method=integrator).squeeze(1)
                y_pred_full[last_end:N] = pred[:, 0].cpu().numpy() * (180 / np.pi)

            y_real = x_ds[:, 0].cpu().numpy() * (180 / np.pi)
            mask   = ~np.isnan(y_pred_full)
            y_r = y_real[mask]
            y_p = y_pred_full[mask]

            rmse = float(np.sqrt(np.mean((y_p - y_r) ** 2)))
            ss_res = np.sum((y_r - y_p) ** 2)
            ss_tot = np.sum((y_r - np.mean(y_r)) ** 2)
            r2   = float(1 - ss_res / (ss_tot + 1e-12))
            fit  = float((1 - np.linalg.norm(y_p - y_r) /
                          (np.linalg.norm(y_r - np.mean(y_r)) + 1e-12)) * 100)

            resultados.append({
                'name': ds['name'], 'rmse': rmse, 'r2': r2, 'fit': fit,
                'y_real': y_real, 'y_pred': y_pred_full,
                't': t_ds.cpu().numpy(),
                'chunk_starts': chunk_starts,
                'k_steps': k_steps,
            })
    return resultados


def val_rmse(model, val_datasets, k_steps=None):
    if k_steps is None:
        # Full free-run evaluation
        res = avalia_free_run(model, val_datasets)
        return np.mean([r['rmse'] for r in res])
    
    # Chunk-based evaluation
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for ds in val_datasets:
            # Datasets já estão no device (GPU)
            t_ds = ds['t']
            u_ds = ds['u']
            x_ds = ds['x']
            model.t_series = t_ds
            model.u_series = u_ds
            
            # Use deterministic evenly spaced chunks
            num_chunks = max(1, len(t_ds) // k_steps)
            if num_chunks == 0:
                continue
            start_indices = np.linspace(0, len(t_ds) - k_steps - 1, num_chunks, dtype=int)
            
            x0 = x_ds[start_indices]
            model.batch_start_times = t_ds[start_indices].reshape(-1, 1)
            
            dt = float(t_ds[1]-t_ds[0])
            t_eval = torch.arange(0, k_steps * dt, dt, device=t_ds.device)[:k_steps]
            
            pred_state = odeint(model, x0, t_eval, method='rk4')
            batch_targets = torch.stack([x_ds[i:i + k_steps] for i in start_indices], dim=1)
            
            # RMSE in degrees
            pred_deg = pred_state[:, :, 0] * (180 / np.pi)
            target_deg = batch_targets[:, :, 0] * (180 / np.pi)
            
            loss = torch.sqrt(torch.mean((pred_deg - target_deg)**2))
            total_loss += loss.item()
            count += 1
            
    return total_loss / count if count > 0 else 999.0


def train_node(model, name, train_datasets, val_datasets,
               epochs=3000, lr=0.015, k_steps=400,
               batch_size=128, integrator='rk4', patience=500):
    print(f"  Treinando {name} ({model.n_params()} params, {epochs} épocas, k={k_steps})...")
    model.to(device)

    # Optimizer: weight decay separado na MLP do Híbrido
    if hasattr(model, 'mlp'):
        mlp_p  = [p for n, p in model.named_parameters() if 'mlp' in n]
        phys_p = [p for n, p in model.named_parameters() if 'mlp' not in n]
        optimizer = optim.Adam([
            {'params': phys_p,  'weight_decay': 1e-4},
            {'params': mlp_p,   'weight_decay': 1e-4},
        ], lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    all_pos   = np.concatenate([d['x'][:, 0].cpu().numpy() for d in train_datasets])
    all_vel   = np.concatenate([d['x'][:, 1].cpu().numpy() for d in train_datasets])
    state_std = torch.tensor([float(np.std(all_pos)), float(np.std(all_vel))],
                              dtype=torch.float32, device=device)

    dt = (train_datasets[0]['t'][1] - train_datasets[0]['t'][0]).item()
    t_eval = torch.arange(0, k_steps * dt, dt, device=device)[:k_steps]

    samples_per_ds = max(1, batch_size // len(train_datasets))

    best_val   = float('inf')
    best_sd    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    nan_total  = 0
    nan_window = []  # Rolling window das últimas 100 epochs (True=NaN)

    # Validação inicial
    v = val_rmse(model, val_datasets, k_steps=k_steps)
    if not np.isnan(v):
        best_val = v
    print(f"    Val inicial: {best_val:.3f}°")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0

        for ds in train_datasets:
            t_ds, u_ds, x_ds = ds['t'], ds['u'], ds['x']
            model.t_series, model.u_series = t_ds, u_ds

            start_idx = np.random.randint(0, len(t_ds) - k_steps, size=samples_per_ds)
            x0 = x_ds[start_idx]
            model.batch_start_times = t_ds[start_idx].reshape(-1, 1)

            pred_state    = odeint(model, x0, t_eval, method=integrator)
            batch_targets = torch.stack([x_ds[i:i + k_steps] for i in start_idx], dim=1)
            loss = torch.mean(((pred_state - batch_targets) / state_std) ** 2)
            
            # Penalidade de estabilidade
            pred_theta = pred_state[:, :, 0]
            penalty = torch.mean(torch.relu(pred_theta - 3.14)**2 + torch.relu(-0.5 - pred_theta)**2) * 1e4
            total_loss += (loss + penalty)

        total_loss = total_loss / len(train_datasets) + model.reg_loss()

        # NaN guard: janela rolante — aborta se >50% das últimas 100 epochs forem NaN
        is_nan = bool(torch.isnan(total_loss) or torch.isinf(total_loss))
        nan_window.append(is_nan)
        if len(nan_window) > 100:
            nan_window.pop(0)

        if is_nan:
            nan_total += 1
            if nan_total == 1 or nan_total % 50 == 0:
                print(f"    ⚠ NaN epoch {epoch} (total NaN={nan_total}). Restaurando checkpoint.")
            
            if len(nan_window) >= 100 and sum(nan_window) > 50:
                print(f"    ✖ {sum(nan_window)}/100 epochs com NaN. Modelo preso — abortando.")
                break
            
            model.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
            optimizer.zero_grad()
            scheduler.step()
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        # Val check + early stopping a cada 50 epochs
        if epoch % 50 == 0 or epoch == epochs:
            val = val_rmse(model, val_datasets, k_steps=k_steps)
            if not np.isnan(val) and val < best_val:
                best_val   = val
                best_sd    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 50

        if no_improve >= patience:
            print(f"    Early stop epoch {epoch} (patience={patience})")
            break

        if epoch % 300 == 0 or epoch == epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch:4d} | k={k_steps} | LR={lr_now:.5f} | "
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


def plot_chunk_fit(resultado, titulo, out_path=None):
    """Chunk fit: sinal real completo + predição por chunks sobrepostos.
    
    Mostra linhas verticais tracejadas em cada início de chunk para
    visualizar onde o modelo reinicia com CI real.
    """
    n    = len(resultado)
    fig, axs = plt.subplots(n, 1, figsize=(14, 4 * n))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')
    if n == 1:
        axs = [axs]

    for i, r in enumerate(resultado):
        ax = axs[i]
        t = r['t']
        ax.plot(t, r['y_real'], 'k', lw=1.0, label='Real', alpha=0.85)
        ax.plot(t, r['y_pred'], 'r', lw=1.2, label=f'Pred (chunks k={r["k_steps"]})', alpha=0.9)

        # Marcar início de cada chunk
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        for s in r['chunk_starts']:
            ax.axvline(t[s], color='blue', lw=0.5, ls=':', alpha=0.4)

        ax.set_title(
            f"{r['name']}  —  RMSE={r['rmse']:.2f}°  R²={r['r2']:.3f}  FIT={r['fit']:.1f}%",
            fontsize=10)
        ax.set_xlabel('Tempo (s)', fontsize=9)
        ax.set_ylabel('Ângulo (°)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, lw=0.3, alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
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
# 6. MAIN — Sequencial + GPU optimizado
# ──────────────────────────────────────────────────────────────────────

def _setup_gpu():
    """Configurações de GPU para máximo desempenho."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    import datetime, os, json, time
    import numpy as np
    import torch

    _setup_gpu()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = f"resultados_v9_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Resultados em: {out_dir}/\n")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} ({vram_total:.1f} GB VRAM)\n")

    # ── Carregar dados na GPU (uma única vez, sem duplicação) ──
    print("Carregando Val na GPU...")
    val_datasets  = carregar_lista(VAL_FILES, to_device=device)

    print(f"Carregando {len(ALL_TRAIN_FILES)} arquivos de treino na GPU...")
    train_datasets = carregar_lista(ALL_TRAIN_FILES, to_device=device)

    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"  VRAM usada pelos datasets: {vram_mb:.1f} MB")

    mod_nomes  = list(MODELOS.keys())

    # ── Plotar datasets (CPU temporário só para matplotlib) ──
    print("\nPlotando datasets de treino e validação...")
    train_ds_cpu = [{'name': d['name'], 't': d['t'].cpu(), 'u': d['u'].cpu(), 'x': d['x'].cpu()} for d in train_datasets]
    val_ds_cpu   = [{'name': d['name'], 't': d['t'].cpu(), 'u': d['u'].cpu(), 'x': d['x'].cpu()} for d in val_datasets]

    plot_datasets(train_ds_cpu, 'Treino — Dataset Completo',
                  out_path=f"{out_dir}/dados_treino.png")
    plot_datasets(val_ds_cpu, 'Validação (Early Stopping)',
                  out_path=f"{out_dir}/dados_val.png")
    del train_ds_cpu, val_ds_cpu

    t_start = time.time()
    n_total = len(mod_nomes)
    print(f"\nIniciando treinamento sequencial de {n_total} modelos (GPU-resident)...")

    resumo = {}
    for i, mod_nome in enumerate(mod_nomes, 1):
        elapsed = (time.time() - t_start) / 60
        eta = (elapsed / max(i-1, 1)) * (n_total - i + 1) if i > 1 else 0
        print(f"\n{'='*60}")
        print(f"  [{i}/{n_total}] {mod_nome}_Global  (elapsed {elapsed:.1f}min | ETA {eta:.1f}min)")
        print(f"{'='*60}")

        ModelClass = MODELOS[mod_nome]
        tag = f"{mod_nome}_Global"

        torch.manual_seed(42)
        np.random.seed(42)
        model = ModelClass()

        compiled_model = model

        compiled_model, best_val = train_node(
            compiled_model, tag, train_datasets, val_datasets,
            epochs=3000, lr=0.015, k_steps=400, batch_size=128, patience=1000
        )

        # Salvar state_dict do modelo original (não compilado)
        if hasattr(compiled_model, '_orig_mod'):
            save_model = compiled_model._orig_mod
        else:
            save_model = compiled_model
        torch.save(save_model.state_dict(), f"{out_dir}/model_{tag}.pth")

        # ── Avaliação por CHUNKS nos dados de TREINO (MIX) ──
        chunk_res = avalia_chunks(compiled_model, train_datasets, k_steps=400)
        rmse_c = np.mean([r['rmse'] for r in chunk_res])
        r2_c   = np.mean([r['r2']   for r in chunk_res])
        fit_c  = np.mean([r['fit']  for r in chunk_res])

        for r in chunk_res:
            print(f"  [{tag} TREINO {r['name']:<30}] RMSE={r['rmse']:.3f}° R²={r['r2']:.4f} FIT={r['fit']:.1f}%")

        resumo[mod_nome] = {
            'chunk_treino': {
                ds_r['name']: {'rmse': round(float(ds_r['rmse']), 4),
                               'r2':   round(float(ds_r['r2']), 4),
                               'fit':  round(float(ds_r['fit']), 2)}
                for ds_r in chunk_res
            },
            'media': {'RMSE': round(float(rmse_c), 4),
                      'R2':   round(float(r2_c), 4),
                      'FIT':  round(float(fit_c), 2)},
            'best_val': round(float(best_val), 4),
            'n_params': save_model.n_params(),
        }
        print(f"  [{tag} MÉDIA CHUNKS TREINO] RMSE={rmse_c:.3f}° R²={r2_c:.4f} FIT={fit_c:.1f}%")
        plot_chunk_fit(chunk_res, f'Chunk Fit — {tag} (k=400, dados de treino)',
                       out_path=f"{out_dir}/chunkfit_{tag}.png")

        del compiled_model, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(f"{out_dir}/resumo.json", 'w', encoding='utf-8') as fp:
        json.dump(resumo, fp, indent=2, ensure_ascii=False)

    # ────────────────── VISUALIZAÇÕES FINAIS ──────────────────
    print(f"\n{'─'*60}")
    print("  Gerando visualizações finais...")
    print(f"{'─'*60}")

    mod_rmse = [resumo[m]['media']['RMSE'] for m in mod_nomes]
    mod_r2   = [resumo[m]['media']['R2']   for m in mod_nomes]
    mod_fit  = [resumo[m]['media']['FIT']  for m in mod_nomes]
    plot_barchart(mod_nomes,
                  {'RMSE (°)': mod_rmse, 'R²': mod_r2, 'FIT%': mod_fit},
                  'Chunk Fit nos Dados de Treino (k=400)',
                  out_path=f"{out_dir}/barchart_por_modelo_global.png")
    print(f"  Barchart salvo: barchart_por_modelo_global.png")

    # ────────────────── TABELA CONSOLE ──────────────────
    print(f"\n{'='*60}")
    print("  RESUMO — RMSE MÉDIA (°) | Chunk Fit Treino (k=400)")
    print(f"{'='*60}")
    for m in mod_nomes:
        print(f"{m:<15}: {resumo[m]['media']['RMSE']:.3f}°")

    best_model = min(mod_nomes, key=lambda m: resumo[m]['media']['RMSE'])
    best_rmse = resumo[best_model]['media']['RMSE']
    total_time = (time.time() - t_start) / 60
    print(f"\n★ Melhor modelo global: {best_model} → RMSE={best_rmse:.3f}°")
    print(f"  Tempo total: {total_time:.1f} minutos")

    print(f"\nResultados em: {out_dir}/")
    print("[V9] Treinamento Global Aerodinâmico e Assimétrico concluído.")
