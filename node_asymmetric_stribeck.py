import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from scipy.signal import savgol_filter, decimate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(0)
np.random.seed(0)

def carregar_experimento(url, decimacao=1):
    df = pd.read_csv(url)
    if 'referencia' in df.columns:
        df = df[df['referencia'] > 0]

    if 'u_pct' in df.columns:
        u_full = df['u_pct'].values.astype(np.float64)
    else:
        u_full = df['motor_percent'].values.astype(np.float64)
    y_full = df['angulo_deg'].values.astype(np.float64)

    if decimacao > 1:
        y_raw = decimate(y_full, decimacao, ftype='iir', zero_phase=True)
        u_raw = u_full[::decimacao]
    else:
        u_raw = u_full
        y_raw = y_full

    dt = 0.010 * decimacao
    t_raw = np.arange(len(y_raw)) * dt

    min_len = min(len(y_raw), len(u_raw))
    t_raw, u_raw, y_raw = t_raw[:min_len], u_raw[:min_len], y_raw[:min_len]

    return t_raw, u_raw, y_raw

def processar_dataset(t_raw, u_raw, y_raw):
    t_raw = t_raw - t_raw[0]
    y_rad = y_raw * (np.pi / 180.0)
    u_norm = np.clip(u_raw / 100.0, -1.0, 1.0)
    
    dt_mean = np.mean(np.diff(t_raw))
    v_rad_s = savgol_filter(y_rad, 11, 3, deriv=1, delta=dt_mean)
    x_matrix = np.vstack((y_rad, v_rad_s)).T
    
    return (torch.tensor(t_raw, dtype=torch.float32),
            torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1),
            torch.tensor(x_matrix, dtype=torch.float32),
            y_rad, v_rad_s, u_norm)

class BaseODE(nn.Module):
    def __init__(self, J0=0.0353, b0=0.05, Gu0=0.9482):
        super().__init__()
        self.m1, self.L1 = 0.122, 0.39
        self.m2, self.L2 = 0.055, 0.347
        self.g = 9.81
        self.log_J = nn.Parameter(torch.log(torch.tensor(float(J0))))
        self.log_b = nn.Parameter(torch.log(torch.tensor(float(b0))))
        self.log_Gu = nn.Parameter(torch.log(torch.tensor(float(Gu0))))
        self.u_series = None
        self.t_series = None
        self.batch_start_times = None

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

class PhysicsODE_Asymmetric_Stribeck(BaseODE):
    def __init__(self, J0=0.03530):
        super().__init__(J0=J0)
        
        # Lado Positivo (Valores extraídos do fit estático V3)
        self.log_Gu_pos = nn.Parameter(torch.log(torch.tensor(0.9482)))
        self.log_Tc_pos = nn.Parameter(torch.log(torch.tensor(0.0356)))
        # Ts é a soma de Tc + exp(log_Ts). Então log_Ts = log(0.1531 - 0.0356) = log(0.1175)
        self.log_Ts_pos = nn.Parameter(torch.log(torch.tensor(0.1175)))
        self.log_vs_pos = nn.Parameter(torch.log(torch.tensor(0.0166)))
        self.log_b_pos  = nn.Parameter(torch.log(torch.tensor(0.0500)))
        
        # Lado Negativo (Copiamos o positivo para o NODE descobrir a real assimetria dinâmica)
        self.log_Gu_neg = nn.Parameter(torch.log(torch.tensor(0.9482)))
        self.log_Tc_neg = nn.Parameter(torch.log(torch.tensor(0.0356)))
        self.log_Ts_neg = nn.Parameter(torch.log(torch.tensor(0.1175)))
        self.log_vs_neg = nn.Parameter(torch.log(torch.tensor(0.0166)))
        self.log_b_neg  = nn.Parameter(torch.log(torch.tensor(0.0500)))

    def forward(self, t, x):
        J = torch.exp(self.log_J)
        
        Gu_pos = torch.exp(self.log_Gu_pos)
        Tc_pos = torch.exp(self.log_Tc_pos)
        Ts_pos = Tc_pos + torch.exp(self.log_Ts_pos)
        vs_pos = torch.exp(self.log_vs_pos)
        b_pos  = torch.exp(self.log_b_pos)
        
        Gu_neg = torch.exp(self.log_Gu_neg)
        Tc_neg = torch.exp(self.log_Tc_neg)
        Ts_neg = Tc_neg + torch.exp(self.log_Ts_neg)
        vs_neg = torch.exp(self.log_vs_neg)
        b_neg  = torch.exp(self.log_b_neg)
        
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        
        # Transição Suave via Sigmoide
        sigma_b = torch.sigmoid(50.0 * theta_dot)
        sigma_u = torch.sigmoid(50.0 * u_t)
        
        Gu = sigma_u * Gu_pos + (1.0 - sigma_u) * Gu_neg
        Tc = sigma_b * Tc_pos + (1.0 - sigma_b) * Tc_neg
        Ts = sigma_b * Ts_pos + (1.0 - sigma_b) * Ts_neg
        vs = sigma_b * vs_pos + (1.0 - sigma_b) * vs_neg
        b  = sigma_b * b_pos  + (1.0 - sigma_b) * b_neg
        
        motor_torque = Gu * u_t * torch.abs(u_t)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        
        # Modelo de Atrito de Stribeck Integrado
        stribeck_effect = Tc + (Ts - Tc) * torch.exp(- (theta_dot / (vs + 1e-6))**2)
        sgn_v = torch.tanh(100.0 * theta_dot)
        friction_torque = stribeck_effect * sgn_v + b * theta_dot
        
        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)

def train_model_v4(model, name, train_datasets, val_datasets, epochs=2000, lr=0.015,
                   k_min=20, k_max=400, curriculum_stage_epochs=400,
                   state_std=None, base_batch_size=1024, integrator='rk4'):
    print(f"\n--- Iniciando Treinamento: {name} ---")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if state_std is None:
        state_std = torch.ones(2, device=device)

    dt = (train_datasets[0]['t'][1] - train_datasets[0]['t'][0]).item()
    
    best_val_loss = float('inf')
    best_state_dict = None
    
    val_loss = avalia_validacao(model, val_datasets, state_std, integrator)

    for epoch in range(1, epochs + 1):
        model.train()
        stage = epoch // curriculum_stage_epochs
        k_steps = int(min(k_min * (2 ** stage), k_max))
        t_eval = torch.arange(0, k_steps * dt, dt, device=device)[:k_steps]
        
        batch_size = max(64, int(base_batch_size * (k_min / k_steps)))
        samples_per_ds = max(1, batch_size // len(train_datasets))
        
        optimizer.zero_grad()
        total_loss = 0
        
        for ds in train_datasets:
            t_ds, u_ds, x_ds = ds['t'].to(device), ds['u'].to(device), ds['x'].to(device)
            model.t_series, model.u_series = t_ds, u_ds
            
            start_idx = np.random.randint(0, len(t_ds) - k_steps, size=samples_per_ds)
            x0 = x_ds[start_idx]
            model.batch_start_times = t_ds[start_idx].reshape(-1, 1)
            
            pred_state = odeint(model, x0, t_eval, method=integrator)
            batch_targets = torch.stack([x_ds[i:i + k_steps] for i in start_idx], dim=1)
            
            loss = torch.mean(((pred_state - batch_targets) / state_std) ** 2)
            total_loss += loss
            
        total_loss = total_loss / len(train_datasets)
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 50 == 0 or epoch == epochs:
            val_loss = avalia_validacao(model, val_datasets, state_std, integrator)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        if epoch % 10 == 0 or epoch == epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d} | k_steps={k_steps:3d} | LR={lr_now:.5f} | Train Loss: {total_loss.item():.6f} | Val Loss: {val_loss:.6f}")

    print(f"[{name}] Concluído. Restaurando melhor modelo (Val Loss: {best_val_loss:.6f}).")
    model.load_state_dict(best_state_dict)
    return model

def avalia_validacao(model, val_datasets, state_std, integrator):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for ds in val_datasets:
            t_t, u_t, x_t = ds['t'].to(device), ds['u'].to(device), ds['x'].to(device)
            model.t_series, model.u_series = t_t, u_t
            model.batch_start_times = torch.zeros(1, 1, device=device)
            
            x0 = x_t[0].unsqueeze(0)
            pred = odeint(model, x0, t_t, method=integrator).squeeze(1)
            val_loss += torch.mean(((pred - x_t) / state_std) ** 2).item()
    return val_loss / len(val_datasets)


if __name__ == '__main__':
    BASE2 = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"
    
    train_files = [
        "RODADA-2/multi-seno-1_0804_19-06.csv",
        "RODADA-2/seq-degraus-2_0804_19-38.csv",
        "RODADA-2/multi-seno-3_0804_19-44.csv",
        "RODADA-2/seq-degraus-1_0804_19-12.csv",
        "multi-seno-1_0807_16-57.csv",
        "multi-seno-2_0807_17-13.csv",
        "seq-degraus-1_0807_16-38.csv",
        "seq-degraus-aprbs-2_0807_17-23.csv",
        "seq-dragus-aprbs_0807_16-50.csv"
    ]
    
    val_files = [
        "RODADA-2/chirp-1_0804_19-19.csv",
        "RODADA-2/multi-seno-2_0804_19-28.csv",
        "RODADA-2/seq-degraus-1_0804_19-09.csv",
    ]
    decimacao = 10

    print("Carregando datasets de TREINO...")
    train_datasets = []
    for f in train_files:
        t_raw, u_raw, y_raw = carregar_experimento(BASE2 + f, decimacao=decimacao)
        t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
        train_datasets.append({'name': f, 't': t_ten, 'u': u_ten, 'x': x_ten})

    print("\nCarregando datasets de VALIDAÇÃO (Checkpoint)...")
    val_datasets = []
    for f in val_files:
        t_raw, u_raw, y_raw = carregar_experimento(BASE2 + f, decimacao=decimacao)
        t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
        val_datasets.append({'name': f, 't': t_ten, 'u': u_ten, 'x': x_ten})

    all_pos = np.concatenate([d['x'][:, 0].numpy() for d in train_datasets])
    all_vel = np.concatenate([d['x'][:, 1].numpy() for d in train_datasets])
    state_std = torch.tensor([float(np.std(all_pos)), float(np.std(all_vel))], dtype=torch.float32, device=device)

    os.makedirs('modelos_salvos', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    asymm_stribeck_model = PhysicsODE_Asymmetric_Stribeck()
    
    asymm_stribeck_model = train_model_v4(
        asymm_stribeck_model, "Asymmetric_Stribeck", train_datasets, val_datasets,
        epochs=500, lr=0.01, k_min=20, k_max=200, curriculum_stage_epochs=100,
        state_std=state_std
    )
    
    nome_salvar = f'modelos_salvos/node_asymmetric_stribeck_{timestamp}.pth'
    torch.save(asymm_stribeck_model.state_dict(), nome_salvar)
    print(f"\n[V Final] Treinamento Finalizado. Modelo salvo em {nome_salvar}")
