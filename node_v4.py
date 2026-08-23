# %% [markdown]
# # Identificação do Aeropêndulo com Neural ODEs (v4.0 - Best Practices)
#
# Ajustes desta versão:
# - Amostragem mista de datasets no mesmo batch
# - Validação contínua (Early Stopping / Checkpoint do melhor modelo)
# - Transição suave (Sigmoide) no modelo Assimétrico
# - Decimação por subsampling em u (preserva degraus)
# - Weight Decay (L2 regularization) na MLP do modelo Híbrido

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

TREINAR_BASELINE = False
TREINAR_ASSIMETRICO = True
TREINAR_HIBRIDO = True

def carregar_experimento(url, decimacao=1, start_idx=None, end_idx=None):
    df = pd.read_csv(url)
    if 'referencia' in df.columns:
        df = df[df['referencia'] > 0]

    if 'u_pct' in df.columns:
        u_full = df['u_pct'].values.astype(np.float64)
    else:
        u_full = df['motor_percent'].values.astype(np.float64)
    y_full = df['angulo_deg'].values.astype(np.float64)

    if decimacao > 1:
        # Decimate aplica filtro passa-baixa (bom para o sensor para evitar aliasing)
        y_raw = decimate(y_full, decimacao, ftype='iir', zero_phase=True)
        # Para o controle (u), usamos subsampling direto para NÃO arredondar os degraus
        u_raw = u_full[::decimacao]
    else:
        u_raw = u_full
        y_raw = y_full

    dt = 0.010 * decimacao
    t_raw = np.arange(len(y_raw)) * dt

    # Pode haver divergência de tamanho de 1 amostra dependendo da paridade
    min_len = min(len(y_raw), len(u_raw))
    t_raw, u_raw, y_raw = t_raw[:min_len], u_raw[:min_len], y_raw[:min_len]

    if start_idx is not None and end_idx is not None:
        t_raw = t_raw[start_idx:end_idx]
        u_raw = u_raw[start_idx:end_idx]
        y_raw = y_raw[start_idx:end_idx]

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

# 1. MODELOS ODE (NODE)
class BaseODE(nn.Module):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0):
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

class PhysicsODE_Baseline(BaseODE):
    def forward(self, t, x):
        J, b, Gu = self.get_params()
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        motor_torque = Gu * u_t * torch.abs(u_t)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot
        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)

class PhysicsODE_Asymmetric(BaseODE):
    def __init__(self, J0=1.0, b_pos0=np.exp(-1.0), b_neg0=np.exp(-1.0), Gu_pos0=1.0, Gu_neg0=1.0):
        super().__init__(J0, b_pos0, Gu_pos0)
        self.log_b_neg = nn.Parameter(torch.log(torch.tensor(float(b_neg0))))
        self.log_Gu_neg = nn.Parameter(torch.log(torch.tensor(float(Gu_neg0))))

    def forward(self, t, x):
        J = torch.exp(self.log_J)
        b_pos = torch.exp(self.log_b)
        b_neg = torch.exp(self.log_b_neg)
        Gu_pos = torch.exp(self.log_Gu)
        Gu_neg = torch.exp(self.log_Gu_neg)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        
        # Transição Suave via Sigmoide
        sigma_b = torch.sigmoid(50.0 * theta_dot)
        b = sigma_b * b_pos + (1.0 - sigma_b) * b_neg
        
        sigma_Gu = torch.sigmoid(50.0 * u_t)
        Gu = sigma_Gu * Gu_pos + (1.0 - sigma_Gu) * Gu_neg
        
        motor_torque = Gu * u_t * torch.abs(u_t)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot
        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)

class PhysicsODE_Hybrid(BaseODE):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0, hidden_dim=16):
        super().__init__(J0, b0, Gu0)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, t, x):
        J, b, Gu = self.get_params()
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        motor_torque = Gu * u_t * torch.abs(u_t)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot
        nn_input = torch.cat([theta, theta_dot, u_t], dim=1)
        residual_torque = self.mlp(nn_input)
        theta_ddot = (motor_torque - gravity_torque - friction_torque + residual_torque) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)


# 2. FUNÇÃO DE TREINAMENTO AVANÇADA (Mix de Datasets, Checkpoint e Weight Decay)
def train_model_v4(model, name, train_datasets, val_datasets, epochs=2000, lr=0.015,
                   k_min=20, k_max=400, curriculum_stage_epochs=400,
                   state_std=None, base_batch_size=1024, integrator='rk4', weight_decay_mlp=1e-4):
    print(f"\n--- Iniciando Treinamento: {name} ---")
    model.to(device)
    
    # Separar os parâmetros da MLP para aplicar o weight decay
    if hasattr(model, 'mlp'):
        mlp_params = [p for n, p in model.named_parameters() if 'mlp' in n]
        phys_params = [p for n, p in model.named_parameters() if 'mlp' not in n]
        optimizer = optim.Adam([
            {'params': phys_params, 'weight_decay': 0.0},
            {'params': mlp_params, 'weight_decay': weight_decay_mlp}
        ], lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if state_std is None:
        state_std = torch.ones(2, device=device)

    dt = (train_datasets[0]['t'][1] - train_datasets[0]['t'][0]).item()
    
    best_val_loss = float('inf')
    best_state_dict = None
    
    # Validação Inicial
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
        
        # Amostragem misturada: extrai pedaços de TODOS os datasets no mesmo passo do otimizador
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
        
        # Checkpoint e Validação a cada 50 épocas
        if epoch % 50 == 0 or epoch == epochs:
            val_loss = avalia_validacao(model, val_datasets, state_std, integrator)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        if epoch % 200 == 0 or epoch == epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d} | k={k_steps:3d} | LR={lr_now:.5f} | Train Loss: {total_loss.item():.5f} | Val Loss: {val_loss:.5f}")

    print(f"[{name}] Concluído. Restaurando melhor modelo (Val Loss: {best_val_loss:.5f}).")
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

def plot_datasets(datasets, title):
    import math
    n = len(datasets)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    fig.suptitle(title, fontsize=16)
    
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]
        
    for i, ax in enumerate(axs):
        if i < n:
            ds = datasets[i]
            y_deg = ds['x'][:, 0].numpy() * (180.0 / np.pi)
            ax.plot(y_deg, color='blue')
            ax.set_title(ds['name'], fontsize=10)
            ax.grid(True)
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.90 if rows > 1 else 0.85)
    plt.show()

if __name__ == '__main__':
    BASE2 = "experimentos/"
    
    train_files = [
        "aprbs-1_0819_18-48.csv",
        "aprbs-2_0819_18-51.csv",
        "aprbs-neg-1_0819_18-54.csv",
        "multi-seno-1_0819_19-23.csv",
        "multi-seno-2_0819_18-59.csv",
        "multi-seno-2_0819_19-11.csv",
        "multi-seno-3_0819_19-19.csv",
        "seq-degraus-aprbs-2_0819_19-16.csv",
        "swept-sine-1_0819_19-02.csv"
    ]
    
    # Estes arquivos não são vistos no treino, servem apenas para o Early Stopping / Checkpoint
    val_files = [
        "RODADA-2/chirp-1_0804_19-19.csv",
        "RODADA-2/multi-seno-2_0804_19-28.csv",
        "RODADA-2/seq-degraus-1_0804_19-09.csv",
    ]
    decimacao = 2

    print("Carregando datasets de TREINO...")
    train_datasets = []
    for f in train_files:
        t_raw, u_raw, y_raw = carregar_experimento(BASE2 + f, decimacao=decimacao, start_idx=250, end_idx=-200)
        t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
        train_datasets.append({'name': f, 't': t_ten, 'u': u_ten, 'x': x_ten})

    plot_datasets(train_datasets, 'Conjuntos de Treino: Ângulo (Graus)')

    print("\nCarregando datasets de VALIDAÇÃO (Checkpoint)...")
    val_datasets = []
    for f in val_files:
        t_raw, u_raw, y_raw = carregar_experimento(BASE2 + f, decimacao=decimacao, start_idx=250, end_idx=-200)
        t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
        val_datasets.append({'name': f, 't': t_ten, 'u': u_ten, 'x': x_ten})

    plot_datasets(val_datasets, 'Conjuntos de Validação: Ângulo (Graus)')

    all_pos = np.concatenate([d['x'][:, 0].numpy() for d in train_datasets])
    all_vel = np.concatenate([d['x'][:, 1].numpy() for d in train_datasets])
    state_std = torch.tensor([float(np.std(all_pos)), float(np.std(all_vel))], dtype=torch.float32, device=device)

    os.makedirs('modelos_salvos', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    modelos = {}

    if TREINAR_BASELINE:
        base_model = PhysicsODE_Baseline()
        base_model = train_model_v4(
            base_model, "Baseline", train_datasets, val_datasets,
            epochs=500, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std
        )
        torch.save(base_model.state_dict(), f'modelos_salvos/node_v4_baseline_{timestamp}.pth')
        modelos["Baseline"] = base_model

    if TREINAR_ASSIMETRICO:
        asymm_model = PhysicsODE_Asymmetric()
        asymm_model = train_model_v4(
            asymm_model, "Assimétrico", train_datasets, val_datasets,
            epochs=2000, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std
        )
        torch.save(asymm_model.state_dict(), f'modelos_salvos/node_v4_asymmetric_{timestamp}.pth')
        modelos["Asymmetric"] = asymm_model
        
    if TREINAR_HIBRIDO:
        hybrid_model = PhysicsODE_Hybrid(hidden_dim=16)
        hybrid_model = train_model_v4(
            hybrid_model, "Híbrido", train_datasets, val_datasets,
            epochs=2000, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std, weight_decay_mlp=1e-4
        )
        torch.save(hybrid_model.state_dict(), f'modelos_salvos/node_v4_hybrid_{timestamp}.pth')
        modelos["Hybrid"] = hybrid_model

    print("\n[V4] Treinamento Finalizado e Modelos Salvos.")
