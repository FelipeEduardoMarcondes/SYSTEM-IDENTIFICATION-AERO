# %% [markdown]
# # Identificação do Aeropêndulo com Neural ODEs - Multi-Experimentos (v3.0 - Comparativo Avançado)
#
# Ajustes desta versão em relação à v2.0:
# - **Motor Bidirecional**: Comando `u` normalizado entre [-1, 1] e torque proporcional a `u * abs(u)`.
# - **Novos Modelos de Atrito/Dinâmica**: 
#     1. Baseline (Atrito Linear simples)
#     2. Assimétrico (Atrito e Ganho de motor diferentes para subida/descida)
#     3. Híbrido/UDE (Física básica + Rede Neural modelando arrasto induzido/atrito complexo)
# - **Herança Orientada a Objetos**: Classes herdam de `BaseODE` para facilitar manutenção.

# %%
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
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(0)
np.random.seed(0)

# %%
# Configuração: Escolha quais modelos treinar
TREINAR_BASELINE = True
TREINAR_ASSIMETRICO = True
TREINAR_HIBRIDO = True

# %%
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
        u_raw = decimate(u_full, decimacao, ftype='iir', zero_phase=True)
        y_raw = decimate(y_full, decimacao, ftype='iir', zero_phase=True)
    else:
        u_raw = u_full
        y_raw = y_full

    dt = 0.010 * decimacao
    t_raw = np.arange(len(y_raw)) * dt

    return t_raw, u_raw, y_raw

def processar_dataset(t_raw, u_raw, y_raw):
    t_raw = t_raw - t_raw[0]
    y_rad = y_raw * (np.pi / 180.0)
    
    # CORREÇÃO v3: u pode ser negativo, então clipamos em [-1.0, 1.0]
    u_norm = np.clip(u_raw / 100.0, -1.0, 1.0)
    
    dt_mean = np.mean(np.diff(t_raw))
    v_rad_s = savgol_filter(y_rad, 11, 3, deriv=1, delta=dt_mean)
    x_matrix = np.vstack((y_rad, v_rad_s)).T
    
    return (torch.tensor(t_raw, dtype=torch.float32),
            torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1),
            torch.tensor(x_matrix, dtype=torch.float32),
            y_rad, v_rad_s, u_norm)

# %%
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
        
        b = torch.where(theta_dot > 0, b_pos, b_neg)
        Gu = torch.where(u_t > 0, Gu_pos, Gu_neg)
        
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


# %%
# 2. FUNÇÃO DE TREINAMENTO (MULTI-DATASET, COM CURRICULUM + LOSS NORMALIZADA + LR SCHEDULE)
def train_model_multi(model, name, datasets, epochs=1500, lr=0.015,
                       k_min=20, k_max=300, curriculum_stage_epochs=300,
                       state_std=None, base_batch_size=1024, integrator='rk4'):
    print(f"\n--- Iniciando Treinamento: {name} ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if state_std is None:
        state_std = torch.ones(2, device=device)

    dt = (datasets[0]['t'][1] - datasets[0]['t'][0]).item()
    loss_history = []

    for epoch in range(epochs + 1):
        stage = epoch // curriculum_stage_epochs
        k_steps = int(min(k_min * (2 ** stage), k_max))
        t_eval = torch.arange(0, k_steps * dt, dt, device=device)[:k_steps]

        batch_size = max(64, int(base_batch_size * (k_min / k_steps)))
        optimizer.zero_grad()

        ds = np.random.choice(datasets)
        t_ds, u_ds, x_ds = ds['t'].to(device), ds['u'].to(device), ds['x'].to(device)

        model.t_series = t_ds
        model.u_series = u_ds

        start_idx = np.random.randint(0, len(t_ds) - k_steps, size=batch_size)
        x0 = x_ds[start_idx]
        model.batch_start_times = t_ds[start_idx].reshape(-1, 1)

        pred_state = odeint(model, x0, t_eval, method=integrator)
        batch_targets = [x_ds[i:i + k_steps] for i in start_idx]
        y_target = torch.stack(batch_targets, dim=1)

        loss = torch.mean(((pred_state - y_target) / state_std) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if epoch % 300 == 0 or epoch == epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d} | k_steps={k_steps:3d} | batch={batch_size:4d} "
                  f"| LR={lr_now:.5f} | Loss: {loss.item():.6f}")

    return model, loss_history

if __name__ == '__main__':
    BASE2 = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"
    
    train_files = [
        "RODADA-2/multi-seno-2_0804_19-31.csv",
        "RODADA-2/seq-degraus-2_0804_19-38.csv",
        "RODADA-2/seq-degraus-1_0804_19-12.csv",
        "multi-seno-1_0807_16-57.csv",
        "seq-degraus-1_0807_16-38.csv"
    ]
    
    test_files = [
        "RODADA-2/chirp-1_0804_19-19.csv",
        "RODADA-2/multi-seno-2_0804_19-28.csv",
        "RODADA-2/seq-degraus-1_0804_19-09.csv",
    ]
    decimacao = 2

    print("Carregando datasets de TREINO...")
    train_datasets = []
    for f in train_files:
        t_raw, u_raw, y_raw = carregar_experimento(BASE2 + f, decimacao=decimacao)
        t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
        train_datasets.append({
            'name': f,
            't': t_ten, 'u': u_ten, 'x': x_ten,
            'y_rad': y_rad, 'v_rad': v_rad, 'u_norm': u_norm
        })
        print(f" -> {f} carregado com {len(t_ten)} amostras.")

    print("\nCarregando datasets de TESTE...")
    test_datasets = []
    for f in test_files:
        t_raw, u_raw, y_raw = carregar_experimento(BASE2 + f, decimacao=decimacao)
        t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
        test_datasets.append({
            'name': f,
            't': t_ten, 'u': u_ten, 'x': x_ten,
            'y_rad': y_rad, 'v_rad': v_rad, 'u_norm': u_norm
        })
        print(f" -> {f} carregado com {len(t_ten)} amostras.")

    all_pos = np.concatenate([d['y_rad'] for d in train_datasets])
    all_vel = np.concatenate([d['v_rad'] for d in train_datasets])
    state_std = torch.tensor([float(np.std(all_pos)), float(np.std(all_vel))], dtype=torch.float32, device=device)

    modelos = {}
    
    os.makedirs('modelos_salvos', exist_ok=True)

    if TREINAR_BASELINE:
        base_model = PhysicsODE_Baseline()
        base_model, hist = train_model_multi(
            base_model, "Baseline", train_datasets,
            epochs=2000, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(base_model.state_dict(), f'modelos_salvos/node_v3_baseline_{timestamp}.pth')
        modelos["Baseline"] = base_model

    if TREINAR_ASSIMETRICO:
        asymm_model = PhysicsODE_Asymmetric()
        asymm_model, hist = train_model_multi(
            asymm_model, "Assimétrico", train_datasets,
            epochs=2000, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std
        )
        torch.save(asymm_model.state_dict(), f'modelos_salvos/node_v3_asymmetric_{timestamp}.pth')
        modelos["Asymmetric"] = asymm_model
        
    if TREINAR_HIBRIDO:
        hybrid_model = PhysicsODE_Hybrid(hidden_dim=16)
        hybrid_model, hist = train_model_multi(
            hybrid_model, "Híbrido", train_datasets,
            epochs=2000, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std
        )
        torch.save(hybrid_model.state_dict(), f'modelos_salvos/node_v3_hybrid_{timestamp}.pth')
        modelos["Hybrid"] = hybrid_model

    print("\n--- Simulação Free-Run (Testes) ---")
    resultados_rmse = {nome: [] for nome in modelos.keys()}

    for i, ds in enumerate(test_datasets):
        t_t = ds['t'].to(device)
        u_t = ds['u'].to(device)
        x_t = ds['x'].to(device)
        y_real_deg = x_t[:, 0].cpu().numpy() * (180.0 / np.pi)
        
        msg = f"{ds['name']:40s}"
        
        with torch.no_grad():
            x0 = x_t[0].unsqueeze(0)
            for nome, modelo in modelos.items():
                modelo.eval()
                modelo.u_series = u_t
                modelo.t_series = t_t
                modelo.batch_start_times = torch.zeros(1, 1, device=device)
                
                pred = odeint(modelo, x0, t_t, method='dopri5', rtol=1e-5, atol=1e-6).squeeze(1).cpu().numpy()
                pred_deg = pred[:, 0] * (180.0 / np.pi)
                rmse = np.sqrt(mean_squared_error(y_real_deg, pred_deg))
                resultados_rmse[nome].append(rmse)
                msg += f" | {nome}: {rmse:6.2f}°"
                
        print(msg)
