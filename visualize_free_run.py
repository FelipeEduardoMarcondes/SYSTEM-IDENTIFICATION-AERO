import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint
from scipy.signal import savgol_filter, decimate
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- CONFIGURAÇÃO DOS MODELOS PARA VISUALIZAÇÃO ---
# Coloque aqui o nome exato dos arquivos que você quer carregar
ARQUIVO_BASELINE = 'node_v3_baseline_20260809_180823.pth'
ARQUIVO_ASSIMETRICO = 'node_v3_asymmetric_20260809_180823.pth'
ARQUIVO_HIBRIDO = 'node_v3_hybrid_20260809_180823.pth'

# Reusing definitions from train_comparative.py
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
    u_norm = np.clip(u_raw / 100.0, -1.0, 1.0)
    dt_mean = np.mean(np.diff(t_raw))
    v_rad_s = savgol_filter(y_rad, 11, 3, deriv=1, delta=dt_mean)
    x_matrix = np.vstack((y_rad, v_rad_s)).T
    return (torch.tensor(t_raw, dtype=torch.float32),
            torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1),
            torch.tensor(x_matrix, dtype=torch.float32),
            y_rad, v_rad_s, u_norm)

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

if __name__ == '__main__':
    BASE2 = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"
    test_files = [
        "RODADA-2/multi-seno-2_0804_19-31.csv",
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
    decimacao = 2

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
        
    baseline = PhysicsODE_Baseline()
    baseline.load_state_dict(torch.load(ARQUIVO_BASELINE, map_location=device, weights_only=True))
    baseline.to(device)
    baseline.eval()

    asymm = PhysicsODE_Asymmetric()
    asymm.load_state_dict(torch.load(ARQUIVO_ASSIMETRICO, map_location=device, weights_only=True))
    asymm.to(device)
    asymm.eval()

    hybrid = PhysicsODE_Hybrid(hidden_dim=16)
    hybrid.load_state_dict(torch.load(ARQUIVO_HIBRIDO, map_location=device, weights_only=True))
    hybrid.to(device)
    hybrid.eval()

    print("\n--- Simulação Free-Run ---")

    n_test = len(test_datasets)
    fig, axes = plt.subplots(n_test, 1, figsize=(14, 5 * n_test), squeeze=False)

    for i, ds in enumerate(test_datasets):
        t_t = ds['t'].to(device)
        u_t = ds['u'].to(device)
        x_t = ds['x'].to(device)
        
        y_real_deg = x_t[:, 0].cpu().numpy() * (180.0 / np.pi)
        t_np = t_t.cpu().numpy()

        with torch.no_grad():
            x0 = x_t[0].unsqueeze(0)
            
            # Baseline
            baseline.u_series = u_t
            baseline.t_series = t_t
            baseline.batch_start_times = torch.zeros(1, 1, device=device)
            pred_base = odeint(baseline, x0, t_t, method='dopri5', rtol=1e-5, atol=1e-6).squeeze(1).cpu().numpy()
            base_deg = pred_base[:, 0] * (180.0 / np.pi)
            rmse_base = np.sqrt(mean_squared_error(y_real_deg, base_deg))
                
            # Asymmetric
            asymm.u_series = u_t
            asymm.t_series = t_t
            asymm.batch_start_times = torch.zeros(1, 1, device=device)
            pred_asymm = odeint(asymm, x0, t_t, method='dopri5', rtol=1e-5, atol=1e-6).squeeze(1).cpu().numpy()
            asymm_deg = pred_asymm[:, 0] * (180.0 / np.pi)
            rmse_asymm = np.sqrt(mean_squared_error(y_real_deg, asymm_deg))

            # Hybrid
            hybrid.u_series = u_t
            hybrid.t_series = t_t
            hybrid.batch_start_times = torch.zeros(1, 1, device=device)
            pred_hybrid = odeint(hybrid, x0, t_t, method='dopri5', rtol=1e-5, atol=1e-6).squeeze(1).cpu().numpy()
            hybrid_deg = pred_hybrid[:, 0] * (180.0 / np.pi)
            rmse_hybrid = np.sqrt(mean_squared_error(y_real_deg, hybrid_deg))

        msg = f"{ds['name']:40s} | Base: {rmse_base:6.2f}° | Asymm: {rmse_asymm:6.2f}° | Hybrid: {rmse_hybrid:6.2f}°"
        print(msg)

        ax = axes[i, 0]
        ax.plot(t_np, y_real_deg, 'k-', linewidth=2, label='Real')
        ax.plot(t_np, base_deg, 'r--', linewidth=1.5, alpha=0.7, label=f'Baseline ({rmse_base:.2f}°)')
        ax.plot(t_np, asymm_deg, 'g-', linewidth=2, alpha=0.9, label=f'Asymmetric ({rmse_asymm:.2f}°)')
        ax.plot(t_np, hybrid_deg, 'b--', linewidth=2, alpha=0.9, label=f'Hybrid ({rmse_hybrid:.2f}°)')
            
        ax.set_title(ds['name'])
        ax.set_xlabel("Tempo (s)"); ax.set_ylabel("Ângulo (graus)")
        ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()
