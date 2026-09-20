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

# ==========================================
# 1. MODELOS ODE (NODE)
# ==========================================
class PhysicsODE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- VALORES FÍSICOS CONHECIDOS DA BANCADA ---
        self.m1 = 0.3  # Massa do lado da hélice (kg)
        self.L1 = 0.39  # Distância do pivô até a hélice (m)
        self.m2 = 0.2  # Massa do contrapeso (kg)
        self.L2 = 0.347  # Distância do pivô até o contrapeso (m)
        self.g = 9.81  # Gravidade (m/s^2)
        
        # Parâmetros Desconhecidos que a Rede Neural vai descobrir:
        self.log_J = nn.Parameter(torch.tensor(0.0))
        self.log_b = nn.Parameter(torch.tensor(-1.0))
        self.log_Gu = nn.Parameter(torch.tensor(0.0))
        
        self.u_series = None
        self.t_series = None
        self.batch_start_times = None

    def get_params(self):
        return torch.exp(self.log_J), torch.exp(self.log_b), torch.exp(self.log_Gu)

    def forward(self, t, x):
        J, b, Gu = self.get_params()
        
        # Equacionamento dinâmico
        p1 = b / J
        p2 = ((self.m1 * self.L1 - self.m2 * self.L2) * self.g) / J
        p3 = Gu / J

        # Alinhamento temporal para mini-batches
        if self.batch_start_times is not None:
            t_abs = self.batch_start_times + t
        else:
            t_abs = t * torch.ones_like(x[:, 0:1])

        # Interpolação do sinal de controle u(t)
        k = torch.searchsorted(self.t_series, t_abs.reshape(-1), right=True)
        k = torch.clamp(k, 1, len(self.t_series) - 1)
        
        t1, t2 = self.t_series[k-1].unsqueeze(1), self.t_series[k].unsqueeze(1)
        u1, u2 = self.u_series[k-1], self.u_series[k]

        denom = (t2 - t1)
        denom[denom < 1e-6] = 1.0
        alpha = (t_abs - t1) / denom
        u_t = u1 + alpha * (u2 - u1)

        # Estados: posição (theta) e velocidade (theta_dot)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        
        # theta_ddot = - (b/J)*theta_dot - ((m1L1-m2L2)g/J)*sin(theta) + u_gain * (u^2)
        theta_ddot = - p1 * theta_dot - p2 * torch.sin(theta) + p3 * (u_t ** 2)
        
        return torch.cat([theta_dot, theta_ddot], dim=1)


class BlackBoxODE(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), # [theta, theta_dot, u]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # [theta_dot, theta_ddot]
        )
        self.u_series = None
        self.t_series = None
        self.batch_start_times = None

    def forward(self, t, x):
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
        u_t = u1 + alpha * (u2 - u1)

        nn_input = torch.cat([x, u_t], dim=1)
        return self.net(nn_input)


# --- FUNÇÕES ---
def carregar_experimento(url, decimacao=1):
    df = pd.read_csv(url)
    
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

if __name__ == '__main__':
    BASE2 = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"
    test_files = [
        "multi-seno-1_0807_16-57.csv",
        "seq-degraus-1_0807_16-38.csv"
    ]
    decimacao = 2
    
    print("\nCarregando datasets de TESTE...")
    test_datasets = []
    for f in test_files:
        t_raw, u_raw, y_raw = carregar_experimento(BASE2 + f, decimacao=decimacao)
        
        # Opcional: Corte das bordas
        corte_inicio = 200
        corte_fim = -200
        t_raw = t_raw[corte_inicio:corte_fim]
        u_raw = u_raw[corte_inicio:corte_fim]
        y_raw = y_raw[corte_inicio:corte_fim]
        
        t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
        test_datasets.append({
            'name': f,
            't': t_ten, 'u': u_ten, 'x': x_ten,
            'y_rad': y_rad, 'v_rad': v_rad, 'u_norm': u_norm
        })
        print(f"Dataset '{f}': {len(t_ten)} amostras.")

    print("\nCarregando modelos ODE...")
    
    # Caixa Cinza
    model_cinza = PhysicsODE().to(device)
    try:
        # Tenta carregar com weights_only=True (PyTorch mais novo) ou sem
        try:
            model_cinza.load_state_dict(torch.load("node_v4_asymmetric_20260811_190734.pth", map_location=device, weights_only=True))
        except TypeError:
            model_cinza.load_state_dict(torch.load("node_caixa_cinza.pth", map_location=device))
        model_cinza.eval()
        print("Modelo Caixa Cinza carregado com sucesso.")
    except Exception as e:
        print(f"Aviso: Não foi possível carregar node_caixa_cinza.pth: {e}")
        
    # Caixa Preta
    model_preta = BlackBoxODE().to(device)
    try:
        try:
            model_preta.load_state_dict(torch.load("node_caixa_preta.pth", map_location=device, weights_only=True))
        except TypeError:
            model_preta.load_state_dict(torch.load("node_caixa_preta.pth", map_location=device))
        model_preta.eval()
        print("Modelo Caixa Preta carregado com sucesso.")
    except Exception as e:
        print(f"Aviso: Não foi possível carregar node_caixa_preta.pth: {e}")

    for ds in test_datasets:
        print(f"\n--- Simulação Free-Run para o dataset: {ds['name']} ---")
        t_ten = ds['t'].to(device)
        u_ten = ds['u'].to(device)
        x0 = ds['x'][0:1].to(device)
        
        results = {}
        
        # Simulação Cinza
        model_cinza.u_series = u_ten
        model_cinza.t_series = t_ten
        with torch.no_grad():
            pred_cinza = odeint(model_cinza, x0, t_ten, method='rk4', options={'step_size': 0.05})
        pred_cinza_np = pred_cinza.squeeze(1).cpu().numpy()
        results['Cinza'] = pred_cinza_np[:, 0]
        
        # Simulação Preta
        model_preta.u_series = u_ten
        model_preta.t_series = t_ten
        with torch.no_grad():
            pred_preta = odeint(model_preta, x0, t_ten, method='rk4', options={'step_size': 0.05})
        pred_preta_np = pred_preta.squeeze(1).cpu().numpy()
        results['Preta'] = pred_preta_np[:, 0]
        
        # Original
        y_true = ds['y_rad']
        
        # Métricas
        rmse_cinza = np.sqrt(mean_squared_error(y_true, results['Cinza']))
        rmse_preta = np.sqrt(mean_squared_error(y_true, results['Preta']))
        print(f"RMSE (Caixa Cinza) : {rmse_cinza:.4f} rad")
        print(f"RMSE (Caixa Preta) : {rmse_preta:.4f} rad")
        
        # Plota os resultados
        plt.figure(figsize=(12, 6))
        plt.plot(ds['t'].numpy(), y_true * 180 / np.pi, 'k-', label='Real', alpha=0.5)
        plt.plot(ds['t'].numpy(), results['Cinza'] * 180 / np.pi, 'b--', label=f'Caixa Cinza (RMSE={rmse_cinza:.3f})')
        plt.plot(ds['t'].numpy(), results['Preta'] * 180 / np.pi, 'r:', label=f'Caixa Preta (RMSE={rmse_preta:.3f})')
        plt.title(f"Simulação Free-Run: {ds['name']}")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Ângulo (graus)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
