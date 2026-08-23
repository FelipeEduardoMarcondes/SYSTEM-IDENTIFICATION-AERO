from sympy import false
import os
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
# 1. CONFIGURAÇÃO DOS MODELOS PARA COMPARAR
# ==========================================
# Dicionário de modelos que você deseja carregar e comparar.
# Formato: "Nome no Gráfico": {"tipo": "TipoDoModelo", "caminho": "caminho/do/arquivo.pth"}
# Tipos disponíveis: "Baseline", "Asymmetric", "Hybrid", "BlackBox"

MODELOS = {}
if os.path.exists("modelos_salvos"):
    for file in os.listdir("modelos_salvos"):
        if file.endswith(".pth"):
            nome = file.replace(".pth", "")
            if "tustin" in file.lower():
                tipo = "Tustin"
            elif "asymmetric" in file.lower():
                tipo = "Asymmetric"
            elif "hybrid" in file.lower() or "caixa_cinza" in file.lower():
                tipo = "Hybrid"
            elif "coulomb" in file.lower():
                tipo = "Coulomb"
            elif "baseline" in file.lower():
                tipo = "Baseline"
            elif "caixa_preta" in file.lower():
                tipo = "BlackBox"
            else:
                continue
            MODELOS[nome] = {"tipo": tipo, "caminho": os.path.join("modelos_salvos", file), "comparar": True}

# ==========================================
# 2. DEFINIÇÕES DOS MODELOS
# ==========================================

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

class PhysicsODE_Coulomb(BaseODE):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0, Tc0=0.01):
        super().__init__(J0, b0, Gu0)
        self.log_Tc = nn.Parameter(torch.log(torch.tensor(float(Tc0))))

    def forward(self, t, x):
        J, b, Gu = self.get_params()
        Tc = torch.exp(self.log_Tc)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        
        motor_torque = Gu * u_t * torch.abs(u_t)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        
        friction_torque = b * theta_dot + Tc * torch.tanh(50.0 * theta_dot)
        
        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)

class PhysicsODE_Tustin(BaseODE):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0, Tc0=0.01, Ts0=0.02, vs0=0.1):
        super().__init__(J0, b0, Gu0)
        self.log_Tc = nn.Parameter(torch.log(torch.tensor(float(Tc0))))
        self.log_Ts = nn.Parameter(torch.log(torch.tensor(float(Ts0))))
        self.log_vs = nn.Parameter(torch.log(torch.tensor(float(vs0))))

    def forward(self, t, x):
        J, b, Gu = self.get_params()
        Tc = torch.exp(self.log_Tc)
        Ts = torch.exp(self.log_Ts)
        vs = torch.exp(self.log_vs)
        
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        
        motor_torque = Gu * u_t * torch.abs(u_t)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        
        stribeck_effect = Tc + (Ts - Tc) * torch.exp(-torch.abs(theta_dot / vs))
        coulomb_stribeck = stribeck_effect * torch.tanh(50.0 * theta_dot)
        
        friction_torque = b * theta_dot + coulomb_stribeck
        
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
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0, hidden_dim=32):
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

    def forward(self, t, x):
        u_t = self._get_u_t(t, x)
        nn_input = torch.cat([x, u_t], dim=1)
        return self.net(nn_input)

# ==========================================
# 3. FUNÇÕES AUXILIARES E EXECUÇÃO
# ==========================================

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

if __name__ == '__main__':
    BASE_URL = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"
    test_files = [
        "RODADA-3/multi-seno-1_0807_16-57.csv",
        "RODADA-3/seq-degraus-1_0807_16-38.csv"
    ]
    decimacao = 2 # Decimação maior para simulação mais rápida
    
    print("\n[1] Carregando datasets de TESTE...")
    test_datasets = []
    for f in test_files:
        t_raw, u_raw, y_raw = carregar_experimento(BASE_URL + f, decimacao=decimacao)
        t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
        test_datasets.append({
            'name': f,
            't': t_ten, 'u': u_ten, 'x': x_ten,
            'y_rad': y_rad, 'v_rad': v_rad, 'u_norm': u_norm
        })
        print(f"Dataset '{f}' carregado com {len(t_ten)} amostras.")

    print("\n[2] Carregando modelos ODE...")
    instancias_modelos = {}
    
    for nome, config in MODELOS.items():
        if not config.get("comparar", True):
            continue
        
        tipo = config["tipo"]
        caminho = config["caminho"]
        
        try:
            if tipo == "Baseline":
                modelo = PhysicsODE_Baseline()
            elif tipo == "Coulomb":
                modelo = PhysicsODE_Coulomb()
            elif tipo == "Tustin":
                modelo = PhysicsODE_Tustin()
            elif tipo == "Asymmetric":
                modelo = PhysicsODE_Asymmetric()
            elif tipo == "Hybrid":
                modelo = PhysicsODE_Hybrid(hidden_dim=32)
            elif tipo == "BlackBox":
                modelo = BlackBoxODE(hidden_dim=32)
            else:
                print(f"Aviso: Tipo de modelo desconhecido '{tipo}' para '{nome}'. Ignorando.")
                continue
                
            try:
                try:
                    modelo.load_state_dict(torch.load(caminho, map_location=device, weights_only=True))
                except TypeError:
                    modelo.load_state_dict(torch.load(caminho, map_location=device))
            except RuntimeError:
                if tipo == "Hybrid":
                    modelo = PhysicsODE_Hybrid(hidden_dim=16)
                elif tipo == "BlackBox":
                    modelo = BlackBoxODE(hidden_dim=16)
                try:
                    modelo.load_state_dict(torch.load(caminho, map_location=device, weights_only=True))
                except TypeError:
                    modelo.load_state_dict(torch.load(caminho, map_location=device))
                
            modelo.to(device)
            modelo.eval()
            instancias_modelos[nome] = modelo
            print(f"Sucesso: '{nome}' carregado do arquivo {caminho}")
        except Exception as e:
            print(f"Falha ao carregar '{nome}' do arquivo {caminho}: {e}")

    if not instancias_modelos:
        print("\nNenhum modelo foi carregado com sucesso. Verifique os caminhos dos arquivos em MODELOS.")
        exit(1)

    print("\n[3] Iniciando Simulação Free-Run (isso pode levar alguns segundos)...")
    
    n_test = len(test_datasets)
    fig, axes = plt.subplots(n_test, 1, figsize=(14, 6 * n_test), squeeze=False)
    colors = ['r--', 'g--', 'b--', 'm--', 'c--', 'y--']

    for i, ds in enumerate(test_datasets):
        t_t = ds['t'].to(device)
        u_t = ds['u'].to(device)
        x_t = ds['x'].to(device)
        
        y_real_deg = x_t[:, 0].cpu().numpy() * (180.0 / np.pi)
        t_np = t_t.cpu().numpy()

        ax = axes[i, 0]
        ax.plot(t_np, y_real_deg, 'k-', linewidth=2, label='Real')
        
        print(f"\nSimulando para dataset: {ds['name']}")
        
        with torch.no_grad():
            x0 = x_t[0].unsqueeze(0)
            
            resultados = []
            
            for j, (nome, modelo) in enumerate(instancias_modelos.items()):
                modelo.u_series = u_t
                modelo.t_series = t_t
                modelo.batch_start_times = torch.zeros(1, 1, device=device)
                
                # Utiliza dopri5 para maior rapidez, ou rk4 se preferir. 
                # dopri5 tem passo adaptativo e é muito mais eficiente em Python.
                pred = odeint(modelo, x0, t_t, method='dopri5', rtol=1e-5, atol=1e-6).squeeze(1).cpu().numpy()
                pred_deg = pred[:, 0] * (180.0 / np.pi)
                rmse = np.sqrt(mean_squared_error(y_real_deg, pred_deg))
                
                print(f"  {nome:15s} | RMSE: {rmse:6.2f}°")
                resultados.append((nome, rmse, pred_deg))
                
            # Ordena os resultados pelo RMSE e pega os 3 melhores
            resultados.sort(key=lambda x: x[1])
            melhores = resultados[:3]
            
            print(f"  -> Top 3 para este dataset: {[r[0] for r in melhores]}")
            
            for j, (nome, rmse, pred_deg) in enumerate(melhores):
                color = colors[j % len(colors)]
                ax.plot(t_np, pred_deg, color, linewidth=2.0, alpha=0.8, label=f'{nome} (RMSE={rmse:.2f}°)')
                
        ax.set_title(f"Simulação Free-Run: {ds['name']} (Top 3 Modelos)")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Ângulo (graus)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    print("\nVisualização concluída!")
