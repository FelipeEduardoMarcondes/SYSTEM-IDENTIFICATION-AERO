import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from scipy.signal import savgol_filter, decimate
import torch.nn as nn

# ---------------------------------------------------------------------------
# Copied/Adapted from node_v6.py to allow interactive matplotlib plotting
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"
DECIMACAO  = 2
START_IDX  = 200
END_IDX    = -2000

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
        J      = torch.exp(self.log_J)
        b_pos  = torch.exp(self.log_b)
        b_neg  = torch.exp(self.log_b_neg)
        Gu_pos = torch.exp(self.log_Gu)
        Gu_neg = torch.exp(self.log_Gu_neg)
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

def avalia_free_run(model, datasets, integrator='rk4'):
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

TEST_FILES_BY_TYPE = {
    "APRBS":     "RODADA-4/aprbs-2_0819_18-51.csv",
    "MultiSeno": "RODADA-2/multi-seno-1_0804_19-06.csv",
    "Degraus":   "RODADA-2/seq-degraus-2_0804_19-38.csv",
    "Chirp":     "RODADA-2/chirp-1_0804_19-19.csv",
}

def plot_free_run_interactive(resultado, exp_name):
    n = len(resultado)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    fig.suptitle(f'Free-Run NODE v6 — {exp_name}', fontsize=13)
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
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Free Run of a NODE Model')
    parser.add_argument('--results_dir', type=str, default=r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\resultados_v6_20260830_125325')
    parser.add_argument('--exp_name', type=str, default='F_Mix', help='Experiment Name to load e.g. A_APRBS, B_MultiSeno, F_Mix')
    args = parser.parse_args()
    
    model_path = os.path.join(args.results_dir, f'model_{args.exp_name}.pth')
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        print("Available models in directory:")
        if os.path.exists(args.results_dir):
            for f in os.listdir(args.results_dir):
                if f.endswith('.pth'):
                    print("  - " + f.replace('model_', '').replace('.pth', ''))
        exit(1)
        
    print(f"Loading model {model_path} ...")
    model = PhysicsODE_Asymmetric().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print("Loading test datasets (might take a moment to download) ...")
    test_por_tipo = {
        tipo: carregar_lista([arquivo])
        for tipo, arquivo in TEST_FILES_BY_TYPE.items()
    }
    
    todos_resultados = []
    print("Running full free-run evaluation ...")
    for tipo, dsl in test_por_tipo.items():
        res = avalia_free_run(model, dsl)
        todos_resultados.extend(res)
        
    print(f"Opening interactive plot for {args.exp_name} ...")
    plot_free_run_interactive(todos_resultados, args.exp_name)
