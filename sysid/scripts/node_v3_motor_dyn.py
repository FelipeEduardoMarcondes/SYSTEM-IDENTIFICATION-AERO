# %% [markdown]
# # Identificação do Aeropêndulo com Neural ODEs - Multi-Experimentos (v4.0 - Dinâmica de 3ª Ordem)
#
# Ajustes desta versão em relação à v3.0:
# - **Motor de 3ª Ordem (Caminho B)**: O estado passa a ter 3 dimensões: `[theta, theta_dot, a]`.
# - A variável `a` é a ativação do motor, que segue uma dinâmica de primeira ordem com constante de tempo `tau`:
#   `da/dt = (u - a) / tau`
# - O torque gerado passa a ser proporcional a `a * abs(a)` ao invés de responder instantaneamente a `u`.
# - Isso ajuda a modelar a inércia da hélice e o atraso na resposta do sistema.

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
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import copy
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(0)
np.random.seed(0)

# %%
# Configuração: Escolha quais modelos treinar
TREINAR_BASELINE = True
TREINAR_ASSIMETRICO = True
TREINAR_ASSIMETRICO_AERO = True
TREINAR_ASSIMETRICO_AERO_COULOMB = True
TREINAR_HIBRIDO = False

# %%
# Adiciona o diretório raiz ao path para conseguir importar aerodata
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from aerodata import readData

def carregar_experimento(dataset_name, decimacao=1):
    # Usa a função oficial readData do aerodata
    y_raw, u_raw, t_raw = readData(dataset_name, decimar=decimacao)
    return t_raw, u_raw, y_raw

def processar_dataset(t_raw, u_raw, y_raw):
    t_raw = t_raw - t_raw[0]
    y_rad = y_raw * (np.pi / 180.0)

    # Clip em [-1.0, 1.0] para o comando
    u_norm = np.clip(u_raw / 100.0, -1.0, 1.0)

    dt_mean = np.mean(np.diff(t_raw))
    v_rad_s = savgol_filter(y_rad, 11, 3, deriv=1, delta=dt_mean)
    
    # NOVO v4: O estado agora tem dimensão 3 [posição, velocidade, ativação_motor].
    # Para o instante inicial de cada batch, assumimos que o motor já estava em regime (a = u).
    # O treinamento não calculará loss para o 3º estado, pois não temos a medição dele.
    x_matrix = np.vstack((y_rad, v_rad_s, u_norm)).T

    return (torch.tensor(t_raw, dtype=torch.float32),
            torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1),
            torch.tensor(x_matrix, dtype=torch.float32),
            y_rad, v_rad_s, u_norm)

# %%
# 1. MODELOS ODE (NODE) - AGORA COM DINÂMICA DE 3ª ORDEM
class BaseODE(nn.Module):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0, tau0=0.1):
        super().__init__()
        self.m1, self.L1 = 0.122, 0.39
        self.m2, self.L2 = 0.055, 0.347
        self.g = 9.81
        self.log_J = nn.Parameter(torch.log(torch.tensor(float(J0))))
        self.log_b = nn.Parameter(torch.log(torch.tensor(float(b0))))
        self.log_Gu = nn.Parameter(torch.log(torch.tensor(float(Gu0))))
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau0)))) # NOVO: Constante de tempo do motor
        self.u_series = None
        self.t_series = None
        self.batch_start_times = None

    def get_params(self):
        return torch.exp(self.log_J), torch.exp(self.log_b), torch.exp(self.log_Gu), torch.exp(self.log_tau)

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
        J, b, Gu, tau = self.get_params()
        u_t = self._get_u_t(t, x)
        
        # Estado 3D
        theta, theta_dot, a = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        # Dinâmica do Motor (1ª Ordem no sinal de controle)
        a_dot = (u_t - a) / tau

        motor_torque = Gu * a * torch.abs(a)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot

        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        
        return torch.cat([theta_dot, theta_ddot, a_dot], dim=1)

class PhysicsODE_Asymmetric(BaseODE):
    def __init__(self, J0=1.0, b_pos0=np.exp(-1.0), b_neg0=np.exp(-1.0), Gu_pos0=1.0, Gu_neg0=1.0, tau0=0.1):
        super().__init__(J0, b_pos0, Gu_pos0, tau0)
        self.log_b_neg = nn.Parameter(torch.log(torch.tensor(float(b_neg0))))
        self.log_Gu_neg = nn.Parameter(torch.log(torch.tensor(float(Gu_neg0))))

    def forward(self, t, x):
        J = torch.exp(self.log_J)
        b_pos = torch.exp(self.log_b)
        b_neg = torch.exp(self.log_b_neg)
        Gu_pos = torch.exp(self.log_Gu)
        Gu_neg = torch.exp(self.log_Gu_neg)
        tau = torch.exp(self.log_tau)

        u_t = self._get_u_t(t, x)
        theta, theta_dot, a = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        a_dot = (u_t - a) / tau

        b = torch.where(theta_dot > 0, b_pos, b_neg)
        Gu = torch.where(a > 0, Gu_pos, Gu_neg)

        motor_torque = Gu * a * torch.abs(a)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot

        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot, a_dot], dim=1)

class PhysicsODE_AsymmetricAero(BaseODE):
    def __init__(self, J0=1.0, b_pos0=np.exp(-1.0), b_neg0=np.exp(-1.0),
                 Gu_pos0=1.0, Gu_neg0=1.0, c_pos0=0.01, c_neg0=0.01, tau0=0.1):
        super().__init__(J0, b_pos0, Gu_pos0, tau0)
        self.log_b_neg = nn.Parameter(torch.log(torch.tensor(float(b_neg0))))
        self.log_Gu_neg = nn.Parameter(torch.log(torch.tensor(float(Gu_neg0))))
        self.log_c_pos = nn.Parameter(torch.log(torch.tensor(float(c_pos0))))
        self.log_c_neg = nn.Parameter(torch.log(torch.tensor(float(c_neg0))))

    def forward(self, t, x):
        J = torch.exp(self.log_J)
        b_pos = torch.exp(self.log_b)
        b_neg = torch.exp(self.log_b_neg)
        Gu_pos = torch.exp(self.log_Gu)
        Gu_neg = torch.exp(self.log_Gu_neg)
        c_pos = torch.exp(self.log_c_pos)
        c_neg = torch.exp(self.log_c_neg)
        tau = torch.exp(self.log_tau)

        u_t = self._get_u_t(t, x)
        theta, theta_dot, a = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        a_dot = (u_t - a) / tau

        b = torch.where(theta_dot > 0, b_pos, b_neg)
        Gu = torch.where(a > 0, Gu_pos, Gu_neg)
        c = torch.where(theta_dot > 0, c_pos, c_neg)

        motor_torque = Gu * a * torch.abs(a)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot + c * theta_dot * torch.abs(theta_dot)

        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot, a_dot], dim=1)

class PhysicsODE_AsymmetricAeroCoulomb(BaseODE):
    def __init__(self, J0=1.0, b_pos0=np.exp(-1.0), b_neg0=np.exp(-1.0),
                 Gu_pos0=1.0, Gu_neg0=1.0, c_pos0=0.01, c_neg0=0.01,
                 Tc_pos0=0.01, Tc_neg0=0.01, tau0=0.1):
        super().__init__(J0, b_pos0, Gu_pos0, tau0)
        self.log_b_neg = nn.Parameter(torch.log(torch.tensor(float(b_neg0))))
        self.log_Gu_neg = nn.Parameter(torch.log(torch.tensor(float(Gu_neg0))))
        self.log_c_pos = nn.Parameter(torch.log(torch.tensor(float(c_pos0))))
        self.log_c_neg = nn.Parameter(torch.log(torch.tensor(float(c_neg0))))
        self.log_Tc_pos = nn.Parameter(torch.log(torch.tensor(float(Tc_pos0))))
        self.log_Tc_neg = nn.Parameter(torch.log(torch.tensor(float(Tc_neg0))))

    def forward(self, t, x):
        J = torch.exp(self.log_J)
        b_pos = torch.exp(self.log_b)
        b_neg = torch.exp(self.log_b_neg)
        Gu_pos = torch.exp(self.log_Gu)
        Gu_neg = torch.exp(self.log_Gu_neg)
        c_pos = torch.exp(self.log_c_pos)
        c_neg = torch.exp(self.log_c_neg)
        Tc_pos = torch.exp(self.log_Tc_pos)
        Tc_neg = torch.exp(self.log_Tc_neg)
        tau = torch.exp(self.log_tau)

        u_t = self._get_u_t(t, x)
        theta, theta_dot, a = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        a_dot = (u_t - a) / tau

        b = torch.where(theta_dot > 0, b_pos, b_neg)
        Gu = torch.where(a > 0, Gu_pos, Gu_neg)
        c = torch.where(theta_dot > 0, c_pos, c_neg)
        Tc = torch.where(theta_dot > 0, Tc_pos, Tc_neg)

        motor_torque = Gu * a * torch.abs(a)
        gravity_torque = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot + c * theta_dot * torch.abs(theta_dot) + Tc * torch.tanh(100.0 * theta_dot)

        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot, a_dot], dim=1)


# %%
# 2. FUNÇÃO DE TREINAMENTO
def train_model_multi(model, name, datasets, epochs=1500, lr=0.015,
                       k_min=20, k_max=300, curriculum_stage_epochs=300,
                       state_std=None, base_batch_size=1024, integrator='rk4'):
    print(f"\n--- Iniciando Treinamento: {name} ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if state_std is None:
        state_std = torch.ones(2, device=device) # Normaliza apenas os 2 primeiros estados

    dt = (datasets[0]['t'][1] - datasets[0]['t'][0]).item()
    loss_history = []
    
    best_loss = float('inf')
    best_model_state = None

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
        
        # x_ds tem dimensão 3 (posição, velocidade, ativação_inicial)
        x0 = x_ds[start_idx]
        model.batch_start_times = t_ds[start_idx].reshape(-1, 1)

        pred_state = odeint(model, x0, t_eval, method=integrator)
        batch_targets = [x_ds[i:i + k_steps] for i in start_idx]
        y_target = torch.stack(batch_targets, dim=1)

        # Calculamos o erro APENAS nas variáveis observadas (theta e theta_dot)
        pred_obs = pred_state[:, :, :2]
        target_obs = y_target[:, :, :2]

        loss = torch.mean(((pred_obs - target_obs) / state_std) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if k_steps == k_max and loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = copy.deepcopy(model.state_dict())

        if epoch % 300 == 0 or epoch == epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d} | k_steps={k_steps:3d} | batch={batch_size:4d} "
                  f"| LR={lr_now:.5f} | Loss: {loss.item():.6f} | Tau estimado: {torch.exp(model.log_tau).item():.4f}s")

    if best_model_state is not None:
        print(f"Restaurando melhor modelo com Loss: {best_loss:.6f}")
        model.load_state_dict(best_model_state)

    return model, loss_history

if __name__ == '__main__':
    train_files = [
        "RODADA-7/MIX_DC_45.csv",
        "RODADA-7/MIX_DC_60.csv"
    ]

    test_files = [
        "RODADA-2/chirp-1_0804_19-19.csv",
        "RODADA-2/multi-seno-2_0804_19-28.csv",
        "RODADA-2/seq-degraus-1_0804_19-09.csv",
    ]
    decimacao = 1

    print("Carregando datasets de TREINO...")
    train_datasets = []
    for f in train_files:
        t_raw, u_raw, y_raw = carregar_experimento(f, decimacao=decimacao)
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
        t_raw, u_raw, y_raw = carregar_experimento(f, decimacao=decimacao)
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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"resultados_motor3a_ordem_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Resultados em: {out_dir}/\n")

    if TREINAR_BASELINE:
        base_model = PhysicsODE_Baseline()
        base_model, hist = train_model_multi(
            base_model, "Baseline", train_datasets,
            epochs=4000, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std, base_batch_size=4096
        )
        torch.save(base_model.state_dict(), f'{out_dir}/model_baseline.pth')
        modelos["Baseline"] = base_model

    if TREINAR_ASSIMETRICO:
        asymm_model = PhysicsODE_Asymmetric()
        asymm_model, hist = train_model_multi(
            asymm_model, "Assimétrico", train_datasets,
            epochs=4000, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std, base_batch_size=4096
        )
        torch.save(asymm_model.state_dict(), f'{out_dir}/model_asymmetric.pth')
        modelos["Asymmetric"] = asymm_model

    if TREINAR_ASSIMETRICO_AERO:
        aero_model = PhysicsODE_AsymmetricAero()
        aero_model, hist = train_model_multi(
            aero_model, "Assimétrico+Aero", train_datasets,
            epochs=4000, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std, base_batch_size=4096
        )
        torch.save(aero_model.state_dict(), f'{out_dir}/model_asymm_aero.pth')
        modelos["AsymmAero"] = aero_model

    if TREINAR_ASSIMETRICO_AERO_COULOMB:
        aero_coulomb_model = PhysicsODE_AsymmetricAeroCoulomb()
        aero_coulomb_model, hist = train_model_multi(
            aero_coulomb_model, "Assimétrico+Aero+Coulomb", train_datasets,
            epochs=4000, lr=0.015, k_min=20, k_max=400, curriculum_stage_epochs=400,
            state_std=state_std, base_batch_size=4096
        )
        torch.save(aero_coulomb_model.state_dict(), f'{out_dir}/model_asymm_aero_coulomb.pth')
        modelos["AsymmAeroCoulomb"] = aero_coulomb_model

    print("\n--- Simulação Free-Run (Testes) ---")
    resultados_metricas = {nome: {} for nome in modelos.keys()}
    
    import json
    
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
                
                # Métricas
                rmse = np.sqrt(mean_squared_error(y_real_deg, pred_deg))
                
                numerador = np.linalg.norm(y_real_deg - pred_deg)
                denominador = np.linalg.norm(y_real_deg - np.mean(y_real_deg))
                fit_pct = 100.0 * (1.0 - numerador / denominador) if denominador != 0 else 0.0
                
                resultados_metricas[nome][ds['name']] = {'rmse': float(rmse), 'fit': float(fit_pct)}
                msg += f" | {nome}: {fit_pct:5.1f}% (RMSE {rmse:5.2f}°)"
                
                # Plotting (Apenas Posição para ficar igual ao anterior)
                plt.figure(figsize=(10, 4))
                plt.plot(t_t.cpu().numpy(), y_real_deg, 'k-', lw=1.2, label='Real')
                plt.plot(t_t.cpu().numpy(), pred_deg, 'r--', lw=1.2, label=f'Pred ({nome})')
                plt.title(f"Teste: {ds['name']} | Modelo: {nome} | FIT: {fit_pct:.1f}% | RMSE: {rmse:.2f}°")
                plt.xlabel("Tempo (s)")
                plt.ylabel("Ângulo (°)")
                plt.legend()
                plt.grid(True, alpha=0.4)
                
                safe_name = ds['name'].replace('/', '_').replace('\\', '_')
                plt.savefig(f"{out_dir}/freerun_{nome}_{safe_name}.png", dpi=100, bbox_inches='tight')
                plt.close()

        print(msg)
        
    with open(f"{out_dir}/resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados_metricas, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em {out_dir}/")
