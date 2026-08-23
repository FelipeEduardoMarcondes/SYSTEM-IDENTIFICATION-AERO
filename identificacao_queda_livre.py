#!/usr/bin/env python3
import sys
"""
identificacao_queda_livre.py
============================
Identifica atrito do aeropendulo a partir de oscilacao livre (motor OFF).

Equacao de movimento (sem motor):
  J*theta_dd + b*theta_d + Tc*tanh(c*theta_d) + tau_grav*sin(theta) = 0

Dados: degraus 90->0 com motor desligado apos a queda.
Entrada: somente theta(t).
Saida: J, b, Tc.

Etapa 1: Decremento logaritmico -> estimativa inicial
Etapa 2: Multiple shooting diferenciavel (torchdiffeq) -> refinamento
Etapa 3: Validacao + curva de atrito
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==============================
# Constantes fisicas (medidas)
# ==============================
M1, L1 = 0.122, 0.39
M2, L2 = 0.055, 0.347
GRAV   = 9.81
DT     = 0.010   # 100 Hz
TAU_GRAV = (M1*L1 - M2*L2) * GRAV  # ~0.2795 N.m

# Base do Github (igual ao node_v4.py)
BASE_URL = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"

# Arquivos: degraus 90->0 (motor desligado apos queda)
PATHS = [
    BASE_URL + "degraus_0819_19-29.csv",
    BASE_URL + "degraus_0819_19-30.csv",
    BASE_URL + "degraus_0819_19-33.csv",
    BASE_URL + "queda-livre-1.csv"
]


# ===========================================================================
# Carregar dados: recortar somente a fase de queda livre (ref = 0, motor OFF)
# ===========================================================================
def carregar_queda_livre(path):
    df = pd.read_csv(path)
    # Ajuste para URLs (usando split com '/') ou caminho local ('\\')
    nome = path.replace('\\', '/').split('/')[-1].replace('.csv', '')

    refs = df['referencia'].values
    theta_deg = df['angulo_deg'].values.astype(np.float64)
    t = np.arange(len(df)) * DT

    # Encontrar instante em que referencia cai para 0
    idx_queda = np.where(np.diff(refs) < -40)[0]
    if len(idx_queda) == 0:
        print(f"  AVISO: nao encontrou queda em {nome}")
        return None
    i0 = idx_queda[0] + 1

    # Recortar a partir da queda
    t_free = t[i0:] - t[i0]
    theta_free = np.radians(theta_deg[i0:])

    # Velocidade via Savitzky-Golay (para condicoes iniciais do MS)
    omega_free = savgol_filter(theta_free, 51, 3, deriv=1, delta=DT)

    print(f"  {nome}: queda em t={t[i0]:.1f}s, {len(t_free)} amostras livres, "
          f"theta=[{np.degrees(theta_free.min()):.0f}, {np.degrees(theta_free.max()):.0f}] deg")

    return {
        'nome': nome, 't': t_free, 'theta': theta_free,
        'omega': omega_free, 'theta_deg': theta_deg[i0:],
    }


# ===========================================================================
# ETAPA 1: Decremento Logaritmico -> estimativa de J, zeta, b
# ===========================================================================
def etapa1(datasets):
    print("\n" + "="*65)
    print("  ETAPA 1: Decremento Logaritmico")
    print("="*65)

    # Usar o dataset com oscilacoes mais limpas (19-29)
    ds = datasets[0]
    theta_deg = ds['theta_deg']

    # Picos positivos
    peaks, _ = find_peaks(theta_deg, prominence=1.0, distance=80)
    valleys, _ = find_peaks(-theta_deg, prominence=1.0, distance=80)

    if len(peaks) < 3:
        print("  Poucos picos - usando estimativa manual")
        return 0.045, 0.015, 0.005

    peak_amps = np.abs(theta_deg[peaks])
    peak_times = ds['t'][peaks]

    # Decremento logaritmico
    deltas = []
    for i in range(len(peak_amps) - 1):
        if peak_amps[i+1] > 0.3:
            d = np.log(peak_amps[i] / peak_amps[i+1])
            if 0 < d < 3:
                deltas.append(d)

    delta_med = np.mean(deltas) if deltas else 0.3
    zeta = delta_med / np.sqrt(4*np.pi**2 + delta_med**2)

    # Periodo amortecido
    Td = np.median(np.diff(peak_times))
    omega_d = 2*np.pi / Td
    omega_n = omega_d / np.sqrt(max(1 - zeta**2, 0.01))

    J_est = TAU_GRAV / (omega_n**2)
    b_est = 2 * zeta * omega_n * J_est
    Tc_est = 0.005  # chute inicial pequeno

    print(f"\n  Picos: {len(peaks)},  Deltas validos: {len(deltas)}")
    print(f"  delta medio  = {delta_med:.4f}")
    print(f"  Td           = {Td:.3f} s")
    print(f"  omega_d      = {omega_d:.3f} rad/s")
    print(f"  omega_n      = {omega_n:.3f} rad/s")
    print(f"  zeta         = {zeta:.4f}")
    print(f"\n  Estimativas iniciais:")
    print(f"    J  = {J_est:.5f} kg.m2")
    print(f"    b  = {b_est:.5f} N.m.s/rad")
    print(f"    Tc = {Tc_est:.5f} N.m")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ds['t'], theta_deg, 'b-', lw=1, label='Angulo (motor OFF)')
    ax.plot(ds['t'][peaks], theta_deg[peaks], 'rv', ms=8, label='Picos')
    if len(valleys) > 0:
        ax.plot(ds['t'][valleys], theta_deg[valleys], 'g^', ms=8, label='Vales')
    # Envelope
    A0 = peak_amps[0]
    t_env = np.linspace(0, ds['t'][peaks[-1]], 300)
    env = A0 * np.exp(-zeta * omega_n * t_env)
    ax.plot(t_env, env, 'r--', lw=2, label=f'Envelope: zeta={zeta:.4f}')
    ax.plot(t_env, -env, 'r--', lw=2)
    ax.set(xlabel='Tempo (s)', ylabel='Angulo (deg)',
           title=f'Decremento Logaritmico: J={J_est:.4f}  b={b_est:.5f}  Td={Td:.2f}s')
    ax.legend(); ax.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig('queda_livre_etapa1.png', dpi=150); plt.close()
    print("  Grafico -> queda_livre_etapa1.png")

    return J_est, b_est, Tc_est


# ===========================================================================
# ETAPA 2: Multiple Shooting (torchdiffeq) -> [J, b, Tc]
# ===========================================================================
class QuedaLivreODE(nn.Module):
    """
    J*theta_dd = -tau_grav*sin(theta) - b*theta_d - Tc*tanh(50*theta_d)
    Sem motor. Sem entrada externa.
    """
    def __init__(self, J0, b0, Tc0):
        super().__init__()
        self.log_J  = nn.Parameter(torch.tensor(np.log(max(J0, 1e-4)), dtype=torch.float32))
        self.log_b  = nn.Parameter(torch.tensor(np.log(max(b0, 1e-6)), dtype=torch.float32))
        self.log_Tc = nn.Parameter(torch.tensor(np.log(max(Tc0, 1e-6)), dtype=torch.float32))

    def forward(self, t, state):
        theta, omega = state[0], state[1]
        J  = torch.exp(self.log_J)
        b  = torch.exp(self.log_b)
        Tc = torch.exp(self.log_Tc)

        tau_grav = TAU_GRAV * torch.sin(theta)
        tau_fric = b * omega + Tc * torch.tanh(50.0 * omega)
        alpha = (-tau_grav - tau_fric) / J

        return torch.stack([omega, alpha])

    def params(self):
        return {
            'J':  torch.exp(self.log_J).item(),
            'b':  torch.exp(self.log_b).item(),
            'Tc': torch.exp(self.log_Tc).item(),
        }


def etapa2(datasets, J0, b0, Tc0):
    print("\n" + "="*65)
    print("  ETAPA 2: Multiple Shooting (queda livre)")
    print("="*65)

    WINDOW = 200   # 2.0 s
    windows = []

    for ds in datasets:
        n = len(ds['t'])
        for i in range(0, n - WINDOW, WINDOW):
            j = i + WINDOW
            t_w  = torch.tensor(ds['t'][i:j] - ds['t'][i], dtype=torch.float32).to(device)
            th_w = torch.tensor(ds['theta'][i:j], dtype=torch.float32).to(device)
            x0   = torch.tensor([ds['theta'][i], ds['omega'][i]], dtype=torch.float32).to(device)
            windows.append((t_w, th_w, x0))

    print(f"  Janelas: {len(windows)} (W={WINDOW} = {WINDOW*DT:.1f}s)")

    model = QuedaLivreODE(J0, b0, Tc0).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=150,
                                                      factor=0.5, min_lr=1e-5)

    EPOCHS = 1500
    BATCH  = min(24, len(windows))
    best_loss = float('inf')
    best_state = None

    print(f"  Treinando ({EPOCHS} epocas, batch={BATCH})..."); sys.stdout.flush()

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()
        indices = np.random.choice(len(windows), BATCH, replace=False)
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        count = 0

        for idx in indices:
            t_w, th_w, x0 = windows[idx]
            try:
                x_pred = odeint(model, x0, t_w, method='rk4',
                                options={'step_size': DT})
                loss = torch.mean((x_pred[:, 0] - th_w)**2)
                total_loss = total_loss + loss
                count += 1
            except Exception:
                continue

        if count == 0:
            continue

        mean_loss = total_loss / count
        mean_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(mean_loss.detach())

        if mean_loss.item() < best_loss:
            best_loss = mean_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 250 == 0:
            p = model.params()
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:5d} | Loss: {mean_loss.item():.6f} | "
                  f"J={p['J']:.5f} b={p['b']:.5f} Tc={p['Tc']:.5f} | lr={lr:.1e}")
            sys.stdout.flush()

    model.load_state_dict(best_state)
    p = model.params()

    wn = np.sqrt(TAU_GRAV / p['J'])
    zeta = p['b'] / (2 * wn * p['J'])

    print(f"\n  === Parametros Finais (loss={best_loss:.6f}) ===")
    print(f"    J  = {p['J']:.5f} kg.m2")
    print(f"    b  = {p['b']:.5f} N.m.s/rad")
    print(f"    Tc = {p['Tc']:.5f} N.m")
    print(f"    wn = {wn:.3f} rad/s   zeta = {zeta:.4f}   T = {2*np.pi/wn:.2f} s")

    return p['J'], p['b'], p['Tc']


# ===========================================================================
# ETAPA 3: Validacao + Curva de Atrito
# ===========================================================================
def simular_rk4(t, theta0, omega0, J, b, Tc):
    """Simula queda livre com RK4 fixo."""
    n = len(t)
    theta_sim = np.zeros(n)
    theta_sim[0] = theta0
    state = np.array([theta0, omega0])

    for k in range(n - 1):
        def f(s):
            th, om = s
            tg = TAU_GRAV * np.sin(th)
            tf = b * om + Tc * np.tanh(50.0 * om)
            return np.array([om, (-tg - tf) / J])
        k1 = f(state)
        k2 = f(state + DT/2*k1)
        k3 = f(state + DT/2*k2)
        k4 = f(state + DT*k3)
        state = state + DT/6*(k1 + 2*k2 + 2*k3 + k4)
        theta_sim[k+1] = state[0]

    return theta_sim


def etapa3(datasets, J, b, Tc):
    print("\n" + "="*65)
    print("  ETAPA 3: Validacao")
    print("="*65)

    fig, axes = plt.subplots(len(datasets), 1, figsize=(15, 5*len(datasets)), squeeze=False)

    for i, ds in enumerate(datasets):
        # Simulacao completa (single shooting desde t=0)
        theta_sim = simular_rk4(ds['t'], ds['theta'][0], ds['omega'][0], J, b, Tc)
        rmse_full = np.sqrt(np.mean((np.degrees(ds['theta']) - np.degrees(theta_sim))**2))

        # Simulacao segmentada (reinicia a cada 2s)
        seg = 200
        theta_seg = np.full(len(ds['t']), np.nan)
        for s in range(0, len(ds['t']), seg):
            e = min(s + seg, len(ds['t']))
            ts = simular_rk4(ds['t'][s:e], ds['theta'][s], ds['omega'][s], J, b, Tc)
            theta_seg[s:e] = ts
        rmse_seg = np.sqrt(np.nanmean((np.degrees(ds['theta']) - np.degrees(theta_seg))**2))

        ax = axes[i, 0]
        ax.plot(ds['t'], ds['theta_deg'], 'b-', lw=1.5, alpha=.8, label='Medido (motor OFF)')
        ax.plot(ds['t'], np.degrees(theta_sim), 'r-', lw=1.2, alpha=.7,
                label=f'Single-shot (RMSE={rmse_full:.1f} deg)')
        ax.plot(ds['t'], np.degrees(theta_seg), 'g--', lw=1.2, alpha=.7,
                label=f'Segmentado 2s (RMSE={rmse_seg:.1f} deg)')
        ax.set(xlabel='Tempo (s)', ylabel='Angulo (deg)', title=ds['nome'])
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)
        print(f"  {ds['nome']}: RMSE single={rmse_full:.2f} deg, seg={rmse_seg:.2f} deg")

    plt.suptitle(f'Validacao Queda Livre: J={J:.4f}  b={b:.5f}  Tc={Tc:.5f}', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('queda_livre_validacao.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Grafico -> queda_livre_validacao.png")

    # ----- Curva de Atrito -----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    SG = 101; m = SG//2
    for ds in datasets:
        omega = savgol_filter(ds['theta'], SG, 3, deriv=1, delta=DT)
        alpha = savgol_filter(ds['theta'], SG, 3, deriv=2, delta=DT)
        s = slice(m, len(ds['t'])-m)
        # tau_fric = -tau_grav*sin(theta) - J*alpha  (da EOM sem motor)
        tau_f = -TAU_GRAV*np.sin(ds['theta'][s]) - J*alpha[s]
        ax2.scatter(omega[s], tau_f, s=2, alpha=.3, label=ds['nome'])

    w = np.linspace(-4, 4, 500)
    ax2.plot(w, b*w + Tc*np.tanh(50*w), 'k-', lw=3,
             label=f'Modelo: b={b:.5f} + Tc={Tc:.5f}*sgn(w)')
    ax2.axhline(0, c='gray', lw=.5); ax2.axvline(0, c='gray', lw=.5)
    ax2.set(xlabel='Velocidade Angular (rad/s)', ylabel='Torque de Atrito (N.m)',
            title='Curva de Atrito (Queda Livre, Motor OFF)')
    ax2.legend(fontsize=9, markerscale=5); ax2.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig('queda_livre_curva_atrito.png', dpi=150)
    plt.close()
    print("  Grafico -> queda_livre_curva_atrito.png")


# ===========================================================================
if __name__ == '__main__':
    print("\n  Carregando dados de queda livre (motor OFF)...")
    datasets = [d for d in [carregar_queda_livre(p) for p in PATHS] if d is not None]
    if not datasets:
        print("ERRO: nenhum dataset valido"); sys.exit(1)

    J0, b0, Tc0 = etapa1(datasets)
    J, b, Tc = etapa2(datasets, J0, b0, Tc0)
    etapa3(datasets, J, b, Tc)

    print("\n" + "="*65)
    print("  RESULTADO FINAL")
    print("="*65)
    print(f"  J  = {J:.5f} kg.m2    (momento de inercia)")
    print(f"  b  = {b:.5f} N.m.s/rad (atrito viscoso)")
    print(f"  Tc = {Tc:.5f} N.m      (atrito de Coulomb)")
    wn = np.sqrt(TAU_GRAV / J)
    zeta = b / (2*wn*J)
    print(f"  wn = {wn:.3f} rad/s   zeta = {zeta:.4f}   T = {2*np.pi/wn:.2f} s")
    print("="*65)
