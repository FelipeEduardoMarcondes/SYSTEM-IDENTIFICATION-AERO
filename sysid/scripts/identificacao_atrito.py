#!/usr/bin/env python3
import sys
"""
identificacao_atrito.py - Identificacao de Atrito do Aeropendulo
================================================================

Abordagem:
  Etapa 1: Equilibrio estatico (escada 0->90) -> Ktau
  Etapa 2: Decremento logaritmico das oscilacoes apos degrau -> J, b
  Etapa 3: Multiple shooting DIFERENCIAVEL (PyTorch+torchdiffeq) -> [J, b, Tc]
  Etapa 4: Validacao + curva de atrito
  
Nota: os dados de degrau sao em MALHA FECHADA (PID ativo). O sinal u(t)
e a SAIDA do PID (dado medido), usado como entrada conhecida na simulacao.
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
# Constantes Fisicas (medidas)
# ==============================
M1, L1 = 0.122, 0.39
M2, L2 = 0.055, 0.347
GRAV   = 9.81
DT     = 0.010
TAU_GRAV = (M1*L1 - M2*L2) * GRAV  # ~0.2795 N.m

PATH_ESCADA = r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\escada-atrito-1_0819_19-08.csv"
PATHS_DEGRAUS = [
    r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\degraus_0819_19-29.csv",
    r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\degraus_0819_19-30.csv",
    r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\degraus_0819_19-33.csv",
]


# ===========================================================================
# ETAPA 1: Equilibrio Estatico -> Ktau
# ===========================================================================
def etapa1():
    print("\n" + "="*65)
    print("  ETAPA 1: Equilibrio Estatico -> Ktau")
    print("="*65)

    df = pd.read_csv(PATH_ESCADA)
    refs = sorted([r for r in df['referencia'].unique() if r > 0])
    dados = []
    for ref in refs:
        sub = df[df['referencia'] == ref]
        tail = sub.iloc[len(sub)//2:]
        dados.append({
            'ref': ref,
            'theta_deg': tail['angulo_deg'].mean(),
            'u_pct': tail['u_pct'].mean(),
            'u_std': tail['u_pct'].std(),
        })
    dados = pd.DataFrame(dados)

    sin_th = np.sin(np.radians(dados['theta_deg'].values))
    u_norm = dados['u_pct'].values / 100.0

    # --- Linear: tau = Ktau * u ---
    A = np.vstack([sin_th, np.ones_like(sin_th)]).T
    (slope, u0), *_ = np.linalg.lstsq(A, u_norm, rcond=None)
    Ktau = TAU_GRAV / slope
    u_pred = A @ [slope, u0]
    r2 = 1 - np.sum((u_norm - u_pred)**2) / np.sum((u_norm - u_norm.mean())**2)

    # --- Quadratico: tau = Gu * u * |u| ---
    (sq, c0), *_ = np.linalg.lstsq(A, u_norm**2, rcond=None)
    Gu = TAU_GRAV / sq
    u_pred_q = np.sqrt(np.clip(A @ [sq, c0], 0, None))
    r2q = 1 - np.sum((u_norm - u_pred_q)**2) / np.sum((u_norm - u_norm.mean())**2)

    print(f"\n  Linear  tau = Ktau*u   -> Ktau = {Ktau:.5f} N.m  R2 = {r2:.6f}")
    print(f"  Quadrat tau = Gu*u|u|  -> Gu   = {Gu:.5f} N.m  R2 = {r2q:.6f}")

    # Selecionar melhor
    if r2 >= r2q:
        tipo, K = 'linear', Ktau
        print(f"\n  -> Selecionado: LINEAR  Ktau = {Ktau:.5f}  u0 = {u0*100:.2f}%")
    else:
        tipo, K = 'quadratico', Gu
        print(f"\n  -> Selecionado: QUADRATICO  Gu = {Gu:.5f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rr = np.linspace(0, 90, 200)
    sr = np.sin(np.radians(rr))
    ax1.errorbar(dados['ref'], dados['u_pct'], yerr=dados['u_std'],
                 fmt='ko', capsize=4, ms=5, label='Patamares medidos')
    ax1.plot(rr, (slope*sr + u0)*100, 'r-', lw=2,
             label=f'Linear: Ktau={Ktau:.4f} (R2={r2:.4f})')
    ax1.plot(rr, np.sqrt(np.clip(sq*sr + c0, 0, None))*100, 'b--', lw=2,
             label=f'Quadrat: Gu={Gu:.4f} (R2={r2q:.4f})')
    ax1.set(xlabel='Referencia (deg)', ylabel='u (%)', title='Equilibrio Estatico')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=.3)
    res = (u_norm - u_pred) * 100
    ax2.stem(dados['ref'], res, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax2.axhline(0, c='k', lw=.5)
    ax2.set(xlabel='Referencia (deg)', ylabel='Residuo (%)', title='Residuos (linear)')
    ax2.grid(True, alpha=.3)
    plt.suptitle('Etapa 1: Constante de Empuxo', fontsize=13)
    plt.tight_layout(); plt.savefig('etapa1_motor.png', dpi=150); plt.close()
    print("  Grafico -> etapa1_motor.png")
    return K, tipo


# ===========================================================================
# ETAPA 2: Decremento Logaritmico -> estimativa de J e b
# ===========================================================================
def etapa2():
    """
    Apos o degrau 90->0, o pendulo oscila livremente em torno de theta=0.
    A taxa de decaimento da amplitude da a razao de amortecimento zeta.
    Linearizando em torno de theta~0:
      omega_d ~= omega_n*sqrt(1-zeta^2),  omega_n = sqrt(TAU_GRAV/J)
    Decremento logaritmico: delta = ln(A_k / A_{k+1}) = 2*pi*zeta/sqrt(1-zeta^2)
    """
    print("\n" + "="*65)
    print("  ETAPA 2: Decremento Logaritmico -> J, zeta, b")
    print("="*65)

    # Usar degrau 90->0 (degraus_19-29) - oscilacoes mais limpas
    df = pd.read_csv(PATHS_DEGRAUS[0])
    theta_deg = df['angulo_deg'].values.astype(np.float64)
    t = np.arange(len(theta_deg)) * DT

    # Localizar a queda (ref muda de 90 para 0)
    refs = df['referencia'].values
    idx_queda = np.where(np.diff(refs) < -40)[0]
    if len(idx_queda) > 0:
        t_start = t[idx_queda[0]]
    else:
        t_start = 10.0  # fallback

    # Recortar a partir da queda
    mask = t >= t_start
    t_osc = t[mask] - t[mask][0]
    theta_osc = theta_deg[mask]

    # Encontrar picos positivos (oscilacoes)
    peaks, props = find_peaks(theta_osc, prominence=2.0, distance=100)
    valleys, _ = find_peaks(-theta_osc, prominence=2.0, distance=100)

    if len(peaks) < 3:
        print("  AVISO: poucos picos encontrados, usando estimativa manual")
        return 0.045, 0.015, 0.005

    peak_amps = np.abs(theta_osc[peaks])
    peak_times = t_osc[peaks]

    # Decremento logaritmico (media de pares consecutivos)
    deltas = []
    for i in range(len(peak_amps) - 1):
        if peak_amps[i+1] > 0.5:  # descartar picos muito pequenos (ruido)
            d = np.log(peak_amps[i] / peak_amps[i+1])
            if d > 0:
                deltas.append(d)

    if len(deltas) == 0:
        print("  AVISO: nao foi possivel calcular decremento")
        return 0.045, 0.015, 0.005

    delta_med = np.mean(deltas)
    zeta = delta_med / np.sqrt(4*np.pi**2 + delta_med**2)

    # Periodo amortecido
    periodos = np.diff(peak_times)
    Td = np.median(periodos)
    omega_d = 2*np.pi / Td
    omega_n = omega_d / np.sqrt(1 - zeta**2) if zeta < 1 else omega_d

    J_est = TAU_GRAV / (omega_n**2)
    b_est = 2 * zeta * omega_n * J_est

    # Estimativa de Tc pelo offset medio (a curva nao oscila simetricamente em torno de 0)
    if len(valleys) > 2:
        valley_amps = np.abs(theta_osc[valleys])
        asym = np.mean(peak_amps[:len(valley_amps)] - valley_amps[:len(peak_amps)])
        Tc_est = max(0.001, abs(asym) * np.pi/180 * TAU_GRAV / 5)
    else:
        Tc_est = 0.005

    print(f"\n  Picos encontrados: {len(peaks)}")
    print(f"  Decremento logaritmico medio: {delta_med:.4f}")
    print(f"  Periodo amortecido Td: {Td:.3f} s")
    print(f"  omega_d = {omega_d:.3f} rad/s")
    print(f"  omega_n = {omega_n:.3f} rad/s")
    print(f"  zeta    = {zeta:.4f}")
    print(f"\n  Estimativas:")
    print(f"    J  = {J_est:.5f} kg.m2")
    print(f"    b  = {b_est:.5f} N.m.s/rad")
    print(f"    Tc = {Tc_est:.5f} N.m")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_osc, theta_osc, 'b-', lw=1, label='Angulo medido')
    ax.plot(t_osc[peaks], theta_osc[peaks], 'rv', ms=10, label='Picos')
    if len(valleys) > 0:
        ax.plot(t_osc[valleys], theta_osc[valleys], 'g^', ms=10, label='Vales')
    # Envelope exponencial
    t_env = np.linspace(0, t_osc[peaks[-1]], 200)
    A0 = peak_amps[0]
    env = A0 * np.exp(-zeta * omega_n * t_env)
    ax.plot(t_env, env, 'r--', lw=2, label=f'Envelope: zeta={zeta:.4f}')
    ax.plot(t_env, -env, 'r--', lw=2)
    ax.set(xlabel='Tempo (s)', ylabel='Angulo (deg)',
           title=f'Decremento Logaritmico: J={J_est:.4f} b={b_est:.4f} Td={Td:.2f}s')
    ax.legend(); ax.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig('etapa2_decremento.png', dpi=150); plt.close()
    print("  Grafico -> etapa2_decremento.png")

    return J_est, b_est, Tc_est


# ===========================================================================
# ETAPA 3: Multiple Shooting Diferenciavel (PyTorch + torchdiffeq)
# ===========================================================================
class AeroODE(nn.Module):
    """ODE do aeropendulo: J*theta_dd = tau_motor - tau_grav - tau_fric"""
    def __init__(self, J0, b0, Tc0, K_motor, tipo_motor):
        super().__init__()
        self.K = K_motor
        self.tipo = tipo_motor
        self.log_J  = nn.Parameter(torch.tensor(np.log(max(J0, 1e-4)),  dtype=torch.float32))
        self.log_b  = nn.Parameter(torch.tensor(np.log(max(b0, 1e-6)),  dtype=torch.float32))
        self.log_Tc = nn.Parameter(torch.tensor(np.log(max(Tc0, 1e-6)), dtype=torch.float32))
        self._t_buf = None
        self._u_buf = None

    def set_input(self, t_ten, u_ten):
        self._t_buf = t_ten
        self._u_buf = u_ten

    def _interp_u(self, t):
        """ZOH do controle."""
        idx = torch.searchsorted(self._t_buf, t.clamp(max=self._t_buf[-1]))
        idx = idx.clamp(0, len(self._u_buf) - 1)
        return self._u_buf[idx]

    def forward(self, t, state):
        theta = state[0]
        omega = state[1]
        J  = torch.exp(self.log_J)
        b  = torch.exp(self.log_b)
        Tc = torch.exp(self.log_Tc)
        u  = self._interp_u(t)

        if self.tipo == 'linear':
            tau_m = self.K * u
        else:
            tau_m = self.K * u * torch.abs(u)

        tau_g = TAU_GRAV * torch.sin(theta)
        tau_f = b * omega + Tc * torch.tanh(50.0 * omega)
        alpha = (tau_m - tau_g - tau_f) / J
        return torch.stack([omega, alpha])

    def params_dict(self):
        return {
            'J':  torch.exp(self.log_J).item(),
            'b':  torch.exp(self.log_b).item(),
            'Tc': torch.exp(self.log_Tc).item(),
        }


def carregar_para_ms(path):
    """Carrega um experimento e prepara para multiple shooting."""
    df = pd.read_csv(path)
    theta = np.radians(df['angulo_deg'].values.astype(np.float64))
    u = np.clip(df['u_pct'].values.astype(np.float64) / 100.0, -1, 1)
    t = np.arange(len(theta)) * DT
    omega = savgol_filter(theta, 51, 3, deriv=1, delta=DT)
    return t, theta, omega, u, path.split('\\')[-1].replace('.csv', '')


def etapa3(K_motor, tipo_motor, J0, b0, Tc0):
    print("\n" + "="*65)
    print("  ETAPA 3: Multiple Shooting Diferenciavel (torchdiffeq)")
    print("="*65)

    # Carregar todos os experimentos
    exps = [carregar_para_ms(p) for p in PATHS_DEGRAUS]
    for t, th, om, u, nome in exps:
        print(f"  {nome}: {len(t)} pts")

    WINDOW = 200   # 2.0 s
    STRIDE = 200   # sem overlap

    # Preparar janelas
    windows = []
    for t, theta, omega, u, nome in exps:
        n = len(t)
        for i in range(0, n - WINDOW, STRIDE):
            j = i + WINDOW
            t_w = torch.tensor(t[i:j] - t[i], dtype=torch.float32).to(device)
            u_w = torch.tensor(u[i:j], dtype=torch.float32).to(device)
            th_w = torch.tensor(theta[i:j], dtype=torch.float32).to(device)
            x0 = torch.tensor([theta[i], omega[i]], dtype=torch.float32).to(device)
            windows.append((t_w, u_w, th_w, x0))

    print(f"  Janelas: {len(windows)} (W={WINDOW}, stride={STRIDE})")

    model = AeroODE(J0, b0, Tc0, K_motor, tipo_motor).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200,
                                                      factor=0.5, min_lr=1e-5)

    EPOCHS = 1500
    BATCH  = min(32, len(windows))
    best_loss = float('inf')
    best_state = None

    print(f"  Treinando ({EPOCHS} epocas, batch={BATCH}, {len(windows)} janelas)..."); sys.stdout.flush()
    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()
        indices = np.random.choice(len(windows), BATCH, replace=False)
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        count = 0

        for idx in indices:
            t_w, u_w, th_w, x0 = windows[idx]
            model.set_input(t_w, u_w)
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
        scheduler.step(mean_loss)

        if mean_loss.item() < best_loss:
            best_loss = mean_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 500 == 0:
            p = model.params_dict()
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:5d} | Loss: {mean_loss.item():.6f} | "
                  f"J={p['J']:.5f} b={p['b']:.5f} Tc={p['Tc']:.5f} | lr={lr:.1e}")
            sys.stdout.flush()

    model.load_state_dict(best_state)
    p = model.params_dict()

    print(f"\n  Parametros finais (best loss = {best_loss:.6f}):")
    print(f"    J  = {p['J']:.5f} kg.m2")
    print(f"    b  = {p['b']:.5f} N.m.s/rad")
    print(f"    Tc = {p['Tc']:.5f} N.m")

    wn = np.sqrt(TAU_GRAV / p['J'])
    zeta = p['b'] / (2 * wn * p['J'])
    print(f"    wn = {wn:.3f} rad/s   zeta = {zeta:.4f}   T = {2*np.pi/wn:.2f} s")

    return p['J'], p['b'], p['Tc'], exps


# ===========================================================================
# ETAPA 4: Validacao + Curva de Atrito
# ===========================================================================
def simular_segmentado(t, theta, omega, u, J, b, Tc, K, tipo, seg_s=2.0):
    """Simula reiniciando a cada seg_s segundos."""
    seg = int(seg_s / DT)
    n = len(t)
    theta_sim = np.full(n, np.nan)
    theta_sim[0] = theta[0]

    for start in range(0, n, seg):
        end = min(start + seg, n)
        state = np.array([theta[start], omega[start]])
        for k in range(start, end - 1):
            uk = u[k]
            def f(s):
                th, om = s
                tm = K * uk if tipo == 'linear' else K * uk * abs(uk)
                tg = TAU_GRAV * np.sin(th)
                tf = b * om + Tc * np.tanh(50.0 * om)
                return np.array([om, (tm - tg - tf) / J])
            k1 = f(state)
            k2 = f(state + DT/2*k1)
            k3 = f(state + DT/2*k2)
            k4 = f(state + DT*k3)
            state = state + DT/6*(k1 + 2*k2 + 2*k3 + k4)
            theta_sim[k+1] = state[0]
    return theta_sim


def etapa4(exps, J, b, Tc, K, tipo):
    print("\n" + "="*65)
    print("  ETAPA 4: Validacao + Curva de Atrito")
    print("="*65)

    fig, axes = plt.subplots(len(exps), 1, figsize=(15, 4.5*len(exps)), squeeze=False)
    for i, (t, theta, omega, u, nome) in enumerate(exps):
        theta_sim = simular_segmentado(t, theta, omega, u, J, b, Tc, K, tipo)
        v = ~np.isnan(theta_sim)
        rmse = np.sqrt(np.nanmean((np.degrees(theta) - np.degrees(theta_sim))**2))
        ax = axes[i, 0]
        ax.plot(t, np.degrees(theta), 'b-', lw=1.2, alpha=.8, label='Medido')
        ax.plot(t[v], np.degrees(theta_sim[v]), 'r-', lw=1.2, alpha=.8,
                label=f'Simulado (RMSE={rmse:.1f} deg, seg=2s)')
        ax.set(xlabel='Tempo (s)', ylabel='Angulo (deg)', title=nome)
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)
        print(f"  {nome}: RMSE = {rmse:.2f} deg")

    plt.suptitle(f'Validacao: J={J:.4f}  b={b:.4f}  Tc={Tc:.4f}', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('etapa4_validacao.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Grafico -> etapa4_validacao.png")

    # ----- Curva de Atrito -----
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    SG = 101
    m = SG // 2
    for t, theta, omega_raw, u, nome in exps:
        omega = savgol_filter(theta, SG, 3, deriv=1, delta=DT)
        alpha = savgol_filter(theta, SG, 3, deriv=2, delta=DT)
        s = slice(m, len(t)-m)
        if tipo == 'linear':
            tm = K * u[s]
        else:
            tm = K * u[s] * np.abs(u[s])
        tau_f = tm - TAU_GRAV*np.sin(theta[s]) - J*alpha[s]
        ax2.scatter(omega[s], tau_f, s=1, alpha=.2, label=nome)

    w = np.linspace(-5, 5, 500)
    tau_mod = b*w + Tc*np.tanh(50.0*w)
    ax2.plot(w, tau_mod, 'k-', lw=3, label=f'Modelo: b={b:.4f} + Tc={Tc:.4f}*sgn(w)')
    ax2.axhline(0, c='gray', lw=.5); ax2.axvline(0, c='gray', lw=.5)
    ax2.set(xlabel='Velocidade Angular (rad/s)', ylabel='Torque de Atrito (N.m)',
            title='Curva de Atrito Identificada')
    ax2.set_xlim(-5, 5)
    ylim = max(abs(b*5 + Tc), 0.5)
    ax2.set_ylim(-ylim*1.5, ylim*1.5)
    ax2.legend(fontsize=9, markerscale=10)
    ax2.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig('etapa4_curva_atrito.png', dpi=150)
    plt.close()
    print("  Grafico -> etapa4_curva_atrito.png")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == '__main__':
    K, tipo = etapa1()
    J0, b0, Tc0 = etapa2()
    J, b, Tc, exps = etapa3(K, tipo, J0, b0, Tc0)
    etapa4(exps, J, b, Tc, K, tipo)

    print("\n" + "="*65)
    print("  RESUMO FINAL")
    print("="*65)
    if tipo == 'linear':
        print(f"  Motor:    Ktau = {K:.5f} N.m    (tau = Ktau*u)")
    else:
        print(f"  Motor:    Gu   = {K:.5f} N.m    (tau = Gu*u*|u|)")
    print(f"  Inercia:  J    = {J:.5f} kg.m2")
    print(f"  Viscoso:  b    = {b:.5f} N.m.s/rad")
    print(f"  Coulomb:  Tc   = {Tc:.5f} N.m")
    wn = np.sqrt(TAU_GRAV / J)
    zeta = b / (2 * wn * J)
    print(f"  wn = {wn:.3f} rad/s   zeta = {zeta:.4f}   T = {2*np.pi/wn:.2f} s")
    print("="*65)
