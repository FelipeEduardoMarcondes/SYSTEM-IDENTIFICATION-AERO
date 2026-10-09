import os
import sys
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

# Adiciona o diretório raiz para importar aerodata
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from aerodata import readData

# --- Parâmetros Físicos da Bancada ---
m1, L1 = 0.122, 0.39
m2, L2 = 0.055, 0.347
g = 9.81
C_g = (m1*L1 - m2*L2)*g  # Torque máximo da gravidade: ~0.2795 N.m

def extract_static_gain_ramp():
    """
    Passo 1: Curva Semiestática
    Isola o ganho do motor (Ktau) assumindo aceleração e velocidade ~zero
    (rampa lenta, quase-estática). O empuxo é modelado como Ktau * u^2
    (empuxo da hélice ~ omega^2, omega ~ u).
    A média dos ramos de subida e descida (mesmo ângulo) cancela o atrito seco.
    """
    print("--- Fase 1: Identificação Semiestática (Rampa lenta) ---")
    file_path = "data/coletas/RODADA-3_20260807/dados_curva_semi_estatica_0807_17-36.csv"
    y_deg, u, t, _ = readData(dataset_name=file_path, return_ref=True)
    
    # Descarta transitório inicial (amostra com u espúrio)
    y_deg, u, t = y_deg[10:], u[10:], t[10:]
    
    # Separa ramo de subida e de descida (pelo ponto de ângulo máximo)
    i_max = int(np.argmax(y_deg))
    
    # Faixa útil de ângulo: evita zona morta inicial e o batente em ~90°
    # (onde sin(theta)~1 satura e o braço está apoiado, não em equilíbrio)
    th_min, th_max = 5.0, 80.0
    bins = np.arange(th_min, th_max + 1.0, 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    
    def mean_u_by_angle(sl):
        ang, uu = y_deg[sl], u[sl]
        out = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            m = (ang >= bins[i]) & (ang < bins[i+1])
            if m.sum() >= 5:
                out[i] = np.mean(uu[m])
        return out
    
    u_up = mean_u_by_angle(slice(0, i_max))
    u_down = mean_u_by_angle(slice(i_max, None))
    valid = ~np.isnan(u_up) & ~np.isnan(u_down)
    
    u_avg = 0.5 * (u_up[valid] + u_down[valid])
    Torque_g = np.sin(np.deg2rad(centers[valid]))   # sin(theta) normalizado
    
    # Equilíbrio (modelo do professor): 0 = c1*sin(th) + c3*u|u|  =>  sin(th) = k*u|u|
    # com k = c3/|c1| (independe de massas/comprimentos). Ajuste pela origem (MQ).
    u2 = u_avg * np.abs(u_avg)
    Ktau_opt = np.dot(u2, Torque_g) / np.dot(u2, u2)
    
    r2 = 1 - np.sum((Torque_g - Ktau_opt*u2)**2) / np.sum((Torque_g - Torque_g.mean())**2)
    print(f"Razão estática k = c3/|c1|: {Ktau_opt:.4e} 1/%^2 | R2 = {r2:.4f}  (Ktau físico = k*C_g = {Ktau_opt*C_g:.4e})")
    
    # Salva o gráfico comprobatório
    plt.figure(figsize=(8,5))
    plt.scatter(u_up[valid], Torque_g, s=12, alpha=0.5, label='Subida')
    plt.scatter(u_down[valid], Torque_g, s=12, alpha=0.5, label='Descida')
    plt.scatter(u_avg, Torque_g, s=12, c='k', label='Média dos ramos')
    u_line = np.linspace(0, max(u_avg.max(), u_up[valid].max()), 100)
    plt.plot(u_line, Ktau_opt * u_line**2, 'r-', lw=2, label=f'Ajuste k*u|u| = {Ktau_opt:.2e} (R²={r2:.3f})')
    plt.title("Curva Semiestática do Empuxo")
    plt.xlabel("Comando do Motor u (%)")
    plt.ylabel(r"$\sin(\theta)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'curva_semiestatica_rampa.png'))
    plt.close()
    
    return float(Ktau_opt)


def extract_static_gain():
    """
    Passo 1: Curva Semiestática por Escada (patamares)
    Cada patamar da referência é um equilíbrio (v=0, a=0). Usa-se a média da parte
    final de cada patamar (já assentada) de theta MEDIDO e de u. O empuxo é
    Ktau * u^2. A média entre patamares de subida e descida (mesma referência)
    cancela o atrito seco.
    """
    print("--- Fase 1: Identificação Semiestática (Escada) ---")
    file_path = "data/experimentos/coleta_dados/RODADA-4/escada-atrito-1_0819_19-08.csv"
    y_deg, u, t, ref = readData(dataset_name=file_path, return_ref=True)
    
    settle_frac = 0.4          # usa os últimos 40% de cada patamar
    th_max = 80.0              # evita o batente (~90°)
    
    # Segmenta em patamares onde a referência é constante
    change = np.flatnonzero(np.abs(np.diff(ref)) > 1e-6) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(ref)]
    i_peak = int(np.argmax(ref))
    
    plateaus = []  # (ref, theta_mean, u_mean, is_up)
    for s, e in zip(starts, ends):
        n = e - s
        if n < 100:   # patamar curto demais (< ~1 s)
            continue
        a = s + int((1 - settle_frac) * n)
        theta_m = np.mean(y_deg[a:e])
        u_m = np.mean(u[a:e])
        r = ref[s]
        if r <= 0 or theta_m > th_max or u_m < 5:
            continue
        plateaus.append((r, theta_m, u_m, s < i_peak))
    
    up = {p[0]: p for p in plateaus if p[3]}
    down = {p[0]: p for p in plateaus if not p[3]}
    common = sorted(set(up) & set(down))
    
    th_up = np.array([up[r][1] for r in common]); u_up = np.array([up[r][2] for r in common])
    th_dn = np.array([down[r][1] for r in common]); u_dn = np.array([down[r][2] for r in common])
    
    # Média dos dois ramos (cancela atrito seco)
    th_avg = np.deg2rad(0.5 * (th_up + th_dn))
    u_avg = 0.5 * (u_up + u_dn)
    Torque_g = np.sin(th_avg)   # sin(theta) normalizado
    
    # Equilíbrio (modelo do professor): sin(th) = k*u|u|, k = c3/|c1| (MQ pela origem)
    u2 = u_avg * np.abs(u_avg)
    Ktau_opt = np.dot(u2, Torque_g) / np.dot(u2, u2)
    r2 = 1 - np.sum((Torque_g - Ktau_opt*u2)**2) / np.sum((Torque_g - Torque_g.mean())**2)
    print(f"Patamares usados: {len(common)} pares subida/descida (refs: {common[0]:.0f}°..{common[-1]:.0f}°)")
    print(f"Razão estática k = c3/|c1|: {Ktau_opt:.4e} 1/%^2 | R2 = {r2:.4f}  (Ktau físico = k*C_g = {Ktau_opt*C_g:.4e})")
    
    plt.figure(figsize=(8,5))
    plt.scatter(u_up, np.sin(np.deg2rad(th_up)), s=30, alpha=0.7, label='Subida')
    plt.scatter(u_dn, np.sin(np.deg2rad(th_dn)), s=30, alpha=0.7, label='Descida')
    plt.scatter(u_avg, Torque_g, s=30, c='k', label='Média dos ramos')
    u_line = np.linspace(0, max(u_up.max(), u_dn.max()), 100)
    plt.plot(u_line, Ktau_opt * u_line**2, 'r-', lw=2, label=f'Ajuste k*u|u| = {Ktau_opt:.2e} (R²={r2:.3f})')
    plt.title("Curva Semiestática do Empuxo (Escada)")
    plt.xlabel("Comando do Motor u (%)")
    plt.ylabel(r"$\sin(\theta)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'curva_semiestatica.png'))
    plt.close()
    
    return float(Ktau_opt)


def run_dynamic_optimization(k_static):
    """
    Passo 2: Otimização Dinâmica (Multiple Shooting) — modelo do professor:
        theta'' = c1*sin(theta) + c2*theta' + c3*u|u| + Fa_NL(theta')
    com c3 = -k*c1 (k = c3/|c1| vindo da curva semiestática).
    Descobre c1, c2 e os parâmetros do atrito não linear (Coulomb + Stribeck), em rad/s^2.
    """
    print("\n--- Fase 2: Otimização Dinâmica com Atrito Não-Linear (Modelo do Professor) ---")
    file_path = "data/coletas/RODADA-7_20260904/multi-seno-45-040Hz_0904_20-26.csv"
    y_deg, u_data, t, _ = readData(dataset_name=file_path, return_ref=True)
    
    # Mesmo corte usado anteriormente
    trim_start = 10.0
    trim_end = 13.0
    idx_start = np.searchsorted(t, t[0] + trim_start)
    idx_end = np.searchsorted(t, t[-1] - trim_end)
    sl = slice(idx_start, idx_end, 1) # decimação 1 (100Hz)
    
    y_meas = np.deg2rad(y_deg[sl])
    u_val = u_data[sl]
    t_val = t[sl] - t[idx_start]
    dt = np.mean(np.diff(t_val))
    N = len(y_meas)
    
    print(f"Amostras: {N} | Passo (dt): {dt:.4f} s")
    
    opti = cs.Opti()
    X = opti.variable(2, N)
    
    # --- Parâmetros a Descobrir (forma em aceleração, rad/s^2) ---
    c1 = opti.variable() # c1 = -Cg/J   (gravidade)
    c2 = opti.variable() # c2 = -b/J    (atrito viscoso linear)
    Fc = opti.variable() # Coulomb em rad/s^2
    Fs = opti.variable() # Pico Stribeck em rad/s^2
    vs = opti.variable() # Velocidade crítica do Stribeck
    c3 = -k_static * c1  # c3 = Ktau/J, preso à curva semiestática (c3/|c1| = k)
    
    # Limites
    opti.subject_to(opti.bounded(-40.0, c1, -0.5))
    opti.subject_to(opti.bounded(-20.0, c2, 0.0))
    opti.subject_to(opti.bounded(0.0, Fc, 5.0))
    opti.subject_to(opti.bounded(0.0, Fs, 5.0))
    opti.subject_to(opti.bounded(0.01, vs, 5.0))
    
    # Chutes Iniciais (J~0.035: c1=-Cg/J, c2=-b/J, Fc/J...)
    opti.set_initial(c1, -C_g / 0.035)
    opti.set_initial(c2, -0.016 / 0.035)
    opti.set_initial(Fc, 0.005 / 0.035)
    opti.set_initial(Fs, 0.005 / 0.035)
    opti.set_initial(vs, 0.5)
    
    # Condição inicial
    opti.subject_to(X[0, 0] == y_meas[0])
    opti.subject_to(X[1, 0] == (y_meas[1] - y_meas[0])/dt)
    
    # Runge-Kutta 4 Integrator Inline
    def rk4_step(x_k, u_k):
        def f(x_, u_):
            fric_nl = -(Fc * cs.tanh(50 * x_[1])
                        + Fs * cs.exp(-(x_[1]/vs)**2) * cs.tanh(50 * x_[1]))   # Fa_NL
            acc = c1 * cs.sin(x_[0]) + c2 * x_[1] + c3 * u_ * cs.fabs(u_) + fric_nl
            return cs.vertcat(x_[1], acc)
            
        k1 = f(x_k, u_k)
        k2 = f(x_k + dt/2 * k1, u_k)
        k3 = f(x_k + dt/2 * k2, u_k)
        k4 = f(x_k + dt * k3, u_k)
        return x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Restrição de Continuidade (Múltiplos Tiros) e Custo
    cost = 0
    for k in range(N-1):
        opti.subject_to(X[:, k+1] == rk4_step(X[:, k], u_val[k]))
        cost += (X[0, k] - y_meas[k])**2
        
    opti.minimize(cost)
    
    # Configura o IPOPT
    p_opts = {"expand": True}
    s_opts = {"max_iter": 2000, "print_level": 5, "tol": 1e-3, "hessian_approximation": "limited-memory"}    
    opti.solver("ipopt", p_opts, s_opts)
    
    print("Iniciando otimização do IPOPT...")
    try:
        sol = opti.solve()
        val = sol.value
    except Exception as e:
        print("A otimização falhou ao convergir. Usando valores da última iteração...")
        val = opti.debug.value
    X_opt = val(X)
    c1_opt, c2_opt = val(c1), val(c2)
    Fc_opt, Fs_opt, vs_opt = val(Fc), val(Fs), val(vs)
    c3_opt = -k_static * c1_opt
    J_equiv = -C_g / c1_opt   # inércia equivalente (usando C_g físico)
        
    print("\n=============================================")
    print(" Parâmetros Identificados (Modelo do Professor, rad/s^2)")
    print(f" c1 (sin)          = {c1_opt:.6f}   [= -Cg/J]")
    print(f" c2 (theta')       = {c2_opt:.6f}   [= -b/J]")
    print(f" c3 (u|u|)         = {c3_opt:.6e}   [= -k*c1, k estático = {k_static:.4e}]")
    print(f" Coulomb (Fc)      = {Fc_opt:.6f} rad/s^2")
    print(f" Stribeck Max (Fs) = {Fs_opt:.6f} rad/s^2")
    print(f" Vel. Stribeck (vs)= {vs_opt:.6f} rad/s")
    print(f" -> J equivalente  = {J_equiv:.5f} kg.m^2 | b = {-c2_opt*J_equiv:.5f} N.m.s/rad | Ktau = {c3_opt*J_equiv:.4e}")
    print("=============================================")
    
    y_sim_deg = np.rad2deg(X_opt[0, :])
    y_real_deg = np.rad2deg(y_meas)
    
    plt.figure(figsize=(14, 6))
    plt.plot(t_val, y_real_deg, 'k', label='Medição Real')
    plt.plot(t_val, y_sim_deg, 'r--', label='Grey-Box (Linear + Coulomb + Stribeck)', lw=2)
    plt.title("Multiple Shooting com Atritos Não-Lineares (Viscoso + Coulomb + Stribeck)")
    plt.xlabel('Tempo [s]')
    plt.ylabel('Ângulo [graus]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'resultado_modelo_prof.png'))
    print("Gráfico do modelo novo salvo em: resultado_modelo_prof.png")
    plt.close()

if __name__ == "__main__":
    Ktau_descobridor = extract_static_gain()
    Ktau_rampa = extract_static_gain_ramp()
    print(f"Comparação k=c3/|c1|: escada = {Ktau_descobridor:.4e} | rampa = {Ktau_rampa:.4e}")
    run_dynamic_optimization(Ktau_descobridor)
