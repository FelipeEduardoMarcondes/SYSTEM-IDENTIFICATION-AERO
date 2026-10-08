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

def extract_static_gain():
    """
    Passo 1: Curva Semiestática
    Isola o ganho do motor (Ktau) ou g(u) assumindo aceleração e velocidade zero
    em pontos de repouso (steady-state).
    """
    print("--- Fase 1: Identificação Semiestática (Degraus) ---")
    file_path = "data/coletas/RODADA-7_20260904/seq-degraus-45-2_0904_20-45.csv"
    y_deg, u, t, _ = readData(dataset_name=file_path, return_ref=True)
    
    y_rad = np.deg2rad(y_deg)
    dt = np.mean(np.diff(t))
    
    # Calcular velocidade (rad/s)
    v_rad = np.gradient(y_rad, dt)
    
    # Filtrar pontos de estado estacionário (onde a velocidade do braço é ~0)
    static_mask = np.abs(v_rad) < 0.02
    
    u_static = u[static_mask]
    y_static = y_rad[static_mask]
    
    # Filtra u=0 (desligado) para evitar divisão por zero no ganho
    active_mask = u_static > 20.0
    u_static = u_static[active_mask]
    y_static = y_static[active_mask]
    
    # A equação estática é: Ktau * u = C_g * sin(theta) => Ktau = C_g * sin(theta) / u
    sin_theta = np.sin(y_static)
    Torque_g = C_g * sin_theta
    
    # Ajuste Linear de Reta passando pela origem (Mínimos Quadrados)
    # Y = Ktau * X -> Torque_g = Ktau * u
    Ktau_opt = np.dot(u_static, Torque_g) / np.dot(u_static, u_static)
    
    print(f"Ganho Ktau isolado pela curva semiestática: {Ktau_opt:.6f} N.m/%")
    
    # Salva o gráfico comprobatório
    plt.figure(figsize=(8,5))
    plt.scatter(u_static, Torque_g, alpha=0.1, label='Pontos Estacionários Medidos')
    u_line = np.linspace(0, max(u_static), 100)
    plt.plot(u_line, Ktau_opt * u_line, 'r-', lw=2, label=f'Ajuste Ktau (Linear) = {Ktau_opt:.5f}')
    plt.title("Curva Semiestática do Empuxo")
    plt.xlabel("Comando do Motor u (%)")
    plt.ylabel("Torque Gravitacional Resistente $C_g \sin(\\theta)$ [N.m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'curva_semiestatica.png'))
    plt.close()
    
    return float(Ktau_opt)


def run_dynamic_optimization(Ktau_fixed):
    """
    Passo 2: Otimização Dinâmica
    Descobre Inércia, Atrito Linear, Coulomb e Stribeck usando o Ktau já descoberto.
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
    
    # --- Parâmetros Ocultos a Descobrir ---
    J  = opti.variable() # Inércia
    b  = opti.variable() # Atrito Viscoso Linear
    Fc = opti.variable() # Atrito Seco Constante (Coulomb)
    Fs = opti.variable() # Pico de Atrito Estático (Stribeck)
    vs = opti.variable() # Velocidade Crítica do Stribeck
    
    # Limites físicos
    opti.subject_to(opti.bounded(0.01, J, 0.2))
    opti.subject_to(opti.bounded(0.0, b, 0.1))
    opti.subject_to(opti.bounded(0.0, Fc, 0.1))
    opti.subject_to(opti.bounded(0.0, Fs, 0.1))
    opti.subject_to(opti.bounded(0.01, vs, 5.0))
    
    # Chutes Iniciais
    opti.set_initial(J, 0.035)
    opti.set_initial(b, 0.016)
    opti.set_initial(Fc, 0.005)
    opti.set_initial(Fs, 0.005)
    opti.set_initial(vs, 0.5)
    
    # Condição inicial
    opti.subject_to(X[0, 0] == y_meas[0])
    opti.subject_to(X[1, 0] == (y_meas[1] - y_meas[0])/dt)
    
    # Runge-Kutta 4 Integrator Inline
    def rk4_step(x_k, u_k):
        def f(x_, u_):
            fric_lin = b * x_[1]
            fric_coul = Fc * cs.tanh(50 * x_[1])
            fric_strib = Fs * cs.exp(-(x_[1]/vs)**2) * cs.tanh(50 * x_[1])
            fric_tot = fric_lin + fric_coul + fric_strib
            return cs.vertcat(x_[1], (Ktau_fixed * u_ - fric_tot - C_g * cs.sin(x_[0])) / J)
            
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
    s_opts = {"max_iter": 500, "print_level": 5, "tol": 1e-4, "hessian_approximation": "limited-memory"}
    opti.solver("ipopt", p_opts, s_opts)
    
    print("Iniciando otimização do IPOPT...")
    try:
        sol = opti.solve()
        X_opt = sol.value(X)
        J_opt = sol.value(J)
        b_opt = sol.value(b)
        Fc_opt = sol.value(Fc)
        Fs_opt = sol.value(Fs)
        vs_opt = sol.value(vs)
    except Exception as e:
        print("A otimização falhou ao convergir. Usando valores da última iteração...")
        X_opt = opti.debug.value(X)
        J_opt = opti.debug.value(J)
        b_opt = opti.debug.value(b)
        Fc_opt = opti.debug.value(Fc)
        Fs_opt = opti.debug.value(Fs)
        vs_opt = opti.debug.value(vs)
        
    print("\n=============================================")
    print(" Parâmetros Identificados (Modelo do Professor)")
    print(f" Inércia (J)       = {J_opt:.6f} kg.m^2")
    print(f" Atrito Viscoso (b)= {b_opt:.6f} N.m.s/rad")
    print(f" Coulomb (Fc)      = {Fc_opt:.6f} N.m")
    print(f" Stribeck Max (Fs) = {Fs_opt:.6f} N.m")
    print(f" Vel. Stribeck (vs)= {vs_opt:.6f} rad/s")
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
    run_dynamic_optimization(Ktau_descobridor)
