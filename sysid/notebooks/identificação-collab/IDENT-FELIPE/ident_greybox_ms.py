import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import sys
import os

# Adiciona o diretório raiz para importar aerodata
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from aerodata import readData

def load_data():
    """
    Carrega o dataset e faz o corte para pegar uma faixa persistente do sinal.
    """
    print("Baixando/Lendo sinal multisseno local para identificação...")
    file_path = "data/coletas/RODADA-7_20260904/multi-seno-45-040Hz_0904_20-26.csv"
    y, u, t, ref = readData(dataset_name=file_path, return_ref=True)
    
    # Descartando transientes iniciais e o desligamento do motor no final
    trim_start = 10.0
    trim_end = 13.0
    idx_start = np.searchsorted(t, t[0] + trim_start)
    idx_end = np.searchsorted(t, t[-1] - trim_end)
    
    # Decima o sinal. Decimação = 1 significa 100 Hz (todas as amostras).
    decimation = 1
    sl = slice(idx_start, idx_end, decimation)
    
    y = y[sl]
    u = u[sl]
    t = t[sl]
    
    # Convertendo ângulo de graus para radianos para o modelo físico
    y_rad = np.deg2rad(y)
    
    # Centralizando t
    t = t - t[0]
    dt = float(np.mean(np.diff(t)))
    
    return y_rad, u, t, dt, y

# Constantes Físicas (Termo Gravitacional)
# A equação baseada no TCC: J * d2(theta)/dt2 + b * d(theta)/dt + (m1*L1 - m2*L2)*g*sin(theta) = Ktau * u
# Como os parâmetros geométricos e de massa são fáceis de medir na prática, podemos
# fixar C_g = (m1*L1 - m2*L2)*g e estimar apenas [J, b, Ktau].
# Valores default sugeridos (ajuste conforme a bancada real):
m1 = 0.122 # massa do conjunto motor-helice (kg)
L1 = 0.39  # distancia do motor ao pivo (m)
m2 = 0.055 # massa do contrapeso (kg)
L2 = 0.347 # distancia do contrapeso ao pivo (m)
g  = 9.81  # gravidade (m/s^2)

# Fator gravitacional constante da planta (Torque Restaurador em função de theta)
C_g = (m1 * L1 - m2 * L2) * g  

def run_multiple_shooting():
    y_meas, u_data, t, dt, y_deg = load_data()
    N = len(y_meas)
    
    print(f"\n--- Iniciando Otimização Multiple Shooting ---")
    print(f"Amostras: {N} | Passo (dt): {dt:.4f} s")
    print(f"Termo gravitacional C_g fixado em: {C_g:.5f}")
    
    # 1. Configuração CasADi 
    # Estados: x1 = theta (rad), x2 = dtheta/dt (rad/s)
    x = SX.sym('x', 2)
    u = SX.sym('u', 1)
    
    # Parâmetros que queremos descobrir (o Grey-Box)
    p = SX.sym('p', 3)
    J    = p[0]
    b    = p[1]
    Ktau = p[2]
    
    # Equação Diferencial Ordinária (ODE) do aeropêndulo
    xdot = vertcat(
        x[1],
        (Ktau * u - b * x[1] - C_g * sin(x[0])) / J
    )
    
    # 2. Integrador de Estados (Runge-Kutta 4)
    ode = Function('ode', [x, u, p], [xdot])
    k1 = ode(x, u, p)
    k2 = ode(x + dt/2 * k1, u, p)
    k3 = ode(x + dt/2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    F_rk4 = Function('F_rk4', [x, u, p], [x_next])
    
    # 3. Formulação Multiple Shooting (NLP)
    opti = Opti()
    
    X = opti.variable(2, N)
    P = opti.variable(3)
    
    # Bounds Físicos (Valores não podem ser negativos)
    opti.subject_to(P[0] >= 1e-6) # Inércia deve ser positiva
    opti.subject_to(P[1] >= 1e-6) # Atrito deve ser positivo
    opti.subject_to(P[2] >= 1e-6) # Empuxo deve ser positivo
    
    cost = 0
    
    print("Montando restrições de continuidade e função de custo...")
    for k in range(N-1):
        # A principal restrição do Multiple Shooting: O estado no nó k+1 
        # TEM QUE SER o estado integrado a partir do nó k.
        x_k_next = F_rk4(X[:, k], u_data[k], P)
        opti.subject_to(X[:, k+1] == x_k_next)
        
        # O objetivo é minimizar a diferença entre a predição em theta e a medição real
        cost += (X[0, k] - y_meas[k])**2
        
    cost += (X[0, N-1] - y_meas[N-1])**2
    opti.minimize(cost)
    
    # 4. Inicialização do Otimizador (Chutes Iniciais / Warm Start)
    # A velocidade angular (x2) não foi medida, então estimamos via derivada finita de theta
    v_meas = np.gradient(y_meas, dt)
    opti.set_initial(X[0, :], y_meas)
    opti.set_initial(X[1, :], v_meas)
    
    # Chute inicial para os parâmetros físicos (Valores base de referência)
    opti.set_initial(P[0], 0.005) # J
    opti.set_initial(P[1], 0.010) # b
    opti.set_initial(P[2], 0.010) # Ktau
    
    # 5. Solver IPOPT
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "print_level": 5}
    opti.solver('ipopt', p_opts, s_opts)
    
    try:
        sol = opti.solve()
        X_opt = sol.value(X)
        P_opt = sol.value(P)
        print("\n=============================================")
        print(" Otimização Concluída com Sucesso!")
    except Exception as e:
        print("\n=============================================")
        print(" Otimização falhou ou alcançou Max Iterations")
        # Mesmo falhando, extraímos a melhor tentativa
        X_opt = opti.debug.value(X)
        P_opt = opti.debug.value(P)
        
    print(f"--- Parâmetros Identificados (Modelo Grey-Box) ---")
    print(f" Inércia (J)       = {P_opt[0]:.6f} kg.m^2")
    print(f" Atrito (b)        = {P_opt[1]:.6f} N.m.s/rad")
    print(f" C. Empuxo (Ktau)  = {P_opt[2]:.6f} N.m/%")
    print("=============================================\n")
    
    # 6. Validação Gráfica
    y_sim_deg = np.rad2deg(X_opt[0, :])
    
    plt.figure(figsize=(14, 6))
    plt.plot(t, y_deg, 'k', label='Medição Real (Sensoriado)', alpha=0.7)
    plt.plot(t, y_sim_deg, 'r--', label='Modelo Grey-Box Otimizado', lw=2)
    plt.title('Identificação Grey-Box por Multiple Shooting')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Ângulo [graus]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    fig_path = os.path.join(current_dir, 'resultado_multiple_shooting.png')
    plt.savefig(fig_path)
    print(f"Gráfico comparativo salvo em: {fig_path}")
    
    plt.show(block=False)
    plt.pause(2)
    
    # 7. Validação Free-Run
    free_run_validation([float(P_opt[0]), float(P_opt[1]), float(P_opt[2])])

def free_run_validation(P_opt, dt_sim=0.01):
    print("\n--- Iniciando Validação Cruzada (Free-Run) ---")
    val_files = {
        'Chirp (45 deg)': 'data/coletas/RODADA-7_20260904/chirp-45-amp35_0904_20-13.csv',
        'Degraus (45 deg)': 'data/coletas/RODADA-7_20260904/seq-degraus-45-2_0904_20-45.csv'
    }
    
    J_opt, b_opt, Ktau_opt = P_opt
    
    # Recriar ODE e Integrador CasADi para simulação livre
    x = SX.sym('x', 2)
    u = SX.sym('u', 1)
    xdot = vertcat(
        x[1],
        (Ktau_opt * u - b_opt * x[1] - C_g * sin(x[0])) / J_opt
    )
    ode = Function('ode', [x, u], [xdot])
    k1 = ode(x, u)
    k2 = ode(x + dt_sim/2 * k1, u)
    k3 = ode(x + dt_sim/2 * k2, u)
    k4 = ode(x + dt_sim * k3, u)
    x_next = x + dt_sim/6 * (k1 + 2*k2 + 2*k3 + k4)
    F_rk4_sim = Function('F_rk4_sim', [x, u], [x_next])
    
    fig, axes = plt.subplots(len(val_files), 1, figsize=(14, 5*len(val_files)))
    if len(val_files) == 1: axes = [axes]
    
    for ax, (name, filepath) in zip(axes, val_files.items()):
        print(f"Rodando {name}...")
        y, u_data, t_full, ref = readData(dataset_name=filepath, return_ref=True)
        
        # Trim inicial (mesmo do treinamento)
        trim_start = 10.0
        trim_end = 13.0
        idx_start = np.searchsorted(t_full, t_full[0] + trim_start)
        idx_end = np.searchsorted(t_full, t_full[-1] - trim_end)
        sl = slice(idx_start, idx_end, 1) # decimação 1
        
        y_val = y[sl]
        u_val = u_data[sl]
        t_val = t_full[sl] - t_full[idx_start]
        
        N_val = len(y_val)
        x_sim = np.zeros((2, N_val))
        
        # Estado inicial
        x_sim[0, 0] = np.deg2rad(y_val[0])
        x_sim[1, 0] = (np.deg2rad(y_val[1]) - np.deg2rad(y_val[0])) / dt_sim
        
        # Roda o simulador em malha aberta (Free-Run)
        for k in range(N_val - 1):
            x_k_next = F_rk4_sim(x_sim[:, k], u_val[k])
            x_sim[:, k+1] = x_k_next.full().flatten()
            
        y_sim_deg = np.rad2deg(x_sim[0, :])
        rmse = np.sqrt(np.mean((y_val - y_sim_deg)**2))
        
        ax.plot(t_val, y_val, 'k', label='Medição Real (Sensoriado)', alpha=0.7)
        ax.plot(t_val, y_sim_deg, 'r--', label='Grey-Box (Free-Run)', lw=2)
        ax.set_title(f'Validação: {name} | RMSE = {rmse:.2f} graus')
        ax.set_xlabel('Tempo [s]')
        ax.set_ylabel('Ângulo [graus]')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    fig_path = os.path.join(current_dir, 'resultado_validacao_greybox.png')
    plt.savefig(fig_path)
    print(f"Gráfico de validação livre salvo em: {fig_path}")
    plt.show(block=False)
    plt.pause(2)

if __name__ == "__main__":
    run_multiple_shooting()
