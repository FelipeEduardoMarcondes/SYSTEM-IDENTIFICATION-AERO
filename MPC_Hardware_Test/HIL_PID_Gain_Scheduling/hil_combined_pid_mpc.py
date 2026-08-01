# ─────────────────────────────────────────────────────────────────────────────
# HIL — COMBINED PID and MPC GREY-BOX para o 1/4-drone
# ─────────────────────────────────────────────────────────────────────────────
# FASE 1 (0 a 15s): PID (0 a 5s em 0º, 5 a 15s em 30º)
# FASE 2 (15 a 40s): MPC (step em 45º)
# Todas as taxas de amostragem em 20 Hz (Ts = 0.05s)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse
import pandas as pd
from casadi import SX, MX, DM, Function, nlpsol, vertcat, sqrt

parser = argparse.ArgumentParser(description="HIL Combined PID and MPC")
parser.add_argument('--port', default='COM8', help="Porta Serial do Arduino")
args = parser.parse_args()

# ==========================================
# 0. CHAVES DE CONFIGURAÇÃO GERAIS
# ==========================================
REALTIME     = True
SERIAL_PORT  = args.port
BAUD_RATE    = 115200

# Tempos
T_total = 40.0
T_pid = 15.0

Ts = 0.05
nx = 2
N = 20 # Horizonte MPC

steps_total = int(T_total / Ts)
steps_pid = int(T_pid / Ts)
steps_mpc = steps_total - steps_pid

t_vec = np.arange(steps_total) * Ts

# ==========================================
# 1. PARÂMETROS DO MODELO FÍSICO (MPC)
# ==========================================
c1 = -8.29151533
c2 = 0.00418887
c3 = -1.43673669

# ==========================================
# 2. SETUP DO CASADI (MPC)
# ==========================================
print("Montando o problema de otimização no CasADi (aguarde)...")
x = SX.sym('x', nx)
u_sym = SX.sym('u')

def f_dyn(x_vec, u_val):
    return vertcat(x_vec[1], c1*np.sin(x_vec[0]) + c2*u_val*sqrt(u_val**2 + 1e-4) + c3*x_vec[1])

k1 = f_dyn(x, u_sym)
k2 = f_dyn(x + Ts/2 * k1, u_sym)
k3 = f_dyn(x + Ts/2 * k2, u_sym)
k4 = f_dyn(x + Ts * k3, u_sym)
x_next = x + Ts/6 * (k1 + 2*k2 + 2*k3 + k4)
y_k = x_next[0]

F = Function('F', [x, u_sym], [x_next, y_k], ['x0', 'p'], ['xf', 'yk'])

data = {
    'u_min': np.array([0.0]),
    'u_max': np.array([50.0]),
    'u_guess': np.array([1.0]),
    'x_guess': np.zeros(nx),
}

def vcat(lst):
    return vertcat(*[DM(x) if not hasattr(x, 'is_symbolic') else x for x in lst])

w, lbw, ubw, w0 = [], [], [], []
g, lbg, ubg = [], [], []
J = 0

xk_param = MX.sym('xk_param', nx)
Pref = MX.sym('Pref', N)
u_prev_param = MX.sym('u_prev_param', 1)

xk = MX.sym('x0', nx)
w.append(xk)
lbw.append(np.full(nx, -np.inf))
ubw.append(np.full(nx, np.inf))
w0.append(data['x_guess'])

g.append(xk - xk_param)
lbg.append(np.zeros(nx))
ubg.append(np.zeros(nx))

for k in range(N):
    uk = MX.sym(f'u_{k}', 1)
    w.append(uk)
    lbw.append(data['u_min'])
    ubw.append(data['u_max'])
    w0.append(data['u_guess'])
    
    Fk = F(x0=xk, p=uk)
    xnext = Fk['xf']
    yk_deg = Fk['yk'] * 180.0 / np.pi
    
    if k == 0:
        du = uk - u_prev_param
    else:
        du = uk - u_prev
    u_prev = uk
    
    J = J + 1e3 * (yk_deg - Pref[k])**2 + 10 * uk**2 + 200 * du**2
    
    xk = MX.sym(f'x_{k+1}', nx)
    w.append(xk)
    lbw.append(np.full(nx, -np.inf))
    ubw.append(np.full(nx, np.inf))
    w0.append(data['x_guess'])
    
    g.append(xk - xnext)
    lbg.append(np.zeros(nx))
    ubg.append(np.zeros(nx))

w = vertcat(*w)
lbw = vcat(lbw)
ubw = vcat(ubw)
w0 = vcat(w0)
g = vertcat(*g)
lbg = vcat(lbg)
ubg = vcat(ubg)

nlp = {'x': w, 'g': g, 'f': J, 'p': vertcat(xk_param, Pref, u_prev_param)}
solver = nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
print("Solver compilado com sucesso!\n")

# ==========================================
# 3. SINAL DE REFERÊNCIA CONTÍNUO
# ==========================================
# Precisamos de referência para T_total + o horizonte N do MPC
t_ref_full = np.arange(0, T_total + N*Ts, Ts)
ref_full = np.zeros_like(t_ref_full)

for i, t in enumerate(t_ref_full):
    if t < 5.0:
        ref_full[i] = 0.0
    elif t < 15.0:
        ref_full[i] = 30.0
    else:
        ref_full[i] = 45.0

# O vetor final usado nos plots
ref_signal = ref_full[:steps_total]

# ==========================================
# 4. CONEXÃO SERIAL
# ==========================================
try:
    import serial
except ImportError:
    print("A biblioteca 'pyserial' não está instalada.")
    sys.exit(1)

print(f"Conectando ao Arduino em {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print("Aguardando 6s para a calibração automática do IMU/Filtro no Arduino...")
    time.sleep(6)
    print("Conectado com sucesso ao Arduino!")
    input("\n[ATENÇÃO] Pressione ENTER para LIGAR O MOTOR e iniciar o voo... ")
except Exception as e:
    print(f"FALHA na conexão com o Arduino: {e}")
    sys.exit(1)

# ==========================================
# 5. SINCRONIZAÇÃO INICIAL
# ==========================================
y_history = []
u_history = []
dt_history = []
mode_history = [] # Para saber quem estava controlando (PID ou MPC)

print("\nSincronizando estado inicial com o Arduino...")
ser.reset_input_buffer()
ser.reset_output_buffer()

y_atual = 0.0
for _ in range(50):
    ser.write(b"0.0000\n")
    reply = ser.readline().decode('utf-8', errors='ignore').strip()
    if reply:
        try:
            y_atual = float(reply)
        except ValueError:
            pass
    time.sleep(Ts)

print(f"Estado inicial lido: {y_atual:.2f} graus. Iniciando controle!")

# ==========================================
# 6. LAÇO DE CONTROLE HIL (PID -> MPC)
# ==========================================

# Variáveis do PID
Kp = 0.5793
Ki = 0.6647
Kd = 0.2
e_1 = ref_full[0] - y_atual
u_i = 0.0
y_1 = y_atual

# Variáveis do MPC (serão setadas na transição)
w0_val = w0.full().flatten()
xsim = np.zeros(nx)
u_prev_sim = 0.0

print("\n>>> INICIANDO FASE 1: PID (0s a 15s) a 20Hz <<<")

for step in range(steps_total):
    loop_start = time.time()
    r_deg = ref_signal[step]
    
    # ----------------------------------------
    # FASE 1: PID
    # ----------------------------------------
    if step < steps_pid:
        erro = r_deg - y_atual
        u_p = Kp * erro
        u_i = u_i + Ki * (Ts / 2.0) * (erro + e_1)
        u_d = -(Kd / Ts) * (y_atual - y_1)
        u_calc = u_p + u_i + u_d
        
        u_opt = np.clip(u_calc, -10.0, 80.0) # limites que estavam no PID
        
        # Anti-windup
        if u_calc != u_opt:
            u_i -= (u_calc - u_opt)
            
        e_1 = erro
        y_1 = y_atual
        
        solve_time = 0.0
        mode = "PID"
        
        # Se for o último passo do PID, prepara o estado para o MPC
        if step == steps_pid - 1:
            print(f"\n>>> INICIANDO FASE 2: MPC (15s a {T_total}s) a 20Hz <<<")
            y_meas_rad = np.deg2rad(y_atual)
            xsim[0] = y_meas_rad
            xsim[1] = 0.0 # Assume velocidade angular zero na transição
            u_prev_sim = u_opt

    # ----------------------------------------
    # FASE 2: MPC
    # ----------------------------------------
    else:
        # Pega a janela de referência futura
        ref_window = ref_full[step : step + N]
        pval = np.concatenate([xsim, ref_window, [u_prev_sim]])
        
        tic = time.perf_counter()
        sol = solver(x0=w0_val, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
        solve_time = time.perf_counter() - tic
        
        w_opt = sol['x'].full().flatten()
        u_opt = w_opt[nx] # Primeira ação de controle
        
        # Prepara warm start para a próxima iteração
        w0_val = w_opt
        u_prev_sim = u_opt
        
        mode = "MPC"

    # --- Comunicação com o Hardware ---
    y_meas = y_atual
    if ser.is_open:
        ser.write(f"{u_opt:.4f}\n".encode())
        reply = ser.readline().decode('utf-8', errors='ignore').strip()
        
        if reply:
            try:
                y_meas = float(reply)
                y_atual = y_meas
            except ValueError:
                print(f"Erro ao ler Arduino: '{reply}'.")
                
    # --- Atualização de Estados e Histórico ---
    if mode == "MPC":
        y_meas_rad = np.deg2rad(y_meas)
        xsim[1] = (y_meas_rad - xsim[0]) / Ts # Velocidade angular (aproximação)
        xsim[0] = y_meas_rad

    y_history.append(y_meas)
    u_history.append(u_opt)
    dt_history.append(solve_time)
    mode_history.append(mode)

    # --- Tempo Real Estrito ---
    if REALTIME:
        elapsed_time = time.time() - loop_start
        if elapsed_time < Ts:
            time.sleep(Ts - elapsed_time)
        elif step > 0:
            print(f"AVISO: Loop atrasado em {elapsed_time - Ts:.3f}s (Step {step})")

    if step % 20 == 0:
        if mode == "PID":
            print(f"[{mode}] Step {step:04d} | Ref: {r_deg:5.1f} | u: {u_opt:7.2f} | y: {y_meas:7.2f}")
        else:
            print(f"[{mode}] Step {step:04d} | Ref: {r_deg:5.1f} | u: {u_opt:7.2f} | y: {y_meas:7.2f} | Solver: {solve_time*1000:.1f}ms")

# Desliga o motor por segurança ao fim
print("\nDesligando motor por segurança...")
if ser.is_open:
    ser.write(b"0.0000\n")
    ser.close()

print("Experimento HIL finalizado! Salvando dados e gráficos...")

# ==========================================
# 7. SALVA RESULTADOS
# ==========================================
out_dir = os.path.dirname(os.path.abspath(__file__))
out_csv = os.path.join(out_dir, "hil_combined_pid_mpc_hardware.csv")

df_out = pd.DataFrame({
    'Tempo_s': t_vec,
    'Referencia': ref_signal,
    'Saida_y': y_history,
    'Controle_u': u_history,
    'Solver_ms': np.array(dt_history) * 1000,
    'Controlador': mode_history
})
df_out.to_csv(out_csv, index=False)
print(f"Dados salvos com sucesso em: {out_csv}")

# ==========================================
# 8. GRÁFICOS
# ==========================================
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t_vec, ref_signal, 'k--', lw=2, label='Referência')
plt.plot(t_vec, y_history, 'b-', lw=1.5, label='Saída y (Arduino)')
plt.axvline(x=T_pid, color='r', linestyle='--', label='Transição PID -> MPC')
plt.ylabel('Ângulo [graus]')
plt.title(f'HIL: PID (0-15s) + MPC Grey-Box (15-{int(T_total)}s) - Ts=0.05s')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_vec, u_history, 'm-', lw=1.5, label='Controle u [%]')
plt.axvline(x=T_pid, color='r', linestyle='--')
plt.axhline(data['u_max'][0], color='k', ls=':', alpha=0.5)
plt.axhline(data['u_min'][0], color='k', ls=':', alpha=0.5)
plt.ylabel('Controle u [%]')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_vec, np.array(dt_history) * 1000, 'g-', lw=1.5, label='Tempo do Solver (ms)')
plt.axvline(x=T_pid, color='r', linestyle='--')
plt.axhline(50.0, color='k', ls='--', lw=2, label='Limite Real-Time (50ms)')
plt.ylabel('Tempo [ms]')
plt.xlabel('Tempo [s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
out_png = os.path.join(out_dir, "hil_combined_pid_mpc_hardware.png")
plt.savefig(out_png, dpi=110)
print(f"Gráfico salvo em: {out_png}")
plt.show()
