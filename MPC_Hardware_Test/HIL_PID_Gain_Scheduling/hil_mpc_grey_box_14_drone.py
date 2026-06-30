# ─────────────────────────────────────────────────────────────────────────────
# HIL — MPC GREY-BOX para o 1/4-drone (aeropêndulo)
# ─────────────────────────────────────────────────────────────────────────────
# Implementa o controlador ótimo preditivo de mpc_grey_box_14_drone.ipynb em
# malha fechada via SERIAL com o Arduino (emulador ou slave real).
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse
from tqdm.auto import tqdm
from casadi import SX, MX, DM, Function, nlpsol, vertcat, sqrt

parser = argparse.ArgumentParser(description="HIL MPC Grey-Box")
parser.add_argument('--port', default='COM8', help="Porta Serial do Arduino")
args = parser.parse_args()

# ==========================================
# 0. CHAVES DE CONFIGURAÇÃO
# ==========================================
REALTIME     = True
SERIAL_PORT  = args.port
BAUD_RATE    = 115200

# ==========================================
# 1. PARÂMETROS DO MODELO FÍSICO
# ==========================================
c1 = -8.29151533
c2 = 0.00418887
c3 = -1.43673669

Ts = 0.05
nx = 2
N = 20

# ==========================================
# 2. SETUP DO CASADI (MPC)
# ==========================================
print("Montando o problema de otimização no CasADi...")
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
    'u_min': np.array([-100.0]),
    'u_max': np.array([100.0]),
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
# 3. SINAL DE REFERÊNCIA (Mesmo do Notebook)
# ==========================================
T_sim = 190.0 # Same length as the PID script for comparison
steps = int(T_sim / Ts)
t_steps = np.arange(0, T_sim + N*Ts, Ts)

def get_ref(t):
    step = int(t // 5) # 5 seconds per step
    if step <= 18:
        return step * 10.0
    else:
        return max(180.0 - (step - 18) * 10.0, 0.0)

x2ref_full = np.array([get_ref(t) for t in t_steps])
ref_signal = x2ref_full[:steps]
n_steps = len(ref_signal)
t_vec = np.arange(n_steps) * Ts


# ==========================================
# 4. CONEXÃO SERIAL
# ==========================================
try:
    import serial
except ImportError:
    print("A biblioteca 'pyserial' não está instalada. Execute 'pip install pyserial'.")
    sys.exit(1)

print(f"Conectando ao Arduino em {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print("Aguardando 6s para a calibração automática do IMU/Filtro no Arduino...")
    time.sleep(6)
    print("Conectado com sucesso ao Arduino!")
    input("\n[ATENÇÃO] Pressione ENTER para iniciar a comunicação serial... ")
except Exception as e:
    print(f"FALHA na conexão com o Arduino: {e}")
    sys.exit(1)


# ==========================================
# 5. SINCRONIZAÇÃO INICIAL
# ==========================================
y_history, u_history, dt_history = [], [], []

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
    if REALTIME:
        time.sleep(Ts)

print(f"Estado inicial lido: {y_atual:.2f} graus. Iniciando controle MPC!")


# ==========================================
# 6. LAÇO DE CONTROLE HIL — MPC
# ==========================================
print("\nIniciando Loop de Controle HIL...\n")

xsim = np.zeros(nx)
xsim[0] = np.deg2rad(y_atual)
u_prev_sim = 0.0
w0_val = w0.full().flatten()

for k in range(n_steps):
    loop_start = time.time()
    
    r_deg = ref_signal[k]
    
    # Prepara vetor de parâmetros (estado atual + ref futura + u anterior)
    ref_window = x2ref_full[k : k + N]
    pval = np.concatenate([xsim, ref_window, [u_prev_sim]])
    
    # Chama o Solver Otimizador
    tic = time.perf_counter()
    sol = solver(x0=w0_val, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
    solve_time = time.perf_counter() - tic
    dt_history.append(solve_time)
    
    w_opt = sol['x'].full().flatten()
    u_opt = w_opt[nx] # primeira ação de controle da janela ótima
    
    # --- Troca com o hardware/emulador ---
    ser.write(f"{u_opt:.4f}\n".encode())
    reply = ser.readline().decode('utf-8', errors='ignore').strip()
    y_meas = y_atual
    if reply:
        try:
            y_meas = float(reply)
            y_atual = y_meas
        except ValueError:
            print(f"Resposta inválida do Arduino: '{reply}'")
            
    y_history.append(y_meas)
    u_history.append(u_opt)
    
    # Atualiza o estado usando a nova leitura para a próxima iteração
    # O observador de estado (se houvesse) iria aqui. Por ora estimamos que a derivada é
    # a diferença na leitura (aproximação simples) ou atualizamos xsim com o modelo.
    # Como y_meas é direto o angulo em graus, xsim[0] = rad. xsim[1] = rad/s
    y_meas_rad = np.deg2rad(y_meas)
    xsim[1] = (y_meas_rad - xsim[0]) / Ts
    xsim[0] = y_meas_rad
    
    # Warm start
    w0_val = w_opt
    u_prev_sim = u_opt

    # --- Tempo real estrito ---
    if REALTIME:
        elapsed = time.time() - loop_start
        if elapsed < Ts:
            time.sleep(Ts - elapsed)
        elif k > 0:
            print(f"AVISO: loop atrasado em {elapsed - Ts:.3f}s (step {k}) | Solver: {solve_time*1000:.1f}ms")

    if k % 20 == 0:
        print(f"Step {k:04d} | Ref: {r_deg:5.1f} | u: {u_opt:7.2f} | y: {y_meas:7.2f} | Solver: {solve_time*1000:.1f}ms")

# Desliga o motor por segurança
print("\nDesligando motor por segurança...")
ser.write(b"0.0000\n")
ser.close()

print("Experimento HIL (MPC) finalizado! Salvando dados e gráficos...")

# ==========================================
# 7. SALVA RESULTADOS
# ==========================================
out_dir = os.path.dirname(os.path.abspath(__file__))
out_csv = os.path.join(out_dir, "hil_mpc_hardware.csv")

import csv as _csv
with open(out_csv, 'w', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['Tempo_s', 'Referencia', 'Saida_y', 'Controle_u', 'Solver_ms'])
    for i in range(n_steps):
        w.writerow([f"{t_vec[i]:.4f}", f"{ref_signal[i]:.4f}",
                    f"{y_history[i]:.4f}", f"{u_history[i]:.4f}", f"{dt_history[i]*1000:.2f}"])
print(f"Dados salvos em: {out_csv}")

# ==========================================
# 8. GRÁFICOS
# ==========================================
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t_vec, ref_signal, 'k--', lw=2, label='Referência')
plt.plot(t_vec, y_history, 'b-', lw=1.5, label='Saída y (Arduino)')
plt.ylabel('Ângulo [graus]')
plt.title('HIL MPC Grey-Box — Teste Hardware')
plt.legend(); plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_vec, u_history, 'm-', lw=1.5, label='Controle u [%]')
plt.axhline(data['u_max'][0], color='k', ls=':', alpha=0.5)
plt.axhline(data['u_min'][0], color='k', ls=':', alpha=0.5)
plt.ylabel('Controle u [%]')
plt.legend(); plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_vec, np.array(dt_history) * 1000, 'g-', lw=1.5)
plt.axhline(np.mean(dt_history)*1000, color='r', ls=':', label=f'Média: {np.mean(dt_history)*1000:.1f}ms')
plt.axhline(50.0, color='k', ls='--', lw=2, label='Limite Real-Time (50ms)')
plt.ylabel('Solver Time [ms]'); plt.xlabel('Tempo [s]')
plt.legend(); plt.grid(True)

plt.tight_layout()
out_png = os.path.join(out_dir, "hil_mpc_hardware.png")
plt.savefig(out_png, dpi=110)
print(f"Gráfico salvo em: {out_png}")
plt.show()
