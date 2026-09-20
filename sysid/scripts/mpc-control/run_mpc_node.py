import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm.auto import tqdm
from casadi import SX, MX, DM, Function, nlpsol, vertcat
import casadi as ca
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# 1. Definições Iniciais
# ---------------------------------------------------------
Ts = 0.05 # Lembre-se que o NODE foi treinado com dt=0.01 e decimacao=2 -> 0.02, mas vamos usar 0.05 pro MPC
nx = 2 # Estado contínuo: [theta_rad, theta_dot_rad_s]

# ---------------------------------------------------------
# 2. Carregar o Modelo PyTorch NODE v3 (Híbrido)
# ---------------------------------------------------------
class BaseODE(nn.Module):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0):
        super().__init__()
        self.m1, self.L1 = 0.122, 0.39
        self.m2, self.L2 = 0.055, 0.347
        self.g = 9.81
        self.log_J = nn.Parameter(torch.log(torch.tensor(float(J0))))
        self.log_b = nn.Parameter(torch.log(torch.tensor(float(b0))))
        self.log_Gu = nn.Parameter(torch.log(torch.tensor(float(Gu0))))

class PhysicsODE_Hybrid(BaseODE):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0, hidden_dim=16):
        super().__init__(J0, b0, Gu0)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "modelos_salvos", "node_v4_hybrid_20260822_105519.pth")
pytorch_model = PhysicsODE_Hybrid()

if os.path.exists(model_path):
    pytorch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Modelo carregado de {model_path}")
else:
    print(f"AVISO: {model_path} não encontrado! Usando pesos aleatórios (e parâmetros padrão) para demonstração.")
pytorch_model.eval()

# ---------------------------------------------------------
# 3. Transcrição Simbólica para CasADi (Tempo Contínuo + RK4)
# ---------------------------------------------------------
# Extrai parâmetros físicos
J = np.exp(pytorch_model.log_J.item())
b = np.exp(pytorch_model.log_b.item())
Gu = np.exp(pytorch_model.log_Gu.item())

# Extrai matrizes da MLP do atrito residual
W0 = pytorch_model.mlp[0].weight.detach().numpy()
b0 = pytorch_model.mlp[0].bias.detach().numpy()
W1 = pytorch_model.mlp[2].weight.detach().numpy()
b1 = pytorch_model.mlp[2].bias.detach().numpy()

# Constantes físicas fixas
m1, L1 = 0.122, 0.39
m2, L2 = 0.055, 0.347
g_grav = 9.81

x_sym = ca.MX.sym('x', nx) 
u_in_sym = ca.MX.sym('u', 1) 

def f_dyn(x_vec, u_val):
    theta = x_vec[0]
    theta_dot = x_vec[1]
    
    # Torque do Motor com aproximação suave para não zerar o gradiente do IPOPT em u=0!
    motor_torque = Gu * u_val * ca.sqrt(u_val**2 + 1e-4)
    gravity_torque = (m1*L1 - m2*L2) * g_grav * ca.sin(theta)
    friction_torque = b * theta_dot
    
    # MLP (Atrito residual ou efeitos aerodinâmicos complexos)
    nn_input = ca.vertcat(theta, theta_dot, u_val)
    h1 = ca.tanh(ca.mtimes(W0, nn_input) + b0)
    residual_torque = ca.mtimes(W1, h1) + b1
    
    theta_ddot = (motor_torque - gravity_torque - friction_torque + residual_torque) / J
    return ca.vertcat(theta_dot, theta_ddot)

# Integrador Numérico RK4 Manual (Para manter o Hessian rápido e simbólico)
k1 = f_dyn(x_sym, u_in_sym)
k2 = f_dyn(x_sym + Ts/2 * k1, u_in_sym)
k3 = f_dyn(x_sym + Ts/2 * k2, u_in_sym)
k4 = f_dyn(x_sym + Ts * k3, u_in_sym)
x_next = x_sym + Ts/6 * (k1 + 2*k2 + 2*k3 + k4)

# Converter radianos para graus para simplificar o cálculo do custo do MPC
y_k_deg = x_next[0] * (180.0 / np.pi)

F = ca.Function('F', [x_sym, u_in_sym], [x_next, y_k_deg], ['x0', 'p'], ['xf', 'yk'])

# ---------------------------------------------------------
# 4. Configuração do Otimizador MPC Não-Linear (IPOPT)
# ---------------------------------------------------------
N = 10 # Horizonte Preditivo

data = {
    'u_min': np.array([0]), # Normalizado (Corresponde a -100%)
    'u_max': np.array([1.0]),  # Normalizado (Corresponde a +100%)
    'u_guess': np.array([0.5]),
    'x_guess': np.zeros(nx)
}

def vcat(lst):
    return ca.vertcat(*[ca.DM(x_) if not hasattr(x_, 'is_symbolic') else x_ for x_ in lst])

w, lbw, ubw, w0 = [], [], [], []
g, lbg, ubg = [], [], []
J_cost = 0

xk_param = ca.MX.sym('xk_param', nx)
Pref = ca.MX.sym('Pref', N)
u_prev_param = ca.MX.sym('u_prev_param', 1)

xk = ca.MX.sym('x0', nx)
w.append(xk)
lbw.append(np.full(nx, -np.inf))
ubw.append(np.full(nx, np.inf))
w0.append(data['x_guess'])

g.append(xk - xk_param)
lbg.append(np.zeros(nx))
ubg.append(np.zeros(nx))

u_prev = u_prev_param 

for k in range(N):
    uk = ca.MX.sym(f'u_{k}', 1)
    w.append(uk)
    lbw.append(data['u_min'])
    ubw.append(data['u_max'])
    w0.append(data['u_guess'])
    
    Fk = F(x0=xk, p=uk)
    xnext = Fk['xf']
    yk = Fk['yk'] # Retorna em graus!
    
    du = uk - u_prev
    u_prev = uk
    
    # Penalidade: Erro de rastreamento (x1000), Esforço de controle (x0.1), Variação brusca de controle (x50)
    J_cost = J_cost + 1e3 * (yk - Pref[k])**2 + 0.1 * uk**2 + 50.0 * du**2
    
    xk = ca.MX.sym(f'x_{k+1}', nx)
    w.append(xk)
    lbw.append(np.full(nx, -np.inf))
    ubw.append(np.full(nx, np.inf))
    w0.append(data['x_guess'])
    
    g.append(xk - xnext)
    lbg.append(np.zeros(nx))
    ubg.append(np.zeros(nx))

w = ca.vertcat(*w)
lbw = vcat(lbw)
ubw = vcat(ubw)
w0 = vcat(w0)
g = ca.vertcat(*g)
lbg = vcat(lbg)
ubg = vcat(ubg)

nlp = {'x': w, 'g': g, 'f': J_cost, 'p': ca.vertcat(xk_param, Pref, u_prev_param)}
solver = nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})

# ---------------------------------------------------------
# 5. Geração de Sinal Rico e Coleta de Dados MPC Ótimo
# ---------------------------------------------------------
print("Gerando sinal de referência rico (multiseno + degraus)...")
np.random.seed(42)
SETPOINT = 45.0

ms_duration = 30.0
n_ms = int(round(ms_duration / Ts))
t_ms_seg = np.arange(n_ms) * Ts
freqs = np.arange(1.0/ms_duration, 0.25 + 1e-9, 1.0/ms_duration)
phases = np.random.uniform(0, 2 * np.pi, len(freqs))
ms = np.sum([np.sin(2*np.pi*f*t_ms_seg + ph) for f, ph in zip(freqs, phases)], axis=0)
ms = ms / np.max(np.abs(ms)) * 30.0 + SETPOINT

step_pieces = []
for _ in range(15):
    S = np.random.uniform(10.0, 80.0)
    dur = np.random.uniform(2.0, 6.0)
    step_pieces.append(np.full(int(round(dur / Ts)), S))

# Array de Referência Total em Graus
x2ref_train = np.concatenate([np.full(int(5.0/Ts), SETPOINT), ms, np.concatenate(step_pieces), np.full(int(5.0/Ts), SETPOINT)])

steps_train = len(x2ref_train)
sim_steps_train = steps_train - N

# Condição inicial: estabilizado no setpoint (radianos)
xsim_train = np.zeros((nx, 1))
xsim_train[0, 0] = SETPOINT * (np.pi / 180.0)

ysim_train = []
usim_train = []
dt_mpc = []
w0_val = np.zeros(w.shape[0])
u_prev_sim = 0.0

print(f"Simulando o MPC Exato (IPOPT) por {sim_steps_train} passos...")
for k in tqdm(range(sim_steps_train), desc="Simulação MPC"):
    ref_window = x2ref_train[k : k + N]
    
    # Parâmetros pro solver: Estado Contínuo (rad, rad/s), Referência em Graus, Ultimo Controle u_prev
    pval = np.concatenate([xsim_train[:, -1], ref_window, [u_prev_sim]])
    
    tic = time.perf_counter()
    sol = solver(x0=w0_val, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
    dt_mpc.append(time.perf_counter() - tic)
    
    w_opt = sol['x'].full().flatten()
    u_opt = w_opt[nx]
    
    # Ruído exploratório de 3% para enriquecer os dados para a ANN
    u_applied = np.random.normal(u_opt, 0.03) 
    u_applied = np.clip(u_applied, data['u_min'][0], data['u_max'][0])
    
    # Simula avanço físico na Planta (com u_applied ruidoso)
    sim_step = F(x0=xsim_train[:, -1], p=u_applied)
    xk1 = sim_step['xf'].full().flatten()
    yk = sim_step['yk'].full().item() # Ângulo em graus
    
    xsim_train = np.c_[xsim_train, xk1]
    usim_train.append(u_opt) # A rede vai imitar o U ótimo (limpo)
    ysim_train.append(yk)
    w0_val = w_opt
    u_prev_sim = u_opt

dt_mpc = np.array(dt_mpc)
print(f"Tempo médio do solver IPOPT: {dt_mpc.mean()*1000:.2f} ms por passo.")

# ---------------------------------------------------------
# 6. Clonagem de Comportamento (Treino da ANN do MPC)
# ---------------------------------------------------------
print("Preparando dados para a Clonagem de Comportamento (ANN)...")
# X_data: [estado_x (theta_rad, theta_dot), referencia_futura (N pontos), u_prev (1 ponto)]
X_data = []
u_prev_data = 0.0
for k in range(sim_steps_train):
    X_data.append(np.concatenate([xsim_train[:, k], x2ref_train[k : k + N], [u_prev_data]]))
    u_prev_data = usim_train[k]
X_data = np.array(X_data)

Y_data = np.array(usim_train).reshape(-1, 1)

scaler_ann = StandardScaler()
X_scaled = scaler_ann.fit_transform(X_data)

X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, Y_data, test_size=0.1, random_state=42)

class MPCApproximator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

ann_model = MPCApproximator(input_dim=nx + N + 1)
optimizer = optim.Adam(ann_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

epochs = 100
print("Treinando a ANN do MPC...")
for epoch in range(epochs):
    ann_model.train()
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(ann_model(b_x), b_y)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

ann_model.eval()
with torch.no_grad():
    y_te_pred = ann_model(torch.tensor(X_te, dtype=torch.float32)).numpy()
r2 = r2_score(y_te, y_te_pred)
print(f"R2-Score da ANN: {r2:.4f}")

# ---------------------------------------------------------
# 7. Plots Finais (Simulação do MPC Ótimo)
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))
tvec = np.arange(sim_steps_train) * Ts
plt.plot(tvec, ysim_train, label='Mundo Físico Contínuo (Simulado)')
plt.step(tvec, x2ref_train[:sim_steps_train], 'k--', where='post', label='Referência')
plt.title('Rastreamento: MPC Não-Linear Ótimo (Neural ODE Híbrido)')
plt.xlabel('Tempo (s)'); plt.ylabel('Ângulo (deg)')
plt.legend(); plt.grid(); plt.show()

plt.figure(figsize=(12, 3))
plt.step(tvec, np.array(usim_train) * 100.0, 'r', where='post', label='Comando de Aceleração')
plt.xlabel('Tempo (s)'); plt.ylabel('PWM do Motor (%)')
plt.legend(); plt.grid(); plt.show()

# Salva a ANN
torch.save(ann_model.state_dict(), "ann_mpc_node.pth")
print("Rede Neural do MPC salva com sucesso em 'ann_mpc_node.pth'!")
