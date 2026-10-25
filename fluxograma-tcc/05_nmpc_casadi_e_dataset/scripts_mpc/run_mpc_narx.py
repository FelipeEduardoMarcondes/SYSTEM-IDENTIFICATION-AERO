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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



# ---------------------------------------------------------
# 1. Definições Iniciais
# ---------------------------------------------------------
Ts = 0.05
ny = 30
nu = 50
nx = ny + nu

# Escalonadores globais (treinados com limites absolutos físicos)
scaler_u = MinMaxScaler(feature_range=(-1, 1))
scaler_u.fit(np.array([[-100], [100]])) # Limite físico do motor (reverso e direto)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_y.fit(np.array([[-180], [180]]))

# ---------------------------------------------------------
# 2. Carregar o Modelo PyTorch NARX v4
# ---------------------------------------------------------
class NARXModel(nn.Module):
    def __init__(self, ny, nu, hidden_dim1=128, hidden_dim2=64, dropout=0.1):
        super().__init__()
        self.ny = ny
        self.nu = nu
        self.linear_bypass = nn.Linear(ny + nu, 1)
        self.net = nn.Sequential(
            nn.Linear(ny + nu, hidden_dim1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward(self, x):
        return self.linear_bypass(x) + self.net(x)

# Carrega os pesos salvos do treinamento
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "modelos_salvos", "best_narx.pth")
pytorch_model = NARXModel(ny, nu, 256, 128)
if os.path.exists(model_path):
    pytorch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Modelo carregado de {model_path}")
else:
    print(f"AVISO: {model_path} não encontrado! Usando pesos aleatórios para demonstração.")
pytorch_model.eval()

# ---------------------------------------------------------
# 3. Transcrição Simbólica para CasADi
# ---------------------------------------------------------
# Extrai as matrizes de pesos do PyTorch para Numpy
W_bypass = pytorch_model.linear_bypass.weight.detach().numpy()
b_bypass = pytorch_model.linear_bypass.bias.detach().numpy()

W0 = pytorch_model.net[0].weight.detach().numpy()
b0 = pytorch_model.net[0].bias.detach().numpy()

W1 = pytorch_model.net[3].weight.detach().numpy()
b1 = pytorch_model.net[3].bias.detach().numpy()

W2 = pytorch_model.net[6].weight.detach().numpy()
b2 = pytorch_model.net[6].bias.detach().numpy()

# Variável de estado Simbólica (Tudo normalizado [-1, 1])
x_sym = ca.MX.sym('x', nx) 
u_in_sym = ca.MX.sym('u', 1) 

# Concatenação da entrada do NARX (y_past e u_past_com_atual)
# O estado x guarda [y(k-ny) ... y(k-1), u(k-nu) ... u(k-2)]
# Para o modelo predizer y(k), ele recebe [y_past, u_past_deslocado + u_atual]
x_narx_input = ca.vertcat(x_sym[0:ny], x_sym[ny+1:nx], u_in_sym)

# Forward Pass da Rede Neural simbólica
y_bypass_sym = ca.mtimes(W_bypass, x_narx_input) + b_bypass
h1_sym = ca.tanh(ca.mtimes(W0, x_narx_input) + b0)
h2_sym = ca.tanh(ca.mtimes(W1, h1_sym) + b1)
y_mlp_sym = ca.mtimes(W2, h2_sym) + b2

y_pred_sym = y_bypass_sym + y_mlp_sym

# Atualização de Estado (Deslocamento FIFO)
# y_next: descarta o mais antigo x[0], adiciona y_pred
y_next = ca.vertcat(x_sym[1:ny], y_pred_sym)
# u_next: descarta o mais antigo x[ny], adiciona u_in
u_next = ca.vertcat(x_sym[ny+1:nx], u_in_sym)

x_next = ca.vertcat(y_next, u_next)

# A Planta Virtual para o MPC
F = ca.Function('F', [x_sym, u_in_sym], [x_next, y_pred_sym], ['x0', 'p'], ['xf', 'yk'])

# ---------------------------------------------------------
# 4. Configuração do Otimizador MPC Não-Linear (IPOPT)
# ---------------------------------------------------------
N = 10 # Horizonte Preditivo

data = {
    'u_min': np.array([-1.0]), # Normalizado (Corresponde a -100%)
    'u_max': np.array([1.0]),  # Normalizado (Corresponde a +100%)
    'u_guess': np.array([0.0]),
    'x_guess': np.zeros(nx)
}

def vcat(lst):
    return ca.vertcat(*[ca.DM(x_) if not hasattr(x_, 'is_symbolic') else x_ for x_ in lst])

w, lbw, ubw, w0 = [], [], [], []
g, lbg, ubg = [], [], []
J = 0

xk_param = ca.MX.sym('xk_param', nx)
Pref = ca.MX.sym('Pref', N)

xk = ca.MX.sym('x0', nx)
w.append(xk)
lbw.append(np.full(nx, -np.inf))
ubw.append(np.full(nx, np.inf))
w0.append(data['x_guess'])

g.append(xk - xk_param)
lbg.append(np.zeros(nx))
ubg.append(np.zeros(nx))

u_prev = xk[nx-1] # Ultimo controle aplicado

for k in range(N):
    uk = ca.MX.sym(f'u_{k}', 1)
    w.append(uk)
    lbw.append(data['u_min'])
    ubw.append(data['u_max'])
    w0.append(data['u_guess'])
    
    Fk = F(x0=xk, p=uk)
    xnext = Fk['xf']
    yk = Fk['yk']
    
    du = uk - u_prev
    u_prev = uk
    
    # Penalidade: Erro de rastreamento (x1000), Esforço de controle (x0.1), Variação de controle (x50)
    J = J + 1e3 * (yk - Pref[k])**2 + 0.1 * uk**2 + 50.0 * du**2
    
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

nlp = {'x': w, 'g': g, 'f': J, 'p': ca.vertcat(xk_param, Pref)}
# Oculta prints do IPOPT para velocidade máxima
solver = nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})

# ---------------------------------------------------------
# 5. Geração de Sinal Rico e Coleta de Dados MPC Ótimo
# ---------------------------------------------------------
print("Gerando sinal de referência rico (multiseno + degraus)...")
np.random.seed(42)
SETPOINT = 45.0
SETPOINT_NORM = scaler_y.transform(np.array([[SETPOINT]]))[0,0]

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

x2ref_train = np.concatenate([np.full(int(5.0/Ts), SETPOINT), ms, np.concatenate(step_pieces), np.full(int(5.0/Ts), SETPOINT)])
# Normaliza o sinal de referência para bater com a rede
x2ref_norm = scaler_y.transform(x2ref_train.reshape(-1, 1)).flatten()

steps_train = len(x2ref_norm)
sim_steps_train = steps_train - N

xsim_train = np.full((nx, 1), SETPOINT_NORM) # Estado inicial: estabilizado no setpoint
ysim_train = []
usim_train = []
dt_mpc = []
w0_val = np.zeros(w.shape[0])

print(f"Simulando o MPC Exato (IPOPT) por {sim_steps_train} passos...")
for k in tqdm(range(sim_steps_train), desc="Simulação MPC"):
    ref_window = x2ref_norm[k : k + N]
    pval = np.concatenate([xsim_train[:, -1], ref_window])
    
    tic = time.perf_counter()
    sol = solver(x0=w0_val, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pval)
    dt_mpc.append(time.perf_counter() - tic)
    
    w_opt = sol['x'].full().flatten()
    u_opt = w_opt[nx]
    
    # Ruído exploratório de 3% para enriquecer os dados para a ANN
    u_applied = np.random.normal(u_opt, 0.03) 
    u_applied = np.clip(u_applied, data['u_min'][0], data['u_max'][0])
    
    # Simula avanço físico na Planta
    sim_step = F(x0=xsim_train[:, -1], p=u_applied)
    xk1 = sim_step['xf'].full().flatten()
    yk = sim_step['yk'].full().item()
    
    xsim_train = np.c_[xsim_train, xk1]
    usim_train.append(u_opt) # Treinaremos a rede para imitar o u ótimo limpo!
    ysim_train.append(yk)
    w0_val = w_opt

dt_mpc = np.array(dt_mpc)
print(f"Tempo médio do solver IPOPT: {dt_mpc.mean()*1000:.2f} ms por passo.")

# ---------------------------------------------------------
# 6. Clonagem de Comportamento (Treino da ANN do MPC)
# ---------------------------------------------------------
print("Preparando dados para a Clonagem de Comportamento (ANN)...")
# X_data: [estado_x (tamanho nx), referencia_futura (tamanho N)]
X_data = np.array([np.concatenate([xsim_train[:, k], x2ref_norm[k : k + N]]) for k in range(sim_steps_train)])
# Y_data: u_otimo (tamanho 1)
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

ann_model = MPCApproximator(input_dim=nx + N)
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
# Desfaz normalização para os plots
ysim_train_orig = scaler_y.inverse_transform(np.array(ysim_train).reshape(-1,1)).flatten()
usim_train_orig = scaler_u.inverse_transform(np.array(usim_train).reshape(-1,1)).flatten()

plt.figure(figsize=(12, 5))
tvec = np.arange(sim_steps_train) * Ts
plt.plot(tvec, ysim_train_orig, label='Mundo Físico (Simulado)')
plt.step(tvec, x2ref_train[:sim_steps_train], 'k--', where='post', label='Referência')
plt.title('Rastreamento de Trajetória: MPC Não-Linear Ótimo')
plt.xlabel('Tempo (s)'); plt.ylabel('Ângulo (deg)')
plt.legend(); plt.grid(); plt.show()

plt.figure(figsize=(12, 3))
plt.step(tvec, usim_train_orig, 'r', where='post', label='Comando de Aceleração')
plt.xlabel('Tempo (s)'); plt.ylabel('PWM do Motor (%)')
plt.legend(); plt.grid(); plt.show()

# Salva a ANN
torch.save(ann_model.state_dict(), "ann_mpc.pth")
print("Rede Neural do MPC salva com sucesso em 'ann_mpc.pth'!")
