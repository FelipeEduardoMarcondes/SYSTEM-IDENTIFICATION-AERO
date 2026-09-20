# %% [markdown]
# # Identificação do Aeropêndulo com Neural ODEs - Multi-Experimentos (v3.0)
#
# Ajustes desta versão em relação à v2.0:
# - **Anti-aliasing na decimação**: `scipy.signal.decimate` (filtro IIR + zero-phase)
#   em vez de slicing puro, tanto em `u` quanto em `y`.
# - **Curriculum learning no rollout**: `k_steps` começa curto e cresce ao longo do
#   treino, forçando o modelo a acertar também a dinâmica de longo prazo.
# - **Loss normalizada por estado**: posição e velocidade são normalizadas pelo
#   próprio desvio-padrão antes do MSE, evitando que a velocidade (escala maior)
#   domine o gradiente.
# - **LR scheduling (cosine annealing)**: reduz a oscilação da loss entre épocas.
# - **Integrador adaptativo (`dopri5`) na validação**: menos erro numérico
#   acumulado em rollouts longos (free-run) do que RK4 de passo fixo.
# - **Múltiplos arquivos de teste**: separa "problema de alta frequência/aliasing"
#   de "problema de generalização geral".

# %%
from pickle import FALSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from scipy.signal import savgol_filter, decimate
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(0)
np.random.seed(0)

# %%
# Configuração: Escolha quais modelos treinar
TREINAR_CAIXA_CINZA = True
TREINAR_CAIXA_PRETA = False

# %%
def carregar_experimento(url, decimacao=1):
    df = pd.read_csv(url)
    if 'referencia' in df.columns:
        df = df[df['referencia'] > 0]

    if 'u_pct' in df.columns:
        u_full = df['u_pct'].values.astype(np.float64)
    else:
        u_full = df['motor_percent'].values.astype(np.float64)
    y_full = df['angulo_deg'].values.astype(np.float64)

    if decimacao > 1:
        # decimate() aplica um filtro passa-baixas (Chebyshev tipo I, IIR) ANTES
        # de subamostrar -> evita que conteúdo acima da nova frequência de
        # Nyquist "dobre" (aliasing) para dentro da banda de interesse.
        # zero_phase=True usa filtfilt, então não desloca o sinal no tempo.
        # OBS: como qualquer filtro passa-baixas, isso suaviza um pouco as
        # bordas de degraus abruptos em u (seq-degraus) -- é o trade-off
        # inerente a decimar corretamente; se isso distorcer demais o
        # comando em datasets de degrau, considere decimação simples (sem
        # filtro) apenas para u, mantendo o filtro em y.
        u_raw = decimate(u_full, decimacao, ftype='iir', zero_phase=True)
        y_raw = decimate(y_full, decimacao, ftype='iir', zero_phase=True)
    else:
        u_raw = u_full
        y_raw = y_full

    dt = 0.010 * decimacao
    t_raw = np.arange(len(y_raw)) * dt

    return t_raw, u_raw, y_raw

def processar_dataset(t_raw, u_raw, y_raw):
    # Forçar o tempo inicial para zero
    t_raw = t_raw - t_raw[0]

    # Conversão para radianos (mundo físico)
    y_rad = y_raw * (np.pi / 180.0)

    # Normalização da entrada u (0 a 100% -> 0 a 1). O filtro de decimate()
    # pode gerar leve overshoot/undershoot perto de bordas abruptas -> clip
    # para manter u dentro da faixa fisicamente válida do motor.
    u_norm = np.clip(u_raw / 100.0, 0.0, 1.0)

    # Derivação com Savitzky-Golay
    dt_mean = np.mean(np.diff(t_raw))
    window = 11
    poly = 3
    v_rad_s = savgol_filter(y_rad, window, poly, deriv=1, delta=dt_mean)

    # Matriz de estados x = [posição, velocidade]
    x_matrix = np.vstack((y_rad, v_rad_s)).T

    return (torch.tensor(t_raw, dtype=torch.float32),
            torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1),
            torch.tensor(x_matrix, dtype=torch.float32),
            y_rad, v_rad_s, u_norm)

BASE2 = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"

# Arquivos utilizados para treino (variações ricas de dinâmica)
train_files = [
    "RODADA-2/multi-seno-2_0804_19-31.csv",
    "RODADA-2/multi-seno-1_0804_19-06.csv",
    "RODADA-2/seq-degraus-2_0804_19-38.csv",
    "RODADA-2/multi-seno-3_0804_19-44.csv",
    "RODADA-2/seq-degraus-1_0804_19-12.csv",
    "multi-seno-1_0807_16-57.csv",
    "multi-seno-2_0807_17-13.csv",
    "seq-degraus-1_0807_16-38.csv",
    "seq-degraus-aprbs-2_0807_17-23.csv",
    "seq-dragus-aprbs_0807_16-50.csv"

]

# Vários arquivos de teste, nunca vistos no treino: um chirp (alta frequência
# / mais sensível a aliasing), um multi-seno e uma sequência de degraus
# (para checar se a generalização falha em geral ou só em alta frequência).
test_files = [
    "RODADA-2/chirp-1_0804_19-19.csv",
    "RODADA-2/multi-seno-2_0804_19-28.csv",
    "RODADA-2/seq-degraus-1_0804_19-09.csv",
]

decimacao = 2

print("Carregando datasets de TREINO...")
train_datasets = []
for f in train_files:
    t_raw, u_raw, y_raw = carregar_experimento(BASE2 + f, decimacao=decimacao)
    t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
    train_datasets.append({
        'name': f,
        't': t_ten, 'u': u_ten, 'x': x_ten,
        'y_rad': y_rad, 'v_rad': v_rad, 'u_norm': u_norm
    })
    print(f" -> {f} carregado com {len(t_ten)} amostras.")

print("\nCarregando datasets de TESTE (múltiplos, nunca vistos no treino)...")
test_datasets = []
for f in test_files:
    t_raw, u_raw, y_raw = carregar_experimento(BASE2 + f, decimacao=decimacao)
    t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
    test_datasets.append({
        'name': f,
        't': t_ten, 'u': u_ten, 'x': x_ten,
        'y_rad': y_rad, 'v_rad': v_rad, 'u_norm': u_norm
    })
    print(f" -> {f} carregado com {len(t_ten)} amostras.")

# Visualizando um dos datasets de treino
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(train_datasets[0]['t'], train_datasets[0]['y_rad'], label='Posição (rad)')
plt.plot(train_datasets[0]['t'], train_datasets[0]['v_rad'], label='Velocidade (rad/s)', alpha=0.7)
plt.title(f"Exemplo de Treino: {train_datasets[0]['name']}")
plt.grid(); plt.legend()

plt.subplot(2, 1, 2)
plt.plot(train_datasets[0]['t'], train_datasets[0]['u_norm'], label='Comando u (Norm)', color='green')
plt.grid(); plt.legend()
plt.tight_layout(); plt.show()

# %%
# Desvio-padrão de cada estado (calculado sobre TODO o conjunto de treino),
# usado para normalizar o loss e evitar que a velocidade (escala maior)
# domine o gradiente em detrimento da posição.
all_pos = np.concatenate([d['y_rad'] for d in train_datasets])
all_vel = np.concatenate([d['v_rad'] for d in train_datasets])
std_pos = float(np.std(all_pos))
std_vel = float(np.std(all_vel))
state_std = torch.tensor([std_pos, std_vel], dtype=torch.float32, device=device)
print(f"Std posição: {std_pos:.4f} rad | Std velocidade: {std_vel:.4f} rad/s")

# %%
# 1. MODELOS ODE (NODE)
class PhysicsODE(nn.Module):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0):
        super().__init__()
        self.m1, self.L1 = 0.122, 0.39
        self.m2, self.L2 = 0.055, 0.347
        self.g = 9.81

        # Parâmetros Desconhecidos: Inércia (J), Atrito (b), Ganho do Motor (Gu)
        # Valores iniciais configuráveis -- por padrão reproduzem a v2.0
        # (J0=1, b0=exp(-1), Gu0=1). Gu0/b0 podem vir do pré-treino físico
        # (ver `pretreinar_fisico_estatico`), que os ajusta via gradiente
        # (odeint + backprop, igual ao treino principal) usando curva(s)
        # quase-estática(s), sem depender de J (pouco identificável nesse
        # regime, já que a aceleração é baixa o tempo todo).
        self.log_J = nn.Parameter(torch.log(torch.tensor(float(J0))))
        self.log_b = nn.Parameter(torch.log(torch.tensor(float(b0))))
        self.log_Gu = nn.Parameter(torch.log(torch.tensor(float(Gu0))))

        self.u_series = None
        self.t_series = None
        self.batch_start_times = None

    def get_params(self):
        return torch.exp(self.log_J), torch.exp(self.log_b), torch.exp(self.log_Gu)

    def forward(self, t, x):
        J, b, Gu = self.get_params()

        p1 = b / J
        p2 = ((self.m1 * self.L1 - self.m2 * self.L2) * self.g) / J
        p3 = Gu / J

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
        u_t = u1 + alpha * (u2 - u1)

        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        theta_ddot = - p1 * theta_dot - p2 * torch.sin(theta) + p3 * (u_t ** 2)

        return torch.cat([theta_dot, theta_ddot], dim=1)


class BlackBoxODE(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        self.u_series = None
        self.t_series = None
        self.batch_start_times = None

    def forward(self, t, x):
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
        u_t = u1 + alpha * (u2 - u1)

        nn_input = torch.cat([x, u_t], dim=1)
        return self.net(nn_input)

# %%
# 2. FUNÇÃO DE TREINAMENTO (MULTI-DATASET, COM CURRICULUM + LOSS NORMALIZADA + LR SCHEDULE)
def train_model_multi(model, name, datasets, epochs=1500, lr=0.02,
                       k_min=20, k_max=300, curriculum_stage_epochs=400,
                       state_std=None, base_batch_size=1024, integrator='rk4'):
    print(f"--- Iniciando Treinamento: {name} ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if state_std is None:
        state_std = torch.ones(2, device=device)

    # Assume-se dt igual para todos os datasets (pois vieram da mesma bancada e dízima)
    dt = (datasets[0]['t'][1] - datasets[0]['t'][0]).item()

    loss_history = []

    for epoch in range(epochs + 1):
        # --- Curriculum learning: k_steps dobra a cada `curriculum_stage_epochs`,
        # começando em k_min e saturando em k_max. Isso dá convergência rápida
        # e estável no início (janelas curtas) e depois força o modelo a acertar
        # a dinâmica de longo prazo (janelas longas), reduzindo o desalinhamento
        # de fase que aparece em rollouts free-run muito além do horizonte visto
        # no treino.
        stage = epoch // curriculum_stage_epochs
        k_steps = int(min(k_min * (2 ** stage), k_max))
        t_eval = torch.arange(0, k_steps * dt, dt, device=device)[:k_steps]

        # Reduz o batch conforme a janela cresce, para manter custo de
        # memória/tempo do rollout (odeint sobre o batch inteiro) sob controle.
        batch_size = max(64, int(base_batch_size * (k_min / k_steps)))

        optimizer.zero_grad()

        # Sorteia qual dataset será usado para este batch
        ds = np.random.choice(datasets)
        t_ds = ds['t'].to(device)
        u_ds = ds['u'].to(device)
        x_ds = ds['x'].to(device)

        # Alimenta os dados temporais deste dataset ao modelo
        model.t_series = t_ds
        model.u_series = u_ds

        # Seleciona pontos de partida aleatórios
        start_idx = np.random.randint(0, len(t_ds) - k_steps, size=batch_size)
        x0 = x_ds[start_idx]
        model.batch_start_times = t_ds[start_idx].reshape(-1, 1)

        # Integração (Rollout)
        pred_state = odeint(model, x0, t_eval, method=integrator)

        # Compara com o alvo
        batch_targets = [x_ds[i:i + k_steps] for i in start_idx]
        y_target = torch.stack(batch_targets, dim=1)

        # Loss normalizada por estado: posição e velocidade contribuem em
        # escalas comparáveis (cada uma dividida pelo seu próprio desvio-padrão).
        loss = torch.mean(((pred_state - y_target) / state_std) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if epoch % 100 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d} | k_steps={k_steps:3d} | batch={batch_size:4d} "
                  f"| LR={lr_now:.5f} | Loss: {loss.item():.6f}")

    return model, loss_history

# %% [markdown]
# ## Pré-treino físico: ajustando Gu e b via gradiente nos dados quase-estáticos
#
# Em vez de uma fórmula fechada (equação linearizada + limiar de aceleração),
# este pré-treino usa exatamente o mesmo mecanismo do treino principal --
# `odeint` integrando a ODE não-linear completa + backprop -- só que restrito
# ao(s) arquivo(s) quase-estático(s). Isso tem duas vantagens sobre a versão
# anterior:
# - Não precisa linearizar nada nem escolher limiar de aceleração: a ODE real
#   já lida corretamente com os trechos "semi" (velocidade baixa mas não nula).
# - Usa TODOS os dados que você passar (em janelas de rollout), não só uma
#   fração filtrada.
#
# `J` fica congelado por padrão (`treinar_J=False`): numa curva quase-estática
# a aceleração é baixa o tempo todo, então há pouco sinal de gradiente pra
# identificar inércia -- deixá-la livre aqui tende a ficar mal condicionado.
# Os valores retornados (J0, b0, Gu0) são usados só como INICIALIZAÇÃO do
# `PhysicsODE` para o treino dinâmico principal, que continua livre para
# reajustar tudo (inclusive J) usando os dados multi-experimento.

# %%
def pretreinar_fisico_estatico(caminhos_arquivos, decimacao=5, epochs=800, lr=0.01,
                                k_steps=60, batch_size=256, treinar_J=False,
                                J0=1.0, b0=np.exp(-1.0), Gu0=1.0):
    datasets = []
    for path in caminhos_arquivos:
        t_raw, u_raw, y_raw = carregar_experimento(path, decimacao=decimacao)
        t_ten, u_ten, x_ten, y_rad, v_rad, u_norm = processar_dataset(t_raw, u_raw, y_raw)
        datasets.append({
            'name': path.split('/')[-1],
            't': t_ten, 'u': u_ten, 'x': x_ten,
            'y_rad': y_rad, 'v_rad': v_rad, 'u_norm': u_norm
        })
        print(f" -> {path.split('/')[-1]} carregado com {len(t_ten)} amostras.")

    model = PhysicsODE(J0=J0, b0=b0, Gu0=Gu0).to(device)
    model.log_J.requires_grad_(treinar_J)

    # Normalização do loss (mesmo raciocínio do treino principal): posição e
    # velocidade em escalas comparáveis, calculada só sobre estes dados.
    all_pos = np.concatenate([d['y_rad'] for d in datasets])
    all_vel = np.concatenate([d['v_rad'] for d in datasets])
    local_std = torch.tensor([np.std(all_pos), np.std(all_vel)], dtype=torch.float32, device=device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dt = (datasets[0]['t'][1] - datasets[0]['t'][0]).item()
    t_eval = torch.arange(0, k_steps * dt, dt, device=device)[:k_steps]

    print(f"\n--- Pré-treino físico (Gu, b{', J' if treinar_J else ''}) "
          f"nos dados quase-estáticos ---")
    for epoch in range(epochs + 1):
        optimizer.zero_grad()

        ds = np.random.choice(datasets)
        t_ds = ds['t'].to(device)
        u_ds = ds['u'].to(device)
        x_ds = ds['x'].to(device)

        model.t_series = t_ds
        model.u_series = u_ds

        bsz = min(batch_size, len(t_ds) - k_steps)
        start_idx = np.random.randint(0, len(t_ds) - k_steps, size=bsz)
        x0 = x_ds[start_idx]
        model.batch_start_times = t_ds[start_idx].reshape(-1, 1)

        pred_state = odeint(model, x0, t_eval, method='rk4')
        y_target = torch.stack([x_ds[i:i + k_steps] for i in start_idx], dim=1)

        loss = torch.mean(((pred_state - y_target) / local_std) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            J_now, b_now, Gu_now = model.get_params()
            print(f"Epoch {epoch:4d} | Loss {loss.item():.6f} | "
                  f"J={J_now.item():.4f} b={b_now.item():.4f} Gu={Gu_now.item():.4f}")

    J_f, b_f, Gu_f = model.get_params()
    print(f"\nParâmetros pré-treinados: J={J_f.item():.4f}  b={b_f.item():.4f}  Gu={Gu_f.item():.4f}")
    return J_f.item(), b_f.item(), Gu_f.item()

# %%
# Arquivos de curva quase-estática -- coloque os CSVs na mesma pasta do
# script (ou ajuste os caminhos abaixo). Eles NÃO entram em `train_files`:
# servem só para pré-treinar Gu e b antes do treino dinâmico principal.
caminhos_curva_estatica = [
    "experimentos/dados_curva_semi_estatica_0807_17-36.csv",
    "experimentos/dados_curva_semi_estatica-2_0807_17-40.csv",
]

if TREINAR_CAIXA_CINZA:
    J0_pre, b0_pre, Gu0_pre = pretreinar_fisico_estatico(
        caminhos_curva_estatica, decimacao=decimacao,
        epochs=2000, lr=0.01, k_steps=200, batch_size=256, treinar_J=False
    )

    # %%
    # Treinando o modelo Físico -- inicializado com J0/b0/Gu0 do pré-treino
    # nos dados quase-estáticos (J0_pre é só o default, pois treinar_J=False
    # acima manteve J congelado no pré-treino)
    phys_model = PhysicsODE(J0=J0_pre, b0=b0_pre, Gu0=Gu0_pre)
    phys_model, phys_loss_hist = train_model_multi(
        phys_model, "Physics-Informed Model (Caixa-Cinza)", train_datasets,
        epochs=5000, lr=0.025, k_min=20, k_max=1000, curriculum_stage_epochs=500,
        state_std=state_std
    )

    J, b, Gu = phys_model.get_params()
    print("\nParâmetros Físicos Encontrados:")
    print(f"J (Inércia): {J.item():.4f} kg.m^2")
    print(f"b (Atrito): {b.item():.4f}")
    print(f"Gu (Ganho Motor): {Gu.item():.4f}\n")

    # --- Salvando o modelo Físico (Caixa-Cinza) na máquina logo após o treino ---
    caminho_cinza = 'node_caixa_cinza.pth'
    torch.save(phys_model.state_dict(), caminho_cinza)
    print(f'Modelo Físico (Caixa-Cinza) salvo em: {caminho_cinza}\n')

if TREINAR_CAIXA_PRETA:
    # Treinando o modelo Caixa-Preta (MLP)
    bb_model = BlackBoxODE(hidden_dim=32)
    bb_model, bb_loss_hist = train_model_multi(
        bb_model, "Black-Box Model (MLP)", train_datasets,
        epochs=2500, lr=0.02, k_min=20, k_max=300, curriculum_stage_epochs=400,
        state_std=state_std
    )

    # --- Salvando o modelo Neural (Caixa-Preta) na máquina logo após o treino ---
    caminho_preta = 'node_caixa_preta.pth'
    torch.save(bb_model.state_dict(), caminho_preta)
    print(f'Modelo Neural (Caixa-Preta) salvo em: {caminho_preta}\n')

# %%
# Histórico de loss (escala log) -- útil para conferir se o LR schedule
# de fato reduziu a oscilação entre épocas em relação à v2.0.
plt.figure(figsize=(10, 4))
if TREINAR_CAIXA_CINZA:
    plt.plot(phys_loss_hist, label='Physics-Informed', alpha=0.8)
if TREINAR_CAIXA_PRETA:
    plt.plot(bb_loss_hist, label='Black-Box', alpha=0.8)
plt.yscale('log')
plt.xlabel("Época"); plt.ylabel("Loss normalizada (log)")
plt.title("Histórico de Loss")
plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()

# %%
# 3. VALIDAÇÃO FREE-RUN — MÚLTIPLOS ARQUIVOS DE TESTE, INTEGRADOR ADAPTATIVO
print("\n--- Simulação Free-Run em cada conjunto de TESTE ---")

results = []
n_test = len(test_datasets)
# Criando 1 coluna e n_test linhas (um gráfico por linha)
fig, axes = plt.subplots(n_test, 1, figsize=(12, 5 * n_test), squeeze=False)

for i, ds in enumerate(test_datasets):
    t_t = ds['t'].to(device)
    u_t = ds['u'].to(device)
    x_t = ds['x'].to(device)
    
    y_real_deg = x_t[:, 0].cpu().numpy() * (180.0 / np.pi)
    t_np = t_t.cpu().numpy()
    
    rmse_phys = None
    rmse_bb = None

    with torch.no_grad():
        x0 = x_t[0].unsqueeze(0)
        
        if TREINAR_CAIXA_CINZA:
            phys_model.u_series = u_t
            phys_model.t_series = t_t
            phys_model.batch_start_times = torch.zeros(1, 1, device=device)

            pred_phys = odeint(phys_model, x0, t_t, method='dopri5',
                                rtol=1e-5, atol=1e-6).squeeze(1).cpu().numpy()
            phys_deg = pred_phys[:, 0] * (180.0 / np.pi)
            rmse_phys = np.sqrt(mean_squared_error(y_real_deg, phys_deg))
            
        if TREINAR_CAIXA_PRETA:
            bb_model.u_series = u_t
            bb_model.t_series = t_t
            bb_model.batch_start_times = torch.zeros(1, 1, device=device)

            pred_bb = odeint(bb_model, x0, t_t, method='dopri5',
                              rtol=1e-5, atol=1e-6).squeeze(1).cpu().numpy()
            bb_deg = pred_bb[:, 0] * (180.0 / np.pi)
            rmse_bb = np.sqrt(mean_squared_error(y_real_deg, bb_deg))

    results.append({'arquivo': ds['name'], 'rmse_phys': rmse_phys, 'rmse_bb': rmse_bb})

    msg = f"{ds['name']:45s} | "
    if TREINAR_CAIXA_CINZA: msg += f"Physics: {rmse_phys:6.2f}°  "
    if TREINAR_CAIXA_PRETA: msg += f"| Black-Box: {rmse_bb:6.2f}°"
    print(msg)

    ax = axes[i, 0]
    ax.plot(t_np, y_real_deg, 'k-', linewidth=2, label='Real')
    
    title = f"{ds['name']}\n"
    if TREINAR_CAIXA_CINZA:
        ax.plot(t_np, phys_deg, 'r--', linewidth=2, label='Physics-Informed')
        title += f"Physics (RMSE {rmse_phys:.2f}°)  "
    if TREINAR_CAIXA_PRETA:
        ax.plot(t_np, bb_deg, 'b--', linewidth=2, label='Black-Box')
        title += f"Black-Box (RMSE {rmse_bb:.2f}°)"
        
    ax.set_title(title.strip())
    ax.set_xlabel("Tempo (s)"); ax.set_ylabel("Ângulo (graus)")
    ax.legend(); ax.grid(True)

plt.tight_layout()
plt.show()

print("\nResumo comparativo:")
for r in results:
    msg = f"  {r['arquivo']:45s} "
    if TREINAR_CAIXA_CINZA: msg += f"Physics={r['rmse_phys']:6.2f}°  "
    if TREINAR_CAIXA_PRETA: msg += f"BlackBox={r['rmse_bb']:6.2f}°"
    print(msg)

# %%
# Fim do script