# %% [markdown]
# # Identificação do Aeropêndulo com Neural ODEs (v5.0 - Multiple Shooting)
#
# Ajustes desta versão:
# - Abordagem "Multiple Shooting" verdadeira.
# - Divisão dos datasets de treino em blocos fixos (chunks).
# - Condições iniciais otimizáveis (X0_hat) para cada bloco.
# - Função de perda com termo de dados + termo de continuidade.
#
# Otimizações de performance (v5.1):
# - odeint chamado UMA VEZ por época com batch completo de chunks [B, 2].
# - Todos os tensors pré-transferidos para GPU antes do loop de treino.
# - t_eval pré-calculado, sem realocação por chunk/epoch.
# - Validação a cada 25 épocas (reduzido de 10).
#
# Dataset split (state-of-the-art para SysID):
# - TREINO : Rodada 4 (27/08) + raiz — excitação APRBS, multi-seno, swept-sine, seq-degraus
#            com maior variedade de frequências e amplitudes. Rodadas recentes são
#            priorizadas para capturar o estado atual do sistema.
# - VALIDAÇÃO: Rodada 2 (04/08) + Rodada 3 (07/08) — chirps e seq-degraus de
#              sessões completamente diferentes, garantindo generalização temporal
#              (cross-session validation), prática recomendada em benchmark de SysID.
# Referência: Ljung (1999), Schön et al. (2011), Kidger et al. (2021).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from scipy.signal import savgol_filter, decimate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(0)
np.random.seed(0)

TREINAR_BASELINE    = False
TREINAR_ASSIMETRICO = True
TREINAR_HIBRIDO     = True

# ---------------------------------------------------------------------------
# CARREGAMENTO E PRE-PROCESSAMENTO
# ---------------------------------------------------------------------------
def carregar_experimento(url, decimacao=1, start_idx=None, end_idx=None):
    df = pd.read_csv(url)
    if 'referencia' in df.columns:
        df = df[df['referencia'] > 0]

    if 'u_pct' in df.columns:
        u_full = df['u_pct'].values.astype(np.float64)
    else:
        u_full = df['motor_percent'].values.astype(np.float64)
    y_full = df['angulo_deg'].values.astype(np.float64)

    if decimacao > 1:
        y_raw = decimate(y_full, decimacao, ftype='iir', zero_phase=True)
        u_raw = u_full[::decimacao]
    else:
        u_raw = u_full
        y_raw = y_full

    # dt nominal fixo (sem flutuacao numerica entre datasets)
    dt = round(0.010 * decimacao, 6)
    t_raw = np.arange(len(y_raw)) * dt

    min_len = min(len(y_raw), len(u_raw))
    t_raw, u_raw, y_raw = t_raw[:min_len], u_raw[:min_len], y_raw[:min_len]

    if start_idx is not None and end_idx is not None:
        t_raw = t_raw[start_idx:end_idx]
        u_raw = u_raw[start_idx:end_idx]
        y_raw = y_raw[start_idx:end_idx]

    return t_raw, u_raw, y_raw

def processar_dataset(t_raw, u_raw, y_raw):
    t_raw = t_raw - t_raw[0]
    y_rad = y_raw * (np.pi / 180.0)
    u_norm = np.clip(u_raw / 100.0, -1.0, 1.0)

    dt_mean = np.mean(np.diff(t_raw))
    v_rad_s = savgol_filter(y_rad, 11, 3, deriv=1, delta=dt_mean)
    x_matrix = np.vstack((y_rad, v_rad_s)).T

    return (torch.tensor(t_raw, dtype=torch.float32),
            torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1),
            torch.tensor(x_matrix, dtype=torch.float32),
            y_rad, v_rad_s, u_norm)

# ---------------------------------------------------------------------------
# 1. MODELOS ODE (NODE)
# ---------------------------------------------------------------------------
class BaseODE(nn.Module):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0):
        super().__init__()
        self.m1, self.L1 = 0.122, 0.39
        self.m2, self.L2 = 0.055, 0.347
        self.g = 9.81
        self.log_J  = nn.Parameter(torch.log(torch.tensor(float(J0))))
        self.log_b  = nn.Parameter(torch.log(torch.tensor(float(b0))))
        self.log_Gu = nn.Parameter(torch.log(torch.tensor(float(Gu0))))
        self.u_series          = None
        self.t_series          = None
        self.batch_start_times = None

    def get_params(self):
        return torch.exp(self.log_J), torch.exp(self.log_b), torch.exp(self.log_Gu)

    def _get_u_t(self, t, x):
        """Interpolacao linear de u para o instante t (suporta batch de trajetorias)."""
        if self.batch_start_times is not None:
            t_abs = self.batch_start_times + t       # [B, 1]
        else:
            t_abs = t * torch.ones(x.shape[0], 1, device=x.device)

        k = torch.searchsorted(self.t_series, t_abs.reshape(-1), right=True)
        k = torch.clamp(k, 1, len(self.t_series) - 1)
        t1 = self.t_series[k - 1].unsqueeze(1)
        t2 = self.t_series[k    ].unsqueeze(1)
        u1 = self.u_series[k - 1]
        u2 = self.u_series[k    ]
        denom = torch.where((t2 - t1) < 1e-6, torch.ones_like(t2 - t1), t2 - t1)
        alpha = (t_abs.reshape(-1, 1) - t1) / denom
        return u1 + alpha * (u2 - u1)


class PhysicsODE_Baseline(BaseODE):
    def forward(self, t, x):
        J, b, Gu = self.get_params()
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        motor_torque    = Gu * u_t * torch.abs(u_t)
        gravity_torque  = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot
        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)


class PhysicsODE_Asymmetric(BaseODE):
    def __init__(self, J0=1.0, b_pos0=np.exp(-1.0), b_neg0=np.exp(-1.0), Gu_pos0=1.0, Gu_neg0=1.0):
        super().__init__(J0, b_pos0, Gu_pos0)
        self.log_b_neg  = nn.Parameter(torch.log(torch.tensor(float(b_neg0))))
        self.log_Gu_neg = nn.Parameter(torch.log(torch.tensor(float(Gu_neg0))))

    def forward(self, t, x):
        J      = torch.exp(self.log_J)
        b_pos  = torch.exp(self.log_b)
        b_neg  = torch.exp(self.log_b_neg)
        Gu_pos = torch.exp(self.log_Gu)
        Gu_neg = torch.exp(self.log_Gu_neg)
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]

        sigma_b  = torch.sigmoid(50.0 * theta_dot)
        b        = sigma_b * b_pos + (1.0 - sigma_b) * b_neg

        sigma_Gu = torch.sigmoid(50.0 * u_t)
        Gu       = sigma_Gu * Gu_pos + (1.0 - sigma_Gu) * Gu_neg

        motor_torque    = Gu * u_t * torch.abs(u_t)
        gravity_torque  = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot
        theta_ddot = (motor_torque - gravity_torque - friction_torque) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)


class PhysicsODE_Hybrid(BaseODE):
    def __init__(self, J0=1.0, b0=np.exp(-1.0), Gu0=1.0, hidden_dim=16):
        super().__init__(J0, b0, Gu0)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t, x):
        J, b, Gu = self.get_params()
        u_t = self._get_u_t(t, x)
        theta, theta_dot = x[:, 0:1], x[:, 1:2]
        motor_torque    = Gu * u_t * torch.abs(u_t)
        gravity_torque  = (self.m1 * self.L1 - self.m2 * self.L2) * self.g * torch.sin(theta)
        friction_torque = b * theta_dot
        nn_input        = torch.cat([theta, theta_dot, u_t], dim=1)
        residual_torque = self.mlp(nn_input)
        theta_ddot = (motor_torque - gravity_torque - friction_torque + residual_torque) / J
        return torch.cat([theta_dot, theta_ddot], dim=1)


# ---------------------------------------------------------------------------
# 2. FUNCAO DE TREINAMENTO MULTIPLE SHOOTING — OTIMIZADA
# ---------------------------------------------------------------------------
def train_model_v5_ms(model, name, train_datasets, val_datasets, epochs=500, lr=0.01,
                      chunk_size=100, lambda_continuity=1.0,
                      state_std=None, integrator='rk4', weight_decay_mlp=1e-4,
                      val_every=25):
    print(f"\n--- Iniciando Treinamento V5 (Multiple Shooting): {name} ---")
    model.to(device)

    # ------------------------------------------------------------------
    # 1. Preparacao dos dados: Quebrar os datasets em chunks.
    #    Todos os tensores sao pre-transferidos para GPU aqui.
    # ------------------------------------------------------------------
    chunks              = []
    chunk_index_counter = 0
    next_chunk_indices  = []

    for ds in train_datasets:
        t_ds, u_ds, x_ds = ds['t'], ds['u'], ds['x']
        N          = len(t_ds)
        num_chunks = N // chunk_size

        for i in range(num_chunks):
            start   = i * chunk_size
            end     = start + chunk_size
            chunk_t = t_ds[start:end].to(device)
            chunk_u = u_ds[start:end].to(device)
            chunk_x = x_ds[start:end].to(device)
            dt_k    = (chunk_t[1] - chunk_t[0]).item()

            # t_eval relativo pre-calculado — reutilizado em toda epoca
            t_eval_k = torch.arange(chunk_size, device=device, dtype=torch.float32) * dt_k

            chunks.append({
                'id'    : chunk_index_counter,
                't'     : chunk_t,
                'u'     : chunk_u,
                'x'     : chunk_x,
                'dt'    : dt_k,
                't_eval': t_eval_k,
                't0'    : chunk_t[0].reshape(1, 1),
            })

            if i < num_chunks - 1:
                next_chunk_indices.append(chunk_index_counter + 1)
            else:
                next_chunk_indices.append(-1)

            chunk_index_counter += 1

    num_total_chunks = len(chunks)
    print(f"Total de Chunks gerados: {num_total_chunks} (Tamanho do Chunk = {chunk_size})")

    # ------------------------------------------------------------------
    # 2. Verificar se todos os chunks tem o mesmo dt para habilitar
    #    o odeint em batch completo (1 call/epoca).
    # ------------------------------------------------------------------
    # Arredonda dt a 6 casas decimais para absorver erros de ponto flutuante
    # entre datasets com mesmo decimacao mas tamanhos ligeiramente distintos.
    all_dts    = [round(c['dt'], 6) for c in chunks]
    unique_dts = list(set(all_dts))
    batch_mode = len(unique_dts) == 1

    if batch_mode:
        print("Modo: BATCH TOTAL (1 odeint call/epoca) — maxima performance.")
        t_eval_global = chunks[0]['t_eval']   # [chunk_size]

        # Construir t_series global com offset artificial por chunk
        # para que os tempos nao se sobreponham entre datasets distintos.
        MAX_TIME = (chunks[-1]['t'][-1].item() + 1.0) if chunks else 1.0
        shifted_ts = []
        for idx, c in enumerate(chunks):
            shifted_ts.append(c['t'] + idx * MAX_TIME)
        t_series_global = torch.cat(shifted_ts)         # [B*chunk_size]
        u_series_global = torch.cat([c['u'] for c in chunks])  # [B*chunk_size, 1]

        # batch_start_times: [B, 1] com os t0 shiftados por chunk
        batch_starts = torch.stack([
            c['t'][0].reshape(1) + i * MAX_TIME for i, c in enumerate(chunks)
        ]).to(device)   # [B, 1]
    else:
        print(f"Modo: MULTI-DT ({len(unique_dts)} grupos) — fallback com loop por chunk.")

    # ------------------------------------------------------------------
    # 3. Parametros X0_hat (condicoes iniciais treinaveis)
    # ------------------------------------------------------------------
    initial_x0s = torch.stack([c['x'][0] for c in chunks]).to(device)  # [B, 2]
    X0_hat      = nn.Parameter(initial_x0s)

    # ------------------------------------------------------------------
    # 4. Otimizador
    # ------------------------------------------------------------------
    if hasattr(model, 'mlp'):
        mlp_params  = [p for n, p in model.named_parameters() if 'mlp' in n]
        phys_params = [p for n, p in model.named_parameters() if 'mlp' not in n]
        optimizer   = optim.Adam([
            {'params': phys_params, 'weight_decay': 0.0},
            {'params': mlp_params,  'weight_decay': weight_decay_mlp},
            {'params': [X0_hat],    'weight_decay': 0.0, 'lr': lr},
        ], lr=lr)
    else:
        optimizer = optim.Adam(list(model.parameters()) + [X0_hat], lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if state_std is None:
        state_std = torch.ones(2, device=device)

    best_val_loss   = float('inf')
    best_state_dict = None
    val_loss        = float('inf')

    # Target global pre-montado: [B, chunk_size, 2]
    target_global = torch.stack([c['x'] for c in chunks])

    # Indices de continuidade pre-computados
    cont_pairs = [(i, nxt) for i, nxt in enumerate(next_chunk_indices) if nxt != -1]

    # ------------------------------------------------------------------
    # LOOP DE TREINAMENTO
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        if batch_mode:
            # ---- 1 odeint call para TODOS os chunks ----
            model.t_series          = t_series_global
            model.u_series          = u_series_global
            model.batch_start_times = batch_starts   # [B, 1]

            # pred_states: [chunk_size, B, 2]  →  permute →  [B, chunk_size, 2]
            pred_states = odeint(model, X0_hat, t_eval_global, method=integrator)
            pred_states = pred_states.permute(1, 0, 2)

            # Perda de dados (vetorizado, sem loop Python)
            data_loss = torch.mean(((pred_states - target_global) / state_std) ** 2)

            # Perda de continuidade (vetorizado)
            if cont_pairs:
                src_idx     = torch.tensor([p[0] for p in cont_pairs], device=device)
                tgt_idx     = torch.tensor([p[1] for p in cont_pairs], device=device)
                end_states  = pred_states[src_idx, -1, :]
                next_x0s    = X0_hat[tgt_idx]
                cont_loss   = torch.mean(((end_states - next_x0s) / state_std) ** 2)
            else:
                cont_loss = torch.tensor(0.0, device=device)

        else:
            # ---- Fallback: loop por chunk (dts heterogeneos) ----
            losses      = []
            cont_losses = []

            for i, chunk in enumerate(chunks):
                model.t_series          = chunk['t']
                model.u_series          = chunk['u']
                model.batch_start_times = chunk['t0']

                pred_state = odeint(
                    model, X0_hat[i:i+1], chunk['t_eval'], method=integrator
                ).squeeze(1)

                losses.append(torch.mean(((pred_state - chunk['x']) / state_std) ** 2))

                next_idx = next_chunk_indices[i]
                if next_idx != -1:
                    cont_losses.append(torch.mean(
                        ((pred_state[-1] - X0_hat[next_idx]) / state_std) ** 2
                    ))

            data_loss = torch.stack(losses).mean()
            cont_loss = (torch.stack(cont_losses).mean()
                         if cont_losses else torch.tensor(0.0, device=device))

        loss = data_loss + lambda_continuity * cont_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % val_every == 0 or epoch == epochs:
            val_loss = avalia_validacao(model, val_datasets, state_std, integrator)
            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d} | LR={lr_now:.5f} | Data: {data_loss.item():.5f} "
                  f"| Cont: {cont_loss.item():.5f} | Val (Free-Run): {val_loss:.5f}")

    print(f"[{name}] Concluido. Restaurando melhor modelo (Val Loss: {best_val_loss:.5f}).")
    model.load_state_dict(best_state_dict)
    return model


def avalia_validacao(model, val_datasets, state_std, integrator):
    """Free-run evaluation em datasets de validacao (sem chunking)."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for ds in val_datasets:
            t_t, u_t, x_t = ds['t'].to(device), ds['u'].to(device), ds['x'].to(device)
            model.t_series          = t_t
            model.u_series          = u_t
            model.batch_start_times = torch.zeros(1, 1, device=device)

            x0   = x_t[0].unsqueeze(0)
            pred = odeint(model, x0, t_t, method=integrator).squeeze(1)
            val_loss += torch.mean(((pred - x_t) / state_std) ** 2).item()
    return val_loss / len(val_datasets)


# ---------------------------------------------------------------------------
# 3. DATASET SPLIT — STATE-OF-THE-ART PARA SYSTEM IDENTIFICATION
#
# Principio: cross-session validation (Ljung 1999; Schon et al. 2011).
#   - TREINO  : sessoes com excitacao rica e variada (APRBS, multi-seno,
#               swept-sine, seq-degraus). Usamos a sessao mais recente (Rodada 5 /
#               raiz 27-08) como fonte primaria — sistema mais calibrado.
#   - VALIDACAO: sessoes completamente distintas (Rodadas 2 e 3, datas diferentes).
#               Tipos de sinal diferentes (chirp, seq-degraus) para testar
#               generalizacao fora do dominio de treino.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Carregamento online — mesmo padrao do notebook ANN_NARX_AEROPENDULO_v4
    BASE = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/experimentos/"

    # ------------------------------------------------------------------
    # TREINO — Rodada 5 (27/08, raiz) + Rodada 4 (19/08, RODADA-4/)
    # Cobertura: APRBS (pos/neg), multi-seno (4 variantes), swept-sine,
    #            seq-degraus — excitacao espectral ampla.
    # ------------------------------------------------------------------
    train_files = [
        # --- Rodada 5 (27/08) — sessao mais recente, maior confianca ---
        ("",          "aprbs-1_0827_17-19.csv"),
        ("",          "aprbs-2_0827_17-25.csv"),
        ("",          "aprbs-3_0827_17-28.csv"),
        ("",          "multi-seno-1_0827_17-34.csv"),
        ("",          "multi-seno-2_0827_17-37.csv"),
        ("",          "multi-seno-3_0827_17-40.csv"),
        ("",          "seq-degraus-1_0827_17-46.csv"),
        ("",          "seq-degraus-2_0827_17-49.csv"),
        ("",          "swept-sine-1_0827_17-58.csv"),
        # --- Rodada 4 (19/08) — complementa diversidade ---
        ("RODADA-4/", "aprbs-1_0819_18-48.csv"),
        ("RODADA-4/", "aprbs-neg-1_0819_18-54.csv"),
        ("RODADA-4/", "multi-seno-1_0819_19-23.csv"),
        ("RODADA-4/", "multi-seno-3_0819_19-19.csv"),
        ("RODADA-4/", "swept-sine-1_0819_19-02.csv"),
        ("RODADA-4/", "seq-degraus-aprbs-2_0819_19-16.csv"),
    ]

    # ------------------------------------------------------------------
    # VALIDACAO — Rodadas 2 (04/08) e 3 (07/08)
    # Cross-session: datas distintas, condicoes ambientais distintas.
    # Inclui chirp (excitacao nao vista no treino) para testar generalizacao.
    # ------------------------------------------------------------------
    val_files = [
        # Rodada 2 (04/08)
        ("RODADA-2/", "chirp-1_0804_19-19.csv"),
        ("RODADA-2/", "multi-seno-2_0804_19-28.csv"),
        ("RODADA-2/", "seq-degraus-1_0804_19-09.csv"),
        # Rodada 3 (07/08)
        ("RODADA-3/", "chirp-1_0807_16-34.csv"),
        ("RODADA-3/", "seq-degraus-1_0807_16-42.csv"),
        ("RODADA-3/", "multi-seno-1_0807_16-57.csv"),
    ]

    decimacao = 2
    START_IDX, END_IDX = 250, -200

    print("Carregando datasets de TREINO (online — GitHub raw)...")
    print(f"  start_idx={START_IDX}  end_idx={END_IDX}  decimacao={decimacao}  dt={0.010*decimacao:.3f}s\n")
    train_datasets = []
    for subdir, fname in train_files:
        path = BASE + subdir + fname
        try:
            t_raw, u_raw, y_raw = carregar_experimento(
                path, decimacao=decimacao, start_idx=START_IDX, end_idx=END_IDX
            )
            t_ten, u_ten, x_ten, *_ = processar_dataset(t_raw, u_raw, y_raw)
            n = len(t_ten)
            dur = t_raw[-1] - t_raw[0]
            print(f"  [TREINO] {subdir}{fname}")
            print(f"           amostras={n:5d} | duracao={dur:.1f}s "
                  f"| theta=[{x_ten[:,0].min():.2f}, {x_ten[:,0].max():.2f}] rad "
                  f"| u=[{u_ten.min():.2f}, {u_ten.max():.2f}]")
            train_datasets.append({'name': fname, 't': t_ten, 'u': u_ten, 'x': x_ten})
        except Exception as e:
            print(f"  [SKIP] {subdir}{fname}: {e}")

    print("\nCarregando datasets de VALIDACAO (cross-session, online)...")
    print(f"  start_idx={START_IDX}  end_idx={END_IDX}  decimacao={decimacao}  dt={0.010*decimacao:.3f}s\n")
    val_datasets = []
    for subdir, fname in val_files:
        path = BASE + subdir + fname
        try:
            t_raw, u_raw, y_raw = carregar_experimento(
                path, decimacao=decimacao, start_idx=START_IDX, end_idx=END_IDX
            )
            t_ten, u_ten, x_ten, *_ = processar_dataset(t_raw, u_raw, y_raw)
            n = len(t_ten)
            dur = t_raw[-1] - t_raw[0]
            print(f"  [VAL]    {subdir}{fname}")
            print(f"           amostras={n:5d} | duracao={dur:.1f}s "
                  f"| theta=[{x_ten[:,0].min():.2f}, {x_ten[:,0].max():.2f}] rad "
                  f"| u=[{u_ten.min():.2f}, {u_ten.max():.2f}]")
            val_datasets.append({'name': fname, 't': t_ten, 'u': u_ten, 'x': x_ten})
        except Exception as e:
            print(f"  [SKIP] {subdir}{fname}: {e}")

    print(f"\n-> {len(train_datasets)} datasets de treino | {len(val_datasets)} de validacao")

    # Normalizacao do estado pelos dados de treino
    all_pos   = np.concatenate([d['x'][:, 0].numpy() for d in train_datasets])
    all_vel   = np.concatenate([d['x'][:, 1].numpy() for d in train_datasets])
    state_std = torch.tensor([float(np.std(all_pos)), float(np.std(all_vel))],
                              dtype=torch.float32, device=device)
    print(f"state_std: theta={state_std[0]:.4f} rad | theta_dot={state_std[1]:.4f} rad/s")

    os.makedirs('modelos_salvos', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # chunk_size=150 ~ 3 s por chunk com dt=0.02 s (decimacao=2)
    chunk_size = 150
    epochs     = 400
    val_every  = 25   # validacao a cada 25 epocas (antes: 10)

    if TREINAR_BASELINE:
        base_model = PhysicsODE_Baseline()
        base_model = train_model_v5_ms(
            base_model, "Baseline", train_datasets, val_datasets,
            epochs=epochs, lr=0.015, chunk_size=chunk_size, lambda_continuity=1.0,
            state_std=state_std, val_every=val_every
        )
        torch.save(base_model.state_dict(), f'modelos_salvos/node_v5_baseline_{timestamp}.pth')

    if TREINAR_ASSIMETRICO:
        asymm_model = PhysicsODE_Asymmetric()
        asymm_model = train_model_v5_ms(
            asymm_model, "Assimetrico", train_datasets, val_datasets,
            epochs=epochs, lr=0.015, chunk_size=chunk_size, lambda_continuity=1.0,
            state_std=state_std, val_every=val_every
        )
        torch.save(asymm_model.state_dict(), f'modelos_salvos/node_v5_asymmetric_{timestamp}.pth')

    if TREINAR_HIBRIDO:
        hybrid_model = PhysicsODE_Hybrid(hidden_dim=16)
        hybrid_model = train_model_v5_ms(
            hybrid_model, "Hibrido", train_datasets, val_datasets,
            epochs=epochs, lr=0.015, chunk_size=chunk_size, lambda_continuity=1.0,
            state_std=state_std, weight_decay_mlp=1e-4, val_every=val_every
        )
        torch.save(hybrid_model.state_dict(), f'modelos_salvos/node_v5_hybrid_{timestamp}.pth')

    print("\n[V5 - Multiple Shooting] Treinamento Finalizado e Modelos Salvos.")
