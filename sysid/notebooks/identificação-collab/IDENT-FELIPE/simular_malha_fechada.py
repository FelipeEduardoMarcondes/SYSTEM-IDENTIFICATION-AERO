"""
Simulação em Malha Fechada do Aeropêndulo com Controlador PID do Firmware STM32
================================================================================
Compara a simulação em malha fechada do modelo grey-box v2 contra os dados
experimentais reais da bancada (RODADA-7).

Ganhos e algoritmo extraídos de:
  firmware/stm32-standard/Core/Src/aeropendulo_main.c:
    - KP = 0.5793
    - KI = 0.6647
    - KD = 0.2000
    - TS = 0.0100 s (100 Hz)
    - Algoritmo: Tustin para integral, backward para derivada na saída,
                 anti-windup por back-calculation, saturação [-100, 100]%.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
for p in (root_dir, current_dir):
    if p not in sys.path:
        sys.path.append(p)

from aerodata import readData


# =============================================================================
# 1. Controlador PID Digital (Réplica Exata do Firmware STM32)
# =============================================================================
class STM32PIDController:
    """
    Réplica do controlador PID implementado no firmware STM32
    (Core/Src/aeropendulo_main.c, linhas 751-778).
    """
    def __init__(self, kp=0.5793, ki=0.6647, kd=0.2000, ts=0.01, u_max=100.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.ts = float(ts)
        self.u_max = float(u_max)
        self.reset()

    def reset(self, e_init=0.0, y_init=0.0):
        self.e_1 = float(e_init)
        self.y_1 = float(y_init)
        self.u_i = 0.0

    def update(self, r, y_med):
        """
        Calcula o sinal de controle u [%] a partir de r [°] e y_med [°].
        """
        e = r - y_med
        u_p = self.kp * e
        self.u_i += self.ki * (self.ts / 2.0) * (e + self.e_1)  # Tustin
        u_d = -(self.kd / self.ts) * (y_med - self.y_1)         # Derivada na saída
        u_raw = u_p + self.u_i + u_d

        # Anti-windup por back-calculation (idêntico ao firmware C)
        u_sat = np.clip(u_raw, -self.u_max, self.u_max)
        if u_raw != u_sat:
            self.u_i -= (u_raw - u_sat)

        self.e_1 = e
        self.y_1 = y_med
        return float(u_sat)


# =============================================================================
# 2. Simulador Não-Linear da Planta (Grey-box de 3ª Ordem)
# =============================================================================
class AeropenduloSimulator:
    """
    Simulador contínuo-discreto do aeropêndulo não-linear.
    Compatível com v1 (2ª ordem), v2-v5 (3ª ordem quadrático) e v8-v9 (com rigidez k_th,
    arrasto ca e expoente p).
    """
    def __init__(self, params):
        self.c1 = float(params['c1'])
        self.c2 = float(params['c2'])
        self.c3 = float(params['c3'])
        self.k_th = float(params.get('k_th', 0.0))    # Rigidez elástica do cabo elétrico (v8/v9)
        self.Fc = float(params.get('Fc', 0.0))
        self.Fs = float(params.get('Fs', 0.0))
        self.vs = float(params.get('vs', 0.5))
        self.c_aero = float(params.get('ca', params.get('c_aero', 0.0)))
        self.tau_m = float(params.get('tau_m', 0.0))  # 0.0 indica modelo v1 (atuador instantâneo)
        self.u0 = float(params.get('u0', 0.0))        # Zona morta do motor/ESC [%]
        self.p_thrust = float(params.get('p', 2.0))   # Expoente do empuxo aerodinâmico (v9: ~1.759)

    def deriv(self, th, w, a, u):
        fric = -(self.Fc + self.Fs * np.exp(-(w / self.vs)**2)) * np.tanh(50.0 * w)
        
        # Ativação do motor
        val_act = a if self.tau_m > 1e-4 else u
        eff = np.maximum(np.abs(val_act) - self.u0, 0.0)
        thrust = np.sign(val_act) * (eff ** self.p_thrust)

        # Aceleração angular com rigidez de cabo e freio aerodinâmico quadrático
        acc = (self.c1 * np.sin(th)
               - self.k_th * th
               + self.c2 * w
               - self.c_aero * w * np.abs(w)
               + self.c3 * thrust
               + fric)

        a_dot = (u - a) / self.tau_m if self.tau_m > 1e-4 else 0.0
        return w, acc, a_dot

    def rk4_step(self, th, w, a, u, dt):
        k1_th, k1_w, k1_a = self.deriv(th, w, a, u)
        k2_th, k2_w, k2_a = self.deriv(th + 0.5*dt*k1_th, w + 0.5*dt*k1_w, a + 0.5*dt*k1_a, u)
        k3_th, k3_w, k3_a = self.deriv(th + 0.5*dt*k2_th, w + 0.5*dt*k2_w, a + 0.5*dt*k2_a, u)
        k4_th, k4_w, k4_a = self.deriv(th + dt*k3_th, w + dt*k3_w, a + dt*k3_a, u)

        th_n = th + (dt / 6.0) * (k1_th + 2*k2_th + 2*k3_th + k4_th)
        w_n = w + (dt / 6.0) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        a_n = a + (dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
        return th_n, w_n, a_n


# =============================================================================
# 3. Métricas
# =============================================================================
def fit_pct(y_real, y_sim):
    denom = np.linalg.norm(y_real - np.mean(y_real))
    if denom < 1e-9:
        return 0.0
    return float(100.0 * (1.0 - np.linalg.norm(y_real - y_sim) / denom))


def rmse(y_real, y_sim):
    return float(np.sqrt(np.mean((y_real - y_sim)**2)))


# =============================================================================
# 4. Simulação em Malha Fechada
# =============================================================================
def simular_ensaio(exp_path, plant_sim, pid_ctrl, trim_start=5.0, trim_end=5.0):
    """
    Executa a simulação em malha fechada para um ensaio da RODADA-7.
    """
    # Normaliza caminho para o padrão esperado por readData ('data/...')
    if os.path.isabs(exp_path):
        rel_path = os.path.relpath(exp_path, root_dir).replace('\\', '/')
    else:
        rel_path = exp_path.replace('\\', '/')

    res = readData(rel_path, decimar=1, return_ref=True)
    if len(res) == 4:
        y_real_deg, u_real, t, r_deg = res
    else:
        y_real_deg, u_real, t = res
        r_deg = np.zeros_like(y_real_deg)

    # Recorte dos transientes de armação
    i0 = np.searchsorted(t, t[0] + trim_start)
    i1 = np.searchsorted(t, t[-1] - trim_end) if trim_end > 0 else len(t)

    t = t[i0:i1] - t[i0]
    y_real_deg = y_real_deg[i0:i1]
    u_real = u_real[i0:i1]
    r_deg = r_deg[i0:i1]

    dt = float(np.mean(np.diff(t)))
    N = len(t)

    # Estados iniciais sincronizados com a bancada no início do recorte
    th = np.deg2rad(y_real_deg[0])
    w = np.deg2rad((y_real_deg[1] - y_real_deg[0]) / dt) if N > 1 else 0.0
    a = float(u_real[0])

    pid_ctrl.reset(e_init=(r_deg[0] - y_real_deg[0]), y_init=y_real_deg[0])

    y_sim_deg = np.zeros(N)
    u_sim = np.zeros(N)

    for k in range(N):
        y_meas = float(np.rad2deg(th))
        y_sim_deg[k] = y_meas

        # Controle digital STM32
        u_k = pid_ctrl.update(r=r_deg[k], y_med=y_meas)
        u_sim[k] = u_k

        # Evolução contínua da planta
        th, w, a = plant_sim.rk4_step(th, w, a, u_k, dt)

    fit_y = fit_pct(y_real_deg, y_sim_deg)
    rmse_y = rmse(y_real_deg, y_sim_deg)
    fit_u = fit_pct(u_real, u_sim)
    rmse_u = rmse(u_real, u_sim)

    return {
        'name': os.path.basename(exp_path),
        't': t,
        'r': r_deg,
        'y_real': y_real_deg,
        'y_sim': y_sim_deg,
        'u_real': u_real,
        'u_sim': u_sim,
        'fit_y': fit_y,
        'rmse_y': rmse_y,
        'fit_u': fit_u,
        'rmse_u': rmse_u,
    }


# =============================================================================
# 5. Plotagem Comparativa
# =============================================================================
def plotar_comparacao(res, out_png=None):
    fig, (ax_ang, ax_u) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                                       gridspec_kw={'height_ratios': [1.3, 1]})

    t = res['t']
    # Painel 1: Resposta Angular θ e Referência r
    ax_ang.plot(t, res['r'], 'r--', lw=1.3, label='Referência r [°]', alpha=0.9)
    ax_ang.plot(t, res['y_real'], 'k-', lw=1.2, label='Bancada Real θ [°]', alpha=0.9)
    ax_ang.plot(t, res['y_sim'], 'b--', lw=1.4, label=f"Simulação Grey-box (FIT: {res['fit_y']:.1f}%)")

    ax_ang.set_ylabel('Ângulo θ [°]', fontsize=10, fontweight='bold')
    ax_ang.set_title(f"Malha Fechada: {res['name']} — Ângulo θ (FIT: {res['fit_y']:.1f}% | RMSE: {res['rmse_y']:.2f}°)",
                     fontsize=11, fontweight='bold')
    ax_ang.grid(True, linestyle=':', alpha=0.6)
    ax_ang.legend(loc='upper right', fontsize=9)

    # Painel 2: Sinal de Controle u [%]
    ax_u.plot(t, res['u_real'], color='gray', lw=1.0, label='Controle Real u [%]', alpha=0.7)
    ax_u.plot(t, res['u_sim'], color='tab:blue', lw=1.3, label=f"Controle Simulado (FIT: {res['fit_u']:.1f}%)")
    ax_u.set_ylabel('Controle u [%]', fontsize=10, fontweight='bold')
    ax_u.set_xlabel('Tempo [s]', fontsize=10, fontweight='bold')
    ax_u.set_title(f"Sinal de Controle u [%] (FIT: {res['fit_u']:.1f}% | RMSE: {res['rmse_u']:.2f}%)",
                   fontsize=10)
    ax_u.grid(True, linestyle=':', alpha=0.6)
    ax_u.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=120)
        print(f"Gráfico salvo em: {out_png}")
    plt.close(fig)


# =============================================================================
# Execução Principal
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulação em Malha Fechada com PID do STM32")
    parser.add_argument("--version", "--v", type=str, default="v9",
                        help="Versão do modelo grey-box (ex: 'v1', 'v2', 'v3', 'v8', 'v9'). Padrão: v9")
    parser.add_argument("--model", type=str, default=None,
                        help="Modo/chave do modelo no JSON (ex: 'v9_otimizado', 'v9_completo_r1', 'c3 livre', etc.)")
    parser.add_argument("--file", type=str, default=None,
                        help="Arquivo específico de experimento da RODADA-7 para simular")
    parser.add_argument("--all", action="store_true",
                        help="Simular todos os 24 ensaios da RODADA-7 e exibir tabela consolidada")
    args = parser.parse_args()

    # 1. Carrega modelo identificado (v1 a v9)
    json_filename = f"modelo_greybox_{args.version}.json"
    json_path = os.path.join(current_dir, json_filename)
    if not os.path.exists(json_path):
        print(f"Erro: arquivo do modelo não encontrado em {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        mod_data = json.load(f)

    models_dict = mod_data.get("models", {})
    if args.model is None:
        if "v9_otimizado" in models_dict:
            model_key = "v9_otimizado"
        elif "v9_completo_r1" in models_dict:
            model_key = "v9_completo_r1"
        elif "c3 livre" in models_dict:
            model_key = "c3 livre"
        else:
            model_key = list(models_dict.keys())[0]
    else:
        model_key = args.model

    if model_key not in models_dict:
        print(f"Erro: modo '{model_key}' não encontrado no JSON {json_filename}. Modos disponíveis: {list(models_dict.keys())}")
        sys.exit(1)

    params = models_dict[model_key]["params"]
    print("\n" + "=" * 78)
    print(f"SIMULAÇÃO EM MALHA FECHADA — MODELO: {args.version.upper()} ('{model_key.upper()}') + PID STM32")
    print("=" * 78)
    print(f"Arquivo do Modelo: {json_filename}")
    print(f"Parâmetros Identificados:")
    print(f"  c1 = {params['c1']:.4f} | c2 = {params['c2']:.4f} | c3 = {params['c3']:.6e}")
    k_th_val = params.get('k_th', 0.0)
    p_thrust_val = params.get('p', 2.0)
    ca_val = params.get('ca', params.get('c_aero', 0.0))
    tau_m_val = params.get('tau_m', 0.0)
    u0_val = params.get('u0', 0.0)
    Fc_val = params.get('Fc', 0.0)
    print(f"  k_th = {k_th_val:.4f} rad/s^2 | p = {p_thrust_val:.4f} | ca = {ca_val:.5f} rad^-1")
    print(f"  u0 = {u0_val:.2f}% | tau_m = {tau_m_val*1000:.1f} ms | Fc = {Fc_val:.4f} rad/s^2")
    print(f"\nParâmetros do PID do Firmware STM32:")
    print(f"  Kp = 0.5793 | Ki = 0.6647 | Kd = 0.2000 | Ts = 10 ms (100 Hz)")
    print("=" * 78)

    plant = AeropenduloSimulator(params)
    pid = STM32PIDController(kp=0.5793, ki=0.6647, kd=0.2000, ts=0.01, u_max=100.0)

    # 2. Localiza os ensaios da RODADA-7
    r7_dir = os.path.join(root_dir, "data", "experimentos", "coleta_dados", "RODADA-7")
    if args.file:
        cands = [args.file] if os.path.isabs(args.file) else [
            os.path.join(root_dir, args.file),
            os.path.join(r7_dir, args.file),
            os.path.join(current_dir, args.file),
            args.file
        ]
        target_files = [f for f in cands if os.path.exists(f)]
        if not target_files:
            # Tenta busca parcial recursiva apenas em pastas de coletas/experimentos
            raw_cands = glob.glob(os.path.join(root_dir, "data", "**", f"*{args.file}*.csv"), recursive=True)
            target_files = [f for f in raw_cands if "coleta" in f.lower() or "experimento" in f.lower()]
            if not target_files:
                target_files = raw_cands
        if not target_files:
            print(f"Erro: arquivo '{args.file}' não encontrado.")
            sys.exit(1)
    else:
        # Padrão: roda 4 ensaios representativos (ou todos se --all)
        all_files = sorted(glob.glob(os.path.join(r7_dir, "*.csv")))
        if args.all:
            target_files = all_files
        else:
            representativos = [
                "seq-degraus-45-1",
                "chirp-45-amp25",
                "aprbs-45-1",
                "multi-seno-45-030Hz",
                "seq-degraus-60-1",
                "chirp-60-amp40",
                "aprbs-60-1",
                "multi-seno-60-040Hz",
            ]
            target_files = []
            for rep in representativos:
                match = [f for f in all_files if rep in os.path.basename(f)]
                if match:
                    target_files.append(match[0])

    print(f"\nExecutando simulação para {len(target_files)} ensaios...\n")
    print(f"{'Ensaio':32s} | {'FIT Ângulo':>12s} | {'RMSE Ângulo':>12s} | {'FIT Controle':>12s} | {'RMSE Controle':>12s}")
    print("-" * 92)

    results = []
    out_dir = os.path.join(current_dir, "graficos_malha_fechada")
    os.makedirs(out_dir, exist_ok=True)

    for fpath in target_files:
        try:
            res = simular_ensaio(fpath, plant, pid)
        except Exception as e:
            continue
        results.append(res)
        print(f"{res['name'][:32]:32s} | {res['fit_y']:11.1f}% | {res['rmse_y']:10.2f}° | {res['fit_u']:11.1f}% | {res['rmse_u']:10.2f}%")

        clean_name = res['name'].replace('.csv', '')
        png_path = os.path.join(out_dir, f"mf_{clean_name}.png")
        plotar_comparacao(res, out_png=png_path)

    # Resumo
    fits_y = [r['fit_y'] for r in results]
    rmses_y = [r['rmse_y'] for r in results]
    fits_u = [r['fit_u'] for r in results]
    rmses_u = [r['rmse_u'] for r in results]

    print("-" * 92)
    print(f"{'MÉDIA':32s} | {np.mean(fits_y):11.1f}% | {np.mean(rmses_y):10.2f}° | {np.mean(fits_u):11.1f}% | {np.mean(rmses_u):10.2f}%")
    print(f"{'MEDIANA':32s} | {np.median(fits_y):11.1f}% | {np.median(rmses_y):10.2f}° | {np.median(fits_u):11.1f}% | {np.median(rmses_u):10.2f}%")
    print("=" * 92)
    print(f"\nGráficos comparativos salvos em:\n  {out_dir}")
