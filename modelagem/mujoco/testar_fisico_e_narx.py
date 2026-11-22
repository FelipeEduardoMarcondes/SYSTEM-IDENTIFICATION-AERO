"""
testar_fisico_e_narx.py — Ambiente de Teste e Benchmark no MuJoCo:
Integra e compara simultaneamente:
  1. Modelo Físico Grey-Box v9 (`modelo_greybox_v9.json` — `v9_otimizado`, com rigidez torsional
     do cabo K_s, arrasto quadrático d_a, atrito de Coulomb Tc e empuxo sub-quadrático p=1.759
     rodando nativamente dentro do `mj_step` do MuJoCo 3D)
  2. Modelo NARX Polinomial (`narx_model.json` — SR-FROLS + Estabilidade de Aguirre + LM-OE)
  3. Dados Experimentais Reais da Bancada (6 ensaios de validação da RODADA-7)

Modos de execução:
  - Benchmark + Gráficos + Vídeo 3D Lado a Lado:
      python modelagem/mujoco/testar_fisico_e_narx.py
  - Janela 3D Interativa do MuJoCo com os 3 braços (Físico v9 3D + NARX Roxo + Ref Laranja):
      python modelagem/mujoco/testar_fisico_e_narx.py --viewer
"""

import os
import io
import re
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from PIL import Image
import mujoco

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from aerodata import readData

XML_PATH = os.path.join(BASE_DIR, "aeropendulo.xml")
NARX_SR_FROLS_JSON = os.path.join(ROOT_DIR, "sysid", "notebooks", "identificação-collab", "IDENT-TESTES", "narx_model.json")
GREYBOX_V9_JSON = os.path.join(ROOT_DIR, "sysid", "notebooks", "identificação-collab", "IDENT-FELIPE", "modelo_greybox_v9.json")

VAL_DATASETS = [
    ("multisine_45_val", "RODADA-7/multi-seno-45-030Hz_0904_20-23.csv", "Multisseno (45°, 0.30 Hz)"),
    ("chirp_45_val",     "RODADA-7/chirp-45-amp35_0904_20-13.csv",      "Chirp (45°, Amp ±35°)"),
    ("steps_45_val",     "RODADA-7/seq-degraus-45-2_0904_20-45.csv",    "Seq. Degraus (45°)"),
    ("multisine_60_val", "RODADA-7/multi-seno-60-030Hz_0904_20-32.csv", "Multisseno (60°, 0.30 Hz)"),
    ("chirp_60_val",     "RODADA-7/chirp-60-amp40_0904_20-16.csv",      "Chirp (60°, Amp ±40°)"),
    ("steps_60_val",     "RODADA-7/seq-degraus-60-2_0904_20-54.csv",    "Seq. Degraus (60°)"),
]


# ==============================================================================
# 1. CLASSE DO MODELO FÍSICO GREY-BOX V9 ACOPLADO AO MUJOCO (mj_step)
# ==============================================================================
class GreyBoxV9MuJoCo:
    """
    Carrega os parâmetros físicos do Modelo Grey-Box v9 (`modelo_greybox_v9.json`)
    e executa a dinâmica diretamente na árvore física do MuJoCo (`mj_step`):
      J*theta'' + b_ef*theta' + d_a*theta'*|theta'| + Tc*tanh(50*theta') + C_g*sin(theta) + K_s*theta = K_tau*max(a - u0, 0)^p
      da/dt = (u - a) / tau_m
    """

    def __init__(self, json_path: str = GREYBOX_V9_JSON, variant: str = "v9_otimizado"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.variant = variant
        m = data["models"][variant]
        p = m["params"]
        phys = m["physics"]

        self.J = float(phys["J"])          # 0.0407587 kg*m^2
        self.C_g = float(phys["C_g"])      # 0.279536 N*m
        self.K_s = float(phys["K_s"])      # 0.0638036 N*m/rad (rigidez torsional do cabo)
        self.b_ef = float(phys["b_ef"])    # 0.0135097 N*m*s/rad
        self.d_a = float(phys["d_a"])      # 0.0007871 N*m*s^2/rad^2
        self.Tc = float(phys["Tc"])        # 0.0104269 N*m
        self.Ktau = float(phys["Ktau"])    # 0.0005059 N*m/%^p

        self.c1 = float(p["c1"])           # -6.858314 s^-2
        self.c2 = float(p["c2"])           # -0.331455 s^-1
        self.c3 = float(p["c3"])           # 0.012412
        self.ca = float(p["ca"])           # 0.01931
        self.Fc = float(p["Fc"])           # 0.25582
        self.u0 = float(p["u0"])           # 11.852171 %
        self.tau_m = float(p["tau_m"])     # 0.113127 s
        self.p_exp = float(p["p"])         # 1.759243
        self.k_th = float(p["k_th"])       # 1.565398 s^-2

    def configure_mujoco_model(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Sincroniza J, K_s (stiffness), b_ef (damping) e tau_m (dyntype=filter) nativos do MuJoCo com a variante v9."""
        model.jnt_stiffness[0] = self.K_s
        model.dof_damping[0] = self.b_ef
        model.dof_frictionloss[0] = 0.0
        model.dof_armature[0] = 0.0
        model.actuator_dynprm[0, 0] = self.tau_m
        model.actuator_gainprm[0, 0] = 0.0
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulo_movel")
        m_tot = float(model.body_mass[body_id])
        z_cm = -self.C_g / (m_tot * 9.81)
        model.body_ipos[body_id, 2] = z_cm
        i_cm = max(1e-5, self.J - m_tot * (z_cm ** 2))
        model.body_inertia[body_id, 0] = i_cm
        model.body_inertia[body_id, 1] = i_cm
        mujoco.mj_setConst(model, data)

    def equilibrium_u(self, theta_rad: float) -> float:
        """Retorna o comando PWM u (%) de equilíbrio estático para theta_rad (incluindo rigidez do cabo k_th)."""
        if theta_rad <= 0.0:
            return self.u0
        acc_req = abs(self.c1) * np.sin(theta_rad) + self.k_th * theta_rad
        thrust_req = max(0.0, acc_req / self.c3)
        return float(self.u0 + (thrust_req ** (1.0 / self.p_exp)))

    def step_actuator(self, act_state: float, u_cmd: float, dt: float) -> float:
        """Integra a dinâmica de 1ª ordem do motor/EDF: da/dt = (u - a) / tau_m."""
        return float(act_state + ((u_cmd - act_state) / self.tau_m) * dt)

    def compute_mujoco_ctrl_torque(self, theta_rad: float, omega_rad_s: float, act_state: float) -> float:
        """
        Calcula o torque complementar `data.ctrl[0]` (em N·m) para o `mj_step` do MuJoCo.
        Como o MuJoCo já aplica nativamente:
          -C_g * sin(theta) (gravidade física),
          -K_s * theta      (rigidez torsional do cabo na junta `stiffness`),
          -b_ef * omega     (amortecimento viscoso na junta `damping`),
        o atuador `ctrl[0]` aplica o empuxo da turbina EDF, o arrasto aerodinâmico quadrático
        e o atrito seco suave de Coulomb!
        """
        ue = act_state - self.u0 if act_state > self.u0 else 0.0
        ue_s = np.sqrt(ue * ue + 1e-4)
        thrust_torque = self.Ktau * (ue_s ** self.p_exp)
        aero_torque = -self.d_a * omega_rad_s * np.sqrt(omega_rad_s * omega_rad_s + 1e-6)
        fric_torque = -self.Tc * np.tanh(50.0 * omega_rad_s)
        return float(thrust_torque + aero_torque + fric_torque)


# ==============================================================================
# 2. CLASSE DO MODELO NARX POLINOMIAL
# ==============================================================================
class PolynomialNARX:
    """Avaliador genérico de modelos NARX polinomiais a partir de narx_model.json."""

    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.json_path = json_path
        self.ny = int(data["ny"])
        self.nu = int(data["nu"])
        self.terms = data["terms"]
        self.theta = np.array(data["theta"], dtype=float)
        self.decimation = int(data.get("decimation", 1))
        self.ts = float(data.get("Ts", 0.01 * self.decimation))
        sp = data.get("static_and_poles", {})
        self.one_minus_a = float(sp.get("one_minus_a", 0.017674046956128064))
        self.q_y = float(sp.get("q_y", 6.798854103304915e-05))
        self.b_u3 = float(sp.get("b_u3", 6.461157373727681e-06))

        self.compiled_terms = []
        for term in self.terms:
            if term.strip().lower() == "constant":
                self.compiled_terms.append([])
            else:
                factors = []
                for v, lag_str in re.findall(r"([yu])\(k-(\d+)\)", term):
                    factors.append((v, int(lag_str)))
                self.compiled_terms.append(factors)

    def equilibrium_u(self, theta_deg: float) -> float:
        y = float(np.clip(theta_deg, 0.0, 115.0))
        num = self.one_minus_a * y - self.q_y * (y ** 2)
        if num <= 0.0 or self.b_u3 <= 0.0:
            return 0.0
        return float((num / self.b_u3) ** (1.0 / 3.0))

    def step(self, y_hist: np.ndarray, u_hist: np.ndarray) -> float:
        y_k = 0.0
        for th, factors in zip(self.theta, self.compiled_terms):
            prod = 1.0
            for v, lag in factors:
                if v == "y":
                    prod *= y_hist[lag - 1]
                else:
                    prod *= u_hist[lag - 1]
            y_k += th * prod
        return float(y_k)

    def simulate_freerun(self, t_native: np.ndarray, u_native: np.ndarray, y_native_deg: np.ndarray) -> np.ndarray:
        d = self.decimation
        t_dec = t_native[::d]
        u_dec = u_native[::d]
        y_dec = y_native_deg[::d]
        n_dec = len(y_dec)
        ml = max(self.ny, self.nu)

        y_sim_dec = np.zeros(n_dec, dtype=float)
        y_sim_dec[:ml] = y_dec[:ml]

        for k in range(ml, n_dec):
            y_hist = y_sim_dec[k - self.ny : k][::-1]
            u_hist = u_dec[k - self.nu : k][::-1]
            y_sim_dec[k] = self.step(y_hist, u_hist)

        if d == 1:
            return y_sim_dec
        return np.interp(t_native, t_dec, y_sim_dec)


# ==============================================================================
# 3. SIMULAÇÃO FÍSICA NO MOTOR DO MUJOCO (mj_step)
# ==============================================================================
def set_mocap_arm_angle(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, angle_rad: float):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id >= 0:
        mocap_id = model.body_mocapid[body_id]
        if mocap_id >= 0:
            half = 0.5 * angle_rad
            data.mocap_pos[mocap_id] = [0.0, 0.0, 0.470]
            data.mocap_quat[mocap_id] = [np.cos(half), -np.sin(half), 0.0, 0.0]


def update_thrust_jet_visual(model: mujoco.MjModel, u_pct: float):
    frac = max(0.0, u_pct) / 100.0
    for name, scale, y0 in [("jato_empuxo", 0.090, -0.026), ("jato_empuxo_esq", 0.070, -0.026), ("jato_empuxo_dir", 0.070, -0.026)]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            half_len = max(0.002, frac * scale)
            model.geom_size[gid, 1] = half_len
            model.geom_pos[gid, 1] = y0 - half_len


def simulate_mujoco_freerun(model: mujoco.MjModel, gb: GreyBoxV9MuJoCo, t_vec: np.ndarray, u_vec: np.ndarray, y_meas_deg: np.ndarray) -> np.ndarray:
    """
    Simula o Modelo Físico Grey-Box v9 diretamente dentro do motor de física do MuJoCo (`mj_step` RK4)
    usando `mjcb_control` e a dinâmica interna de 1ª ordem `data.act[0]` (`dyntype="filter"`) do MuJoCo,
    de modo que os 4 sub-estágios RK4 de (qpos, qvel, act) sejam avaliados simultaneamente pelo MuJoCo.
    """
    data = mujoco.MjData(model)
    gb.configure_mujoco_model(model, data)
    mujoco.mj_resetData(model, data)

    dt_exp = float(np.mean(np.diff(t_vec)))  # 0.01 s (100 Hz)
    saved_ts = model.opt.timestep
    model.opt.timestep = dt_exp

    data.qpos[0] = np.deg2rad(y_meas_deg[0])
    data.qvel[0] = 0.0
    data.act[0] = float(u_vec[0])

    def _cb_control(m, d):
        d.qfrc_applied[0] = gb.compute_mujoco_ctrl_torque(float(d.qpos[0]), float(d.qvel[0]), float(d.act[0]))

    mujoco.set_mjcb_control(_cb_control)
    try:
        mujoco.mj_forward(model, data)
        n_samples = len(t_vec)
        y_mj_deg = np.zeros(n_samples, dtype=float)
        y_mj_deg[0] = y_meas_deg[0]

        for k in range(n_samples - 1):
            data.ctrl[0] = float(u_vec[k])
            mujoco.mj_step(model, data)
            y_mj_deg[k + 1] = np.rad2deg(data.qpos[0])
    finally:
        mujoco.set_mjcb_control(None)
        model.opt.timestep = saved_ts

    return y_mj_deg


def load_rodada7_dataset(rel_path: str, trim=(10.0, 13.0)):
    fpath = os.path.join(ROOT_DIR, "data", "coletas", rel_path)
    if not os.path.exists(fpath):
        fpath = rel_path
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        y_deg, u_pct, t_sec, ref_deg = readData(fpath, return_ref=True)
    finally:
        sys.stdout = old_stdout
    i0 = np.searchsorted(t_sec, t_sec[0] + trim[0])
    i1 = np.searchsorted(t_sec, t_sec[-1] - trim[1])
    t_c = t_sec[i0:i1] - t_sec[i0]
    return t_c, u_pct[i0:i1].astype(float), y_deg[i0:i1].astype(float), ref_deg[i0:i1].astype(float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.linalg.norm(y_true - np.mean(y_true)))
    fit = float(100.0 * (1.0 - np.linalg.norm(err) / (denom + 1e-12)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    return {"fit": fit, "rmse": rmse, "mae": mae, "r2": r2}


# ==============================================================================
# 4. SIMULAÇÃO EM MALHA FECHADA (RASTREAMENTO + DISTÚRBIO IMPULSIVO)
# ==============================================================================
def run_closed_loop_comparison(model: mujoco.MjModel, gb: GreyBoxV9MuJoCo, narx: PolynomialNARX, duration_s: float = 24.0):
    """
    Teste em malha fechada (24 s) comparando a Planta Física 3D no MuJoCo (Grey-Box v9)
    e a Planta NARX Polinomial:
      - Degraus: 40° (0-6s) -> 55° (6-12s) -> 35° (12-18s) -> 50° (18-24s)
      - Distúrbio impulsivo externo aplicado em t = 9.0 s
    """
    dt_ctrl = 0.01  # 100 Hz
    t_vec = np.arange(0.0, duration_s, dt_ctrl)
    n_steps = len(t_vec)

    ref_deg = np.zeros(n_steps)
    for k, t in enumerate(t_vec):
        if t < 6.0:
            ref_deg[k] = 40.0
        elif t < 12.0:
            ref_deg[k] = 55.0
        elif t < 18.0:
            ref_deg[k] = 35.0
        else:
            ref_deg[k] = 50.0

    # --- 4.1 Simulação na Planta Física 3D do MuJoCo (Grey-Box v9) ---
    data = mujoco.MjData(model)
    gb.configure_mujoco_model(model, data)
    mujoco.mj_resetData(model, data)

    th0_rad = np.deg2rad(40.0)
    data.qpos[0] = th0_rad
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    u_eq_40 = gb.equilibrium_u(th0_rad)
    act_state = float(u_eq_40)
    int_err_mj = 0.0
    kp, ki, kd = 0.70, 0.95, 0.16

    dt_mj = model.opt.timestep
    n_sub = max(1, int(round(dt_ctrl / dt_mj)))

    y_mj_cl = np.zeros(n_steps)
    u_mj_cl = np.zeros(n_steps)

    for k in range(n_steps):
        th_deg = np.rad2deg(data.qpos[0])
        om_deg_s = np.rad2deg(data.qvel[0])
        y_mj_cl[k] = th_deg

        err = ref_deg[k] - th_deg
        int_err_mj = np.clip(int_err_mj + err * dt_ctrl, -25.0, 25.0)
        u_ff = gb.equilibrium_u(np.deg2rad(max(0.0, ref_deg[k])))
        u_cmd = float(np.clip(u_ff + kp * err + ki * int_err_mj - kd * om_deg_s, 0.0, 100.0))
        u_mj_cl[k] = u_cmd

        tau_dist = -0.18 if (9.0 <= t_vec[k] <= 9.15) else 0.0

        for _ in range(n_sub):
            th = float(data.qpos[0])
            om = float(data.qvel[0])
            data.qfrc_applied[0] = gb.compute_mujoco_ctrl_torque(th, om, act_state) + tau_dist
            mujoco.mj_step(model, data)
            act_state = gb.step_actuator(act_state, u_cmd, dt_mj)

    # --- 4.2 Simulação na Planta NARX Polinomial (na taxa Ts = 80 ms do NARX) ---
    d = narx.decimation
    dt_narx = d * dt_ctrl
    t_narx = t_vec[::d]
    ref_narx = ref_deg[::d]
    n_narx = len(t_narx)
    ml = max(narx.ny, narx.nu)

    y_narx_dec = np.full(n_narx, 40.0)
    u_narx_dec = np.full(n_narx, narx.equilibrium_u(40.0))
    int_err_nx = 0.0
    kp_d, ki_d, kd_d = 0.14, 0.25, 0.045

    for k in range(ml, n_narx):
        y_hist = y_narx_dec[k - narx.ny : k][::-1]
        u_hist = u_narx_dec[k - narx.nu : k][::-1]
        yk = narx.step(y_hist, u_hist)
        if 9.0 <= t_narx[k] < 9.0 + dt_narx:
            yk -= 6.5
        yk = float(np.clip(yk, -20.0, 115.0))
        y_narx_dec[k] = yk

        err = ref_narx[k] - yk
        om_est = (y_narx_dec[k] - y_narx_dec[k - 1]) / dt_narx
        int_err_nx = np.clip(int_err_nx + err * dt_narx, -15.0, 15.0)
        u_ff = narx.equilibrium_u(ref_narx[k])
        u_narx_dec[k] = float(np.clip(u_ff + kp_d * err + ki_d * int_err_nx - kd_d * om_est, 0.0, 65.0))

    y_narx_cl = np.interp(t_vec, t_narx, y_narx_dec)
    u_narx_cl = np.interp(t_vec, t_narx, u_narx_dec)

    return {
        "t": t_vec,
        "ref": ref_deg,
        "y_mujoco": y_mj_cl,
        "u_mujoco": u_mj_cl,
        "y_narx": y_narx_cl,
        "u_narx": u_narx_cl,
    }


# ==============================================================================
# 5. BENCHMARK COMPLETO, PAINEL GRÁFICO E VÍDEO 3D LADO A LADO
# ==============================================================================
def run_full_benchmark_and_plots(narx_path: str = NARX_SR_FROLS_JSON, gb_variant: str = "v9_otimizado", render_video: bool = True):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    gb = GreyBoxV9MuJoCo(GREYBOX_V9_JSON, variant=gb_variant)
    narx = PolynomialNARX(narx_path)

    print("=" * 98)
    print(f"BENCHMARK NO MUJOCO: MODELO FÍSICO GREY-BOX v9 ({gb.variant}) vs. MODELO NARX ({os.path.basename(narx_path)})")
    print(f"  - Físico MuJoCo v9: J={gb.J:.5f} kg·m², K_s={gb.K_s:.5f} N·m/rad, b_ef={gb.b_ef:.5f}, p={gb.p_exp:.3f}, u0={gb.u0:.2f}%")
    print(f"  - NARX Polinomial:  ny={narx.ny}, nu={narx.nu}, Ts={narx.ts*1000:.0f} ms (decimation={narx.decimation}), {len(narx.terms)} termos")
    print("=" * 98)
    print(f"{'Ensaio de Validação (RODADA-7)':32s} | {'Físico MuJoCo v9 (FIT / RMSE)':30s} | {'NARX Free-Run (FIT / RMSE)':30s}")
    print("-" * 98)

    results_freerun = []
    fits_mj, rmses_mj = [], []
    fits_nx, rmses_nx = [], []

    for key, rel_csv, label in VAL_DATASETS:
        t_c, u_c, y_c, ref_c = load_rodada7_dataset(rel_csv)
        y_mj = simulate_mujoco_freerun(model, gb, t_c, u_c, y_c)
        y_nx = narx.simulate_freerun(t_c, u_c, y_c)

        m_mj = compute_metrics(y_c, y_mj)
        m_nx = compute_metrics(y_c, y_nx)
        fits_mj.append(m_mj["fit"])
        rmses_mj.append(m_mj["rmse"])
        fits_nx.append(m_nx["fit"])
        rmses_nx.append(m_nx["rmse"])

        print(
            f"{label:32s} | FIT: {m_mj['fit']:5.1f}%  RMSE: {m_mj['rmse']:5.2f}°     | "
            f"FIT: {m_nx['fit']:5.1f}%  RMSE: {m_nx['rmse']:5.2f}°"
        )
        results_freerun.append({
            "key": key, "label": label, "t": t_c, "u": u_c, "y_real": y_c,
            "y_mujoco": y_mj, "y_narx": y_nx, "m_mj": m_mj, "m_nx": m_nx
        })

    print("-" * 98)
    print(
        f"{'MÉDIA 45° (3 ENSAIOS)':32s} | FIT: {np.mean(fits_mj[:3]):5.1f}%  RMSE: {np.mean(rmses_mj[:3]):5.2f}°     | "
        f"FIT: {np.mean(fits_nx[:3]):5.1f}%  RMSE: {np.mean(rmses_nx[:3]):5.2f}°"
    )
    print(
        f"{'MÉDIA 60° (3 ENSAIOS)':32s} | FIT: {np.mean(fits_mj[3:]):5.1f}%  RMSE: {np.mean(rmses_mj[3:]):5.2f}°     | "
        f"FIT: {np.mean(fits_nx[3:]):5.1f}%  RMSE: {np.mean(rmses_nx[3:]):5.2f}°"
    )
    print(
        f"{'MÉDIA GERAL (6 ENSAIOS)':32s} | FIT: {np.mean(fits_mj):5.1f}%  RMSE: {np.mean(rmses_mj):5.2f}°     | "
        f"FIT: {np.mean(fits_nx):5.1f}%  RMSE: {np.mean(rmses_nx):5.2f}°"
    )
    print("=" * 98)

    cl_res = run_closed_loop_comparison(model, gb, narx)

    # --- Plotagem do Painel Comparativo Completo ---
    fig = plt.figure(figsize=(15.5, 11.5), facecolor="#0d1117", dpi=150)
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.15], hspace=0.36, wspace=0.16,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    fig.suptitle(
        f"Gêmeo Digital MuJoCo: Modelo Físico Grey-Box v9 (FIT Médio {np.mean(fits_mj):.1f}%) vs. NARX (FIT Médio {np.mean(fits_nx):.1f}%)",
        color="#e6edf3", fontsize=13.0, fontweight="bold", fontfamily="monospace"
    )

    for idx, res in enumerate(results_freerun):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.grid(True, linestyle="--", color="#21262d", alpha=0.8)

        ax.plot(res["t"], res["y_real"], color="#f0883e", lw=1.4, alpha=0.9, label="Bancada Real (IMU)")
        ax.plot(res["t"], res["y_mujoco"], color="#58a6ff", lw=1.6, label=f"Físico MuJoCo v9 (FIT={res['m_mj']['fit']:.1f}%, RMSE={res['m_mj']['rmse']:.1f}°)")
        ax.plot(res["t"], res["y_narx"], color="#a855f7", lw=1.4, ls="--", label=f"NARX Free-Run (FIT={res['m_nx']['fit']:.1f}%, RMSE={res['m_nx']['rmse']:.1f}°)")

        ax.set_title(f"Free-Run: {res['label']}", color="#e6edf3", fontsize=10, fontweight="bold", fontfamily="monospace")
        ax.set_ylabel("Ângulo (°)", color="#c9d1d9", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8.5)
        ax.legend(loc="upper right", fontsize=7.8, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9")

    ax_cl_y = fig.add_subplot(gs[3, 0])
    ax_cl_u = fig.add_subplot(gs[3, 1])
    for ax in (ax_cl_y, ax_cl_u):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.grid(True, linestyle="--", color="#21262d", alpha=0.8)
        ax.tick_params(colors="#8b949e", labelsize=8.5)

    ax_cl_y.step(cl_res["t"], cl_res["ref"], where="post", color="#f0883e", lw=1.5, ls="--", label="Referência θ_ref")
    ax_cl_y.plot(cl_res["t"], cl_res["y_mujoco"], color="#58a6ff", lw=1.8, label="Planta Física 3D MuJoCo (Grey-Box v9)")
    ax_cl_y.plot(cl_res["t"], cl_res["y_narx"], color="#a855f7", lw=1.6, ls="-.", label="Planta NARX Polinomial")
    ax_cl_y.axvline(9.0, color="#da3633", lw=1.2, ls=":", label="Distúrbio Impulsivo (t = 9s)")
    ax_cl_y.set_title("Malha Fechada: Rastreamento de Degraus + Rejeição a Distúrbio", color="#e6edf3", fontsize=10, fontweight="bold", fontfamily="monospace")
    ax_cl_y.set_xlabel("Tempo (s)", color="#c9d1d9", fontsize=9)
    ax_cl_y.set_ylabel("Ângulo (°)", color="#c9d1d9", fontsize=9)
    ax_cl_y.legend(loc="lower right", fontsize=7.8, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9")

    ax_cl_u.plot(cl_res["t"], cl_res["u_mujoco"], color="#3fb950", lw=1.6, label="Controle u(t) no Físico MuJoCo v9 (%)")
    ax_cl_u.plot(cl_res["t"], cl_res["u_narx"], color="#a855f7", lw=1.4, ls="--", label="Controle u(t) no NARX (%)")
    ax_cl_u.axvline(9.0, color="#da3633", lw=1.2, ls=":", label="Distúrbio Impulsivo (t = 9s)")
    ax_cl_u.set_title("Malha Fechada: Ação de Controle na Turbina EDF u(t)", color="#e6edf3", fontsize=10, fontweight="bold", fontfamily="monospace")
    ax_cl_u.set_xlabel("Tempo (s)", color="#c9d1d9", fontsize=9)
    ax_cl_u.set_ylabel("PWM u (%)", color="#c9d1d9", fontsize=9)
    ax_cl_u.legend(loc="lower right", fontsize=7.8, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9")

    out_png = os.path.join(BASE_DIR, "comparativo_mujoco_fisico_vs_narx.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"-> Painel comparativo salvo em: {out_png}")

    if render_video:
        render_side_by_side_video(model, results_freerun[2])

    return results_freerun, cl_res


def render_side_by_side_video(model: mujoco.MjModel, res_exp: dict):
    """Gera vídeo MP4 e GIF Lado a Lado (Bancada 3D MuJoCo com os 3 braços + Gráficos ao vivo)."""
    t = res_exp["t"]
    u = res_exp["u"]
    y_real = res_exp["y_real"]
    y_mj = res_exp["y_mujoco"]
    y_nx = res_exp["y_narx"]

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=640, width=640)

    fig = plt.figure(figsize=(6.4, 6.4), facecolor="#0d1117", dpi=100)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.3, 1.2], hspace=0.18, left=0.14, right=0.96, top=0.91, bottom=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.grid(True, linestyle="--", color="#21262d", alpha=0.8)
        ax.tick_params(colors="#8b949e", labelsize=8.5)

    ax1.plot(t, y_real, color="#f0883e", lw=1.0, alpha=0.30)
    ax1.plot(t, y_mj, color="#58a6ff", lw=1.0, alpha=0.25)
    ax1.plot(t, y_nx, color="#a855f7", lw=1.0, ls="--", alpha=0.25)

    (ln_real,) = ax1.plot([], [], color="#f0883e", lw=1.8, label="Bancada Real (Braço Laranja)")
    (ln_mj,) = ax1.plot([], [], color="#58a6ff", lw=2.0, label=f"Físico MuJoCo v9 3D (FIT {res_exp['m_mj']['fit']:.1f}%)")
    (ln_nx,) = ax1.plot([], [], color="#a855f7", lw=1.8, ls="--", label=f"NARX Free-Run (Braço Roxo, FIT {res_exp['m_nx']['fit']:.1f}%)")
    (pt_mj,) = ax1.plot([], [], "o", color="#ffffff", markeredgecolor="#58a6ff", markeredgewidth=2.0, markersize=6)
    (pt_nx,) = ax1.plot([], [], "o", color="#ffffff", markeredgecolor="#a855f7", markeredgewidth=2.0, markersize=5)

    ax1.set_title(f"MuJoCo Físico v9 vs. NARX vs. Real — {res_exp['label']}", color="#e6edf3", fontsize=9.5, fontweight="bold", fontfamily="monospace")
    ax1.set_ylabel("Ângulo (°)", color="#c9d1d9", fontsize=9.5)
    ax1.set_xlim(0, t[-1])
    y_min = min(np.min(y_real), np.min(y_mj), np.min(y_nx)) - 4.0
    y_max = max(np.max(y_real), np.max(y_mj), np.max(y_nx)) + 8.0
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc="upper right", fontsize=7.8, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9")

    ax2.plot(t, u, color="#3fb950", lw=1.0, alpha=0.25)
    (ln_u,) = ax2.plot([], [], color="#3fb950", lw=1.8, label="Entrada da Turbina EDF u(t)")
    (pt_u,) = ax2.plot([], [], "o", color="#ffffff", markeredgecolor="#3fb950", markeredgewidth=2.0, markersize=5)
    ax2.set_xlabel("Tempo (s)", color="#c9d1d9", fontsize=9.5)
    ax2.set_ylabel("Controle u (%)", color="#c9d1d9", fontsize=9.5)
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(min(0.0, np.min(u) - 5.0), np.max(u) + 8.0)
    ax2.legend(loc="upper right", fontsize=7.8, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9")

    fig.canvas.draw()
    bg_static = fig.canvas.copy_from_bbox(fig.bbox)

    step_mp4 = 8
    indices = list(range(step_mp4 - 1, len(t), step_mp4))
    if indices[-1] != len(t) - 1:
        indices.append(len(t) - 1)
    fps = 100.0 / step_mp4

    out_mp4 = os.path.join(BASE_DIR, "teste_mujoco_fisico_vs_narx_live.mp4")
    out_gif = os.path.join(BASE_DIR, "teste_mujoco_fisico_vs_narx_live.gif")

    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=2)
    frames_gif = []
    prop_angle = 0.0

    print(f"Renderizando vídeo Lado a Lado (3D MuJoCo Físico v9 + NARX + Real) ({len(indices)} quadros)...")
    t0 = time.time()

    for count, idx in enumerate(indices):
        data.qpos[0] = np.deg2rad(y_mj[idx])
        prop_angle = (prop_angle + (35.0 + 4.5 * max(0.0, u[idx])) * (1.0 / fps)) % (2.0 * np.pi)
        data.qpos[1] = prop_angle

        update_thrust_jet_visual(model, u[idx])
        set_mocap_arm_angle(model, data, "referencia_visual", np.deg2rad(y_real[idx]))
        set_mocap_arm_angle(model, data, "narx_visual", np.deg2rad(y_nx[idx]))
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera="isometrica_cad")
        rgb_3d = renderer.render()

        fig.canvas.restore_region(bg_static)
        t_slice = t[: idx + 1]
        ln_real.set_data(t_slice, y_real[: idx + 1])
        ln_mj.set_data(t_slice, y_mj[: idx + 1])
        ln_nx.set_data(t_slice, y_nx[: idx + 1])
        pt_mj.set_data([t[idx]], [y_mj[idx]])
        pt_nx.set_data([t[idx]], [y_nx[idx]])

        ln_u.set_data(t_slice, u[: idx + 1])
        pt_u.set_data([t[idx]], [u[idx]])

        for artist in (ln_real, ln_mj, ln_nx, pt_mj, pt_nx):
            ax1.draw_artist(artist)
        for artist in (ln_u, pt_u):
            ax2.draw_artist(artist)

        fig.canvas.blit(fig.bbox)
        rgba_plot = np.asarray(fig.canvas.buffer_rgba())
        rgb_plot = rgba_plot[:640, :640, :3].copy()

        frame_combined = np.concatenate([rgb_3d, rgb_plot], axis=1)
        writer.append_data(frame_combined)

        if count % 2 == 0 or idx == indices[-1]:
            pil_half = Image.fromarray(frame_combined).resize((768, 384), Image.Resampling.LANCZOS)
            frames_gif.append(pil_half.quantize(colors=64, method=Image.Quantize.FASTOCTREE))

    writer.close()
    renderer.close()
    plt.close(fig)
    print(f"-> Vídeo Lado a Lado salvo: {out_mp4} ({os.path.getsize(out_mp4)/1024:.1f} KB) em {time.time()-t0:.1f}s")

    if frames_gif:
        frames_gif[0].save(out_gif, save_all=True, append_images=frames_gif[1:], duration=120, loop=0, optimize=True)
        print(f"-> GIF Lado a Lado salvo: {out_gif} ({os.path.getsize(out_gif)/1024:.1f} KB)")


def launch_dual_viewer(narx_path: str = NARX_SR_FROLS_JSON, gb_variant: str = "v9_otimizado"):
    """Abre a janela 3D interativa do MuJoCo rodando Modelo Físico Grey-Box v9 + Modelo NARX em paralelo."""
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    gb = GreyBoxV9MuJoCo(GREYBOX_V9_JSON, variant=gb_variant)
    data = mujoco.MjData(model)
    gb.configure_mujoco_model(model, data)
    narx = PolynomialNARX(narx_path)

    th0_deg = 40.0
    data.qpos[0] = np.deg2rad(th0_deg)
    mujoco.mj_forward(model, data)

    u_eq = gb.equilibrium_u(np.deg2rad(th0_deg))
    act_state = float(u_eq)
    int_err = 0.0
    kp, ki, kd = 0.70, 0.95, 0.16

    y_hist_nx = np.full(narx.ny, th0_deg, dtype=float)
    u_hist_nx = np.full(narx.nu, narx.equilibrium_u(th0_deg), dtype=float)
    y_nx_curr = th0_deg
    last_narx_time = 0.0

    degraus = [(0.0, 40.0), (6.0, 55.0), (12.0, 35.0), (18.0, 50.0)]

    print("Abrindo MuJoCo Viewer 3D Comparativo (Físico Grey-Box v9 + Fantasma Roxo NARX + Fantasma Laranja Ref)...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 132
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.28
        viewer.cam.lookat[:] = [0.0, 0.0, 0.42]

        while viewer.is_running():
            t0 = time.time()
            t_sim = data.time
            dt = model.opt.timestep

            t_mod = t_sim % 24.0
            ref_deg = 40.0
            for t_sw, r_v in degraus:
                if t_mod >= t_sw:
                    ref_deg = r_v

            th_deg = np.rad2deg(data.qpos[0])
            om_deg_s = np.rad2deg(data.qvel[0])

            err = ref_deg - th_deg
            int_err = np.clip(int_err + err * dt, -25.0, 25.0)
            u_ff = gb.equilibrium_u(np.deg2rad(max(0.0, ref_deg)))
            u_cmd = float(np.clip(u_ff + kp * err + ki * int_err - kd * om_deg_s, 0.0, 100.0))

            update_thrust_jet_visual(model, act_state)
            data.qfrc_applied[0] = gb.compute_mujoco_ctrl_torque(data.qpos[0], data.qvel[0], act_state)
            data.ctrl[1] = 40.0 + 3.5 * act_state
            act_state = gb.step_actuator(act_state, u_cmd, dt)

            if t_sim - last_narx_time >= narx.ts:
                y_nx_curr = narx.step(y_hist_nx, u_hist_nx)
                y_hist_nx = np.roll(y_hist_nx, 1)
                y_hist_nx[0] = y_nx_curr
                u_hist_nx = np.roll(u_hist_nx, 1)
                u_hist_nx[0] = u_cmd
                last_narx_time = t_sim

            set_mocap_arm_angle(model, data, "referencia_visual", np.deg2rad(ref_deg))
            set_mocap_arm_angle(model, data, "narx_visual", np.deg2rad(y_nx_curr))

            mujoco.mj_step(model, data)
            viewer.sync()

            el = time.time() - t0
            if el < dt:
                time.sleep(dt - el)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MuJoCo: Modelo Físico Grey-Box v9 vs. Modelo NARX")
    parser.add_argument("--narx", type=str, default=NARX_SR_FROLS_JSON, help="Caminho para narx_model.json")
    parser.add_argument("--greybox", type=str, default="v9_otimizado", help="Variante do modelo_greybox_v9.json (v9_otimizado, v9_completo_r1, v9_3a_sem_cabo)")
    parser.add_argument("--viewer", action="store_true", help="Abre o visualizador 3D interativo rodando Físico v9 + NARX simultaneamente")
    parser.add_argument("--no-video", action="store_true", help="Pula a renderização do vídeo MP4/GIF lado a lado")
    args = parser.parse_args()

    if args.viewer:
        launch_dual_viewer(args.narx, args.greybox)
    else:
        run_full_benchmark_and_plots(args.narx, args.greybox, render_video=not args.no_video)
