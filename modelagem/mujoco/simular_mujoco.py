"""
simular_mujoco.py — Simulador e Visualizador 3D do Aeropêndulo no MuJoCo.

Funcionalidades:
1. Visualizador 3D Interativo em tempo real (`python simular_mujoco.py --viewer`)
   com física não-linear identificada (gravidade, atrito viscoso + Coulomb, arrasto
   aerodinâmico, dinâmica de 1ª ordem do motor BLDC e controlador PID/MLP/Replay).
2. Replay 3D de ensaios experimentais reais em CSV (`python simular_mujoco.py --replay <csv>`)
   exportando vídeo MP4 e GIF em alta resolução do modelo 3D da bancada.
3. Geração de imagens de pré-visualização das 4 câmeras do CAD (`python simular_mujoco.py --preview`).
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import imageio
from PIL import Image
import mujoco

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "aeropendulo.xml")
CSV_VAL1_DEFAULT = r"c:\Users\vicio\Documents\AEROPENDULO\data\coletas\RODADA_20261117\20261117_182241_referencia_legado_val1_mpc_hybrid.csv"

# Parâmetros físicos identificados da bancada (TCC / protocolo_treinamento_v6.md / test_damping.py)
J_EFF = 0.0335       # Momento de inércia efetivo (kg·m²)
C1_GRAV = -8.34      # Termo gravitacional (s^-2) -> tau_g = J * |c1| = 0.2794 N·m
C2_VISC = -0.15      # Amortecimento viscoso (s^-1)
C3_THRUST = 0.00420  # Ganho quadrático do motor (rad / (s^2 · %^2))
FC_COULOMB = 0.139   # Atrito seco de Coulomb (rad/s^2)
CA_AERO = 0.12       # Arrasto aerodinâmico quadrático
U0_DEADZONE = 11.26  # Zona morta do ESC (%)
TAU_MOTOR = 0.08     # Constante de tempo do conjunto ESC + motor BLDC (s)


def set_ghost_reference_angle(model: mujoco.MjModel, data: mujoco.MjData, ref_rad: float):
    """Atualiza a orientação do braço fantasma (laranja translúcido) que indica a referência angular."""
    mocap_id = model.body("referencia_visual").mocapid[0]
    if mocap_id >= 0:
        # Rotação de ref_rad em torno do eixo -X: quat = [cos(theta/2), -sin(theta/2), 0, 0]
        half = 0.5 * ref_rad
        data.mocap_pos[mocap_id] = [0.0, 0.0, 0.470]
        data.mocap_quat[mocap_id] = [np.cos(half), -np.sin(half), 0.0, 0.0]


def update_thrust_jet_visual(model: mujoco.MjModel, u_pct: float):
    """Escala dinamicamente os 3 filetes visuais do jato colimado da turbina EDF conforme u_pct (%)."""
    frac = max(0.0, u_pct) / 100.0
    for name, scale, y0 in [("jato_empuxo", 0.090, -0.026), ("jato_empuxo_esq", 0.070, -0.026), ("jato_empuxo_dir", 0.070, -0.026)]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            half_len = max(0.002, frac * scale)
            model.geom_size[gid, 1] = half_len
            model.geom_pos[gid, 1] = y0 - half_len


def calcular_torque_identificado(theta_rad: float, omega_rad_s: float, act_pct: float, model: mujoco.MjModel) -> float:
    """
    Calcula o torque externo líquido no eixo do pêndulo (em N·m) para que a equação
    diferencial do MuJoCo reproduza exatamente o modelo grey-box identificado da bancada.
    Nota: O MuJoCo já aplica o torque gravitacional físico das massas (m1*L1 - m2*L2)*g*sin(theta)
    e o damping nativo da junta. Aqui aplicamos o torque propulsivo + atrito não-linear complementar.
    """
    thrust_acc = C3_THRUST * (max(act_pct - U0_DEADZONE, 0.0) ** 2)
    fric_acc = -FC_COULOMB * np.tanh(50.0 * omega_rad_s)
    aero_acc = -CA_AERO * omega_rad_s * abs(omega_rad_s)
    visc_acc = C2_VISC * omega_rad_s

    # Compensa o damping linear já presente na junta do XML para evitar dupla contagem
    jnt_damping = model.dof_damping[0]
    tau_net = J_EFF * (thrust_acc + fric_acc + aero_acc + visc_acc) + jnt_damping * omega_rad_s
    return float(tau_net)


def render_preview_cameras(out_dir: str = BASE_DIR):
    """Renderiza as 4 vistas da bancada 3D no MuJoCo (Isométrica, Frontal, Lateral e Superior)."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Coloca o pêndulo em 35° e a referência em 45° para uma foto ilustrativa
    data.qpos[0] = np.deg2rad(35.0)
    set_ghost_reference_angle(model, data, np.deg2rad(45.0))
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)
    cameras = ["isometrica_cad", "frontal_bancada", "lateral_plano_pendulo", "superior_bancada"]
    saved = []

    for cam in cameras:
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        out_path = os.path.join(out_dir, f"preview_{cam}.png")
        Image.fromarray(rgb).save(out_path)
        saved.append(out_path)
        print(f"-> Imagem renderizada ({cam}): {out_path}")

    renderer.close()
    return saved


def replay_csv_to_video(
    csv_path: str = CSV_VAL1_DEFAULT,
    camera: str = "isometrica_cad",
    out_mp4: str = None,
    out_gif: str = None,
    step_mp4: int = 4,
    step_gif: int = 20,
    width: int = 960,
    height: int = 720,
):
    """
    Reproduz um ensaio experimental real (CSV) no modelo 3D do MuJoCo,
    animando o braço do aeropêndulo, a referência fantasma e a rotação da hélice bipá.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    t = df["tempo_ms"].values.astype(float) / 1000.0
    ang_deg = df["angulo_deg"].values.astype(float)
    u_pct = df["u_pct"].values.astype(float)
    ref_deg = df["referencia"].values.astype(float)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    if out_mp4 is None:
        out_mp4 = os.path.join(BASE_DIR, f"aeropendulo_mujoco_val1_{camera}.mp4")
    if out_gif is None:
        out_gif = os.path.join(BASE_DIR, f"aeropendulo_mujoco_val1_{camera}.gif")

    # 1. Renderizar MP4 1.0x (25 fps se step_mp4=4)
    indices_mp4 = list(range(step_mp4 - 1, len(t), step_mp4))
    if indices_mp4[-1] != len(t) - 1:
        indices_mp4.append(len(t) - 1)
    fps_mp4 = 100.0 / step_mp4
    dt_frame = 1.0 / fps_mp4

    writer = imageio.get_writer(
        out_mp4, fps=fps_mp4, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=2
    )
    frames_gif = []
    gif_set = set(range(step_gif - 1, len(t), step_gif))
    gif_set.add(len(t) - 1)

    prop_angle = 0.0
    t0 = time.time()
    print(f"Renderizando replay 3D no MuJoCo ({len(indices_mp4)} quadros, câmera='{camera}')...")

    for idx in indices_mp4:
        th_rad = np.deg2rad(ang_deg[idx])
        ref_rad = np.deg2rad(ref_deg[idx])
        u_val = max(0.0, u_pct[idx])

        # Gira a hélice proporcionalmente ao comando PWM (%)
        omega_prop = 35.0 + 4.5 * u_val  # rad/s visual
        prop_angle = (prop_angle + omega_prop * dt_frame) % (2.0 * np.pi)

        data.qpos[0] = th_rad
        data.qpos[1] = prop_angle
        update_thrust_jet_visual(model, u_val)
        set_ghost_reference_angle(model, data, ref_rad)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        writer.append_data(rgb)

        if idx in gif_set:
            pil_img = Image.fromarray(rgb).resize((width // 2, height // 2), Image.Resampling.LANCZOS)
            frames_gif.append(pil_img.quantize(colors=64, method=Image.Quantize.FASTOCTREE))

    writer.close()
    renderer.close()
    print(f"-> MP4 3D MuJoCo salvo: {out_mp4} ({os.path.getsize(out_mp4)/1024:.1f} KB) em {time.time()-t0:.1f}s")

    if frames_gif:
        frames_gif[0].save(
            out_gif, save_all=True, append_images=frames_gif[1:],
            duration=40, loop=0, optimize=True
        )
        print(f"-> GIF 3D MuJoCo salvo: {out_gif} ({os.path.getsize(out_gif)/1024:.1f} KB)")


def run_interactive_viewer(csv_path: str = None):
    """
    Abre a janela interativa 3D do MuJoCo (`mujoco.viewer`).
    Se `csv_path` for passado, reproduz os comandos/trajetória do CSV;
    caso contrário, roda a simulação física em malha fechada com controlador PID
    seguindo uma sequência de degraus (30° -> 45° -> 60° -> 25°).
    """
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Estado interno do motor BLDC (dinâmica de 1ª ordem) e integrador PID
    act_state = 25.0
    int_err = 0.0
    kp, ki, kd = 1.45, 1.85, 0.35

    # Sequência de referências em graus
    degraus = [(0.0, 30.0), (8.0, 45.0), (16.0, 60.0), (24.0, 30.0), (32.0, 50.0)]

    print("Abrindo MuJoCo Interactive Viewer 3D...")
    print("Dica: Use o botão direito/esquerdo do mouse para girar/mover a câmera, ou selecione as câmeras na aba 'Rendering'.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.cam.distance = 1.25
        viewer.cam.lookat[:] = [0.0, 0.0, 0.38]

        t_start = time.time()
        while viewer.is_running():
            step_start = time.time()
            t_sim = data.time

            # Define referência angular atual
            t_mod = t_sim % 40.0
            ref_deg = 30.0
            for t_switch, r_val in degraus:
                if t_mod >= t_switch:
                    ref_deg = r_val
            ref_rad = np.deg2rad(ref_deg)
            set_ghost_reference_angle(model, data, ref_rad)

            # Leitura dos estados no MuJoCo
            theta = data.qpos[0]
            omega = data.qvel[0]

            # Controlador PID + Feedforward gravitacional
            err_deg = np.rad2deg(ref_rad - theta)
            dt = model.opt.timestep
            int_err = np.clip(int_err + err_deg * dt, -35.0, 35.0)
            u_ff = U0_DEADZONE + np.sqrt(max(0.0, abs(C1_GRAV) * np.sin(max(0.0, ref_rad)) / C3_THRUST))
            u_cmd = u_ff + kp * err_deg + ki * int_err - kd * np.rad2deg(omega)
            u_cmd = float(np.clip(u_cmd, 0.0, 100.0))

            # Dinâmica de 1ª ordem do motor BLDC
            act_state += ((u_cmd - act_state) / TAU_MOTOR) * dt

            # Aplica torque propulsivo + atritos identificados e gira a hélice
            update_thrust_jet_visual(model, act_state)
            data.qfrc_applied[0] = calcular_torque_identificado(theta, omega, act_state, model)
            data.ctrl[1] = 40.0 + 3.0 * act_state

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador e Visualizador 3D do Aeropêndulo no MuJoCo")
    parser.add_argument("--preview", action="store_true", help="Renderiza PNGs das 4 câmeras do modelo 3D")
    parser.add_argument("--replay", type=str, nargs="?", const=CSV_VAL1_DEFAULT, help="Gera vídeo MP4/GIF 3D de um CSV experimental")
    parser.add_argument("--camera", type=str, default="isometrica_cad", help="Nome da câmera (isometrica_cad, lateral_plano_pendulo, frontal_bancada, superior_bancada)")
    parser.add_argument("--viewer", action="store_true", help="Abre a janela 3D interativa do MuJoCo em tempo real")
    args = parser.parse_args()

    if args.preview:
        render_preview_cameras()
    elif args.replay:
        replay_csv_to_video(csv_path=args.replay, camera=args.camera)
    elif args.viewer:
        run_interactive_viewer()
    else:
        render_preview_cameras()
        replay_csv_to_video(csv_path=CSV_VAL1_DEFAULT, camera="isometrica_cad")
        replay_csv_to_video(csv_path=CSV_VAL1_DEFAULT, camera="lateral_plano_pendulo")
