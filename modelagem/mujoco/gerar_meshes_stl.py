"""
gerar_meshes_stl.py — Gera os arquivos STL exatos das peças 3D do Aeropêndulo:
1. Torres piramidais triangulares de MDF (torre_esquerda.stl, torre_direita.stl)
2. Duto aerodinâmico da turbina EDF FMS 50mm com bocal bellmouth, anéis externos,
   estatores traseiros e berço de fixação (edf_duto.stl)
3. Rotor de 11 pás torsidas com ogiva cônica central (edf_rotor_11pas.stl)
"""

import os
import struct
import numpy as np


def write_binary_stl(filepath: str, triangles: list):
    """Salva uma lista de triângulos [(v0, v1, v2), ...] em formato STL binário."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = b"MuJoCo Aeropendulo EDF CAD Mesh" + b" " * (80 - 31)
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for v0, v1, v2 in triangles:
            p0 = np.array(v0, dtype=np.float32)
            p1 = np.array(v1, dtype=np.float32)
            p2 = np.array(v2, dtype=np.float32)
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal = normal / norm
            else:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *p0))
            f.write(struct.pack("<3f", *p1))
            f.write(struct.pack("<3f", *p2))
            f.write(struct.pack("<H", 0))


def add_quad(tris: list, p0, p1, p2, p3):
    tris.append((p0, p1, p2))
    tris.append((p0, p2, p3))


def build_tower_stl(filepath: str, sign_x: float = 1.0):
    """Gera a malha 3D da torre piramidal triangular de MDF."""
    tris = []
    x_ext = -sign_x * 0.165
    x_in_b = sign_x * 0.012
    y_half = 0.0922

    b_back_l = (x_ext, -y_half, 0.0)
    b_back_r = (x_ext, +y_half, 0.0)
    b_front = (x_in_b, 0.0, 0.0)

    x_top_ext = -sign_x * 0.015
    x_top_in = sign_x * 0.015
    y_top = 0.015
    z_top = 0.440

    t_back_l = (x_top_ext, -y_top, z_top)
    t_back_r = (x_top_ext, +y_top, z_top)
    t_front_l = (x_top_in, -y_top, z_top)
    t_front_r = (x_top_in, +y_top, z_top)

    if sign_x > 0:
        add_quad(tris, b_back_r, b_back_l, t_back_l, t_back_r)
        add_quad(tris, b_back_l, b_front, t_front_l, t_back_l)
        add_quad(tris, b_front, b_back_r, t_back_r, t_front_r)
        tris.append((b_front, t_front_r, t_front_l))
        add_quad(tris, t_back_l, t_front_l, t_front_r, t_back_r)
        tris.append((b_back_l, b_back_r, b_front))
    else:
        add_quad(tris, b_back_l, b_back_r, t_back_r, t_back_l)
        add_quad(tris, b_front, b_back_l, t_back_l, t_front_l)
        add_quad(tris, b_back_r, b_front, t_front_r, t_back_r)
        tris.append((b_front, t_front_l, t_front_r))
        add_quad(tris, t_back_r, t_front_r, t_front_l, t_back_l)
        tris.append((b_back_r, b_back_l, b_front))

    write_binary_stl(filepath, tris)


def build_edf_duct_stl(filepath: str):
    """
    Gera a carcaça cilíndrica (shroud) da turbina EDF FMS 50mm:
      - Eixo do duto: eixo Z local (de z = -0.024 m na saída do jato até z = +0.024 m no bocal de admissão)
      - Raio interno nominal: 25.5 mm (diâmetro 51 mm para alojar o rotor de 50 mm)
      - Bocal frontal (intake lip / bellmouth) arredondado abrindo até R_ext = 31.5 mm
      - Parede externa com anel de reforço estrutural
    """
    tris = []
    n_circ = 48

    # Perfil longitudinal (z, r_interno, r_externo) de trás (saída, z=-0.024) para frente (admissão, z=+0.024)
    profile = [
        (-0.024, 0.0250, 0.0272),
        (-0.014, 0.0255, 0.0282),
        (-0.004, 0.0256, 0.0285),
        ( 0.006, 0.0256, 0.0288),
        ( 0.015, 0.0260, 0.0305),
        ( 0.020, 0.0272, 0.0318),
        ( 0.023, 0.0290, 0.0315),
        ( 0.024, 0.0303, 0.0303),  # Borda frontal arredondada do bellmouth
    ]

    angles = np.linspace(0.0, 2.0 * np.pi, n_circ, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Paredes interna e externa por revolução
    for i in range(len(profile) - 1):
        z0, ri0, ro0 = profile[i]
        z1, ri1, ro1 = profile[i + 1]
        for k in range(n_circ):
            kn = (k + 1) % n_circ
            c0, s0 = cos_a[k], sin_a[k]
            c1, s1 = cos_a[kn], sin_a[kn]

            # Superfície externa (normal para fora)
            p00_o = (ro0 * c0, ro0 * s0, z0)
            p10_o = (ro0 * c1, ro0 * s1, z0)
            p11_o = (ro1 * c1, ro1 * s1, z1)
            p01_o = (ro1 * c0, ro1 * s0, z1)
            add_quad(tris, p00_o, p10_o, p11_o, p01_o)

            # Superfície interna (normal para dentro)
            p00_i = (ri0 * c0, ri0 * s0, z0)
            p10_i = (ri0 * c1, ri0 * s1, z0)
            p11_i = (ri1 * c1, ri1 * s1, z1)
            p01_i = (ri1 * c0, ri1 * s0, z1)
            add_quad(tris, p00_i, p01_i, p11_i, p10_i)

    # Fechar anel traseiro (z = -0.024)
    z_b, ri_b, ro_b = profile[0]
    for k in range(n_circ):
        kn = (k + 1) % n_circ
        c0, s0 = cos_a[k], sin_a[k]
        c1, s1 = cos_a[kn], sin_a[kn]
        p0_i = (ri_b * c0, ri_b * s0, z_b)
        p1_i = (ri_b * c1, ri_b * s1, z_b)
        p1_o = (ro_b * c1, ro_b * s1, z_b)
        p0_o = (ro_b * c0, ro_b * s0, z_b)
        add_quad(tris, p0_i, p1_i, p1_o, p0_o)

    write_binary_stl(filepath, tris)


def build_edf_rotor_11blades_stl(filepath: str):
    """
    Gera o rotor da turbina EDF FMS 50mm com 11 pás torsidas e ogiva cônica (spinner):
      - Eixo de rotação: eixo Z local
      - Raio do cubo (hub): 10.5 mm (0.0105 m)
      - Ogiva cônica frontal (spinner) indo de z = +0.006 m até z = +0.020 m
      - 11 pás aerodinâmicas com raio externo = 24.8 mm (diâmetro 49.6 mm dentro do duto de 51 mm)
    """
    tris = []
    n_seg = 33
    r_hub = 0.0105
    z_back = -0.006
    z_front = 0.006
    z_tip = 0.020

    # 1. Cubo cilíndrico + Ogiva cônica frontal (spinner)
    for i in range(n_seg):
        a0 = 2.0 * np.pi * i / n_seg
        a1 = 2.0 * np.pi * (i + 1) / n_seg
        c0, s0 = np.cos(a0), np.sin(a0)
        c1, s1 = np.cos(a1), np.sin(a1)

        p00 = (r_hub * c0, r_hub * s0, z_back)
        p10 = (r_hub * c1, r_hub * s1, z_back)
        p11 = (r_hub * c1, r_hub * s1, z_front)
        p01 = (r_hub * c0, r_hub * s0, z_front)
        add_quad(tris, p00, p10, p11, p01)

        # Tampa traseira do cubo
        tris.append(((0.0, 0.0, z_back), p10, p00))
        # Cone frontal (ogiva)
        tris.append(((0.0, 0.0, z_tip), p01, p11))

    # 2. 11 Pás aerodinâmicas da fan dutada (varredura + torção de 38° na raiz a 24° na ponta)
    n_blades = 11
    r_outer = 0.0246
    n_span = 10
    radii = np.linspace(r_hub * 0.94, r_outer, n_span)

    for b in range(n_blades):
        base_angle = 2.0 * np.pi * b / n_blades
        sections = []
        for r in radii:
            frac = (r - r_hub) / (r_outer - r_hub)
            chord = 0.0105 * (1.0 - 0.12 * frac)
            twist = np.deg2rad(38.0 - 15.0 * frac)
            sweep = 0.15 * frac  # leve enflechamento angular da pá
            thick = 0.0009 * (1.0 - 0.35 * frac)

            ang = base_angle + sweep
            ca, sa = np.cos(ang), np.sin(ang)
            # Direção radial (er) e tangencial (et)
            er = np.array([ca, sa, 0.0])
            et = np.array([-sa, ca, 0.0])
            ez = np.array([0.0, 0.0, 1.0])

            center = r * er
            # Vetor ao longo da corda e normal à superfície da pá
            ec = np.cos(twist) * et + np.sin(twist) * ez
            en = -np.sin(twist) * et + np.cos(twist) * ez

            le_top = tuple(center + 0.5 * chord * ec + thick * en)
            te_top = tuple(center - 0.5 * chord * ec + thick * en)
            te_bot = tuple(center - 0.5 * chord * ec - thick * en)
            le_bot = tuple(center + 0.5 * chord * ec - thick * en)
            sections.append((le_top, te_top, te_bot, le_bot))

        for i in range(len(sections) - 1):
            s0 = sections[i]
            s1 = sections[i + 1]
            for k in range(4):
                kn = (k + 1) % 4
                add_quad(tris, s0[k], s1[k], s1[kn], s0[kn])
        st = sections[-1]
        add_quad(tris, st[0], st[3], st[2], st[1])

    write_binary_stl(filepath, tris)


if __name__ == "__main__":
    mesh_dir = os.path.join(os.path.dirname(__file__), "meshes")
    build_tower_stl(os.path.join(mesh_dir, "torre_esquerda.stl"), sign_x=1.0)
    build_tower_stl(os.path.join(mesh_dir, "torre_direita.stl"), sign_x=-1.0)
    build_edf_duct_stl(os.path.join(mesh_dir, "edf_duto.stl"))
    build_edf_rotor_11blades_stl(os.path.join(mesh_dir, "edf_rotor_11pas.stl"))
    print("Meshes STL (Torres + Turbina Dutada EDF 50mm 11 pás) geradas com sucesso em:", mesh_dir)
