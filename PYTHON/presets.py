"""
presets.py — Sequências de degraus predefinidas para o aeropêndulo.

Cada preset é uma tupla: (nome, duracao_s, [(t_s, ref_deg), ...])

Edite ou acrescente entradas conforme necessário antes de rodar o experimento.
As sequências são usadas pelo menu de experimentos (main.py, modo SEQ).
Para sinais contínuos (chirp, multi-seno), use os parâmetros em main.py.
"""

SEQUENCIAS_PRESET = [
    (
        "Degrau unico 40 deg / 30 s",
        30,
        [(0, 40)],
    ),
    (
        "Dois degraus: 40->30 deg / 30 s",
        30,
        [(0, 40), (15, 30)],
    ),
    (
        "Cinco degraus simetricos 35->45->35->45->35 / 50 s",
        50,
        [(0, 35), (10, 45), (20, 35), (30, 45), (40, 35)],
    ),
    (
        "Degraus validacao (amp variada, seed=0) / 60 s",
        60,
        # Equivale ao sinal gerado pelo MATLAB com amp=30, Tduracao=4 s
        [(0, 45), (4, 53), (8, 29), (12, 65), (16, 38),
         (20, 70), (24, 22), (28, 58), (32, 33), (36, 48),
         (40, 15), (44, 75), (48, 40), (52, 60), (56, 30)],
    ),
    (
        "Rampa 0->90->0 deg de 10 em 10 graus / 175 s",
        175,
        [
            (0, 0),   (5, 10),  (15, 20), (25, 30), (35, 40), (45, 50),
            (55, 60), (65, 70), (75, 80), (85, 90),
            (95, 80), (105, 70), (115, 60), (125, 50),
            (135, 40), (145, 30), (155, 20), (165, 10),
        ],
    ),
    (
        "Varredura amplitude 30->90->120->135 / 150 s",
        150,
        [(0, 0), (5, 30), (15, 50), (25, 75), (35, 90),
         (55, 110), (65, 115), (75, 120), (85, 125),
         (95, 130), (110, 135), (125, 140), (135, 135)],
    ),
]
