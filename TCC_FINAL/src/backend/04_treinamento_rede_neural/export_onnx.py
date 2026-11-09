"""
Script para exportar os pesos de ann_weights.h para o formato ONNX (ann_model.onnx).
O modelo gerado pode ser carregado diretamente no STM32Cube.AI / STM32CubeMX.
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn

# Garante codificacao correta no Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HEADER_PATH = os.path.join(CURRENT_DIR, "ann_weights.h")
ONNX_PATH = os.path.join(CURRENT_DIR, "ann_model.onnx")

print(f"Lendo pesos de: {HEADER_PATH}")
with open(HEADER_PATH, "r", encoding="utf-8") as f:
    text = f.read()

def parse_array(name, size):
    m = re.search(r'' + name + r'\[\d+\]\s*=\s*\{([^}]+)\};', text)
    if not m:
        raise ValueError(f"Array '{name}' nao encontrado no arquivo.")
    raw = m.group(1).replace("f", "").split(",")
    vals = [float(x.strip()) for x in raw if x.strip()]
    assert len(vals) == size, f"Tamanho incorreto para {name}: esperado {size}, obtido {len(vals)}"
    return np.array(vals, dtype=np.float32)

# Carrega pesos e biases
W0 = parse_array("W0", 16640).reshape(128, 130)
b0 = parse_array("b0", 128)
W1 = parse_array("W1", 16384).reshape(128, 128)
b1 = parse_array("b1", 128)
W2 = parse_array("W2", 128).reshape(1, 128)
b2 = parse_array("b2", 1)

print(f"W0 shape: {W0.shape}, b0 shape: {b0.shape}")
print(f"W1 shape: {W1.shape}, b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape}, b2 shape: {b2.shape}")

class MPCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(130, 128)
        self.relu0 = nn.ReLU()
        self.fc1 = nn.Linear(128, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

        with torch.no_grad():
            self.fc0.weight.copy_(torch.from_numpy(W0))
            self.fc0.bias.copy_(torch.from_numpy(b0))
            self.fc1.weight.copy_(torch.from_numpy(W1))
            self.fc1.bias.copy_(torch.from_numpy(b1))
            self.fc2.weight.copy_(torch.from_numpy(W2))
            self.fc2.bias.copy_(torch.from_numpy(b2))

    def forward(self, x):
        x = self.relu0(self.fc0(x))
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

model = MPCModel()
model.eval()

# Entrada com tamanho fixo (1, 130) exigido para alocacao estatica em microcontroladores
dummy_input = torch.zeros(1, 130, dtype=torch.float32)

print(f"Exportando para: {ONNX_PATH}")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    input_names=["input"],
    output_names=["output"],
    dynamo=False
)

import onnx
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
file_size_kb = os.path.getsize(ONNX_PATH) / 1024
print(f"[OK] Modelo ONNX validado com sucesso! Tamanho: {file_size_kb:.2f} KB")
