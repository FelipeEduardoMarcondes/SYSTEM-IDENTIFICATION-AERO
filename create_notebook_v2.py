import json

notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text})

def add_code(text):
    notebook["cells"].append({"cell_type": "code", "metadata": {}, "source": text, "execution_count": None, "outputs": []})

add_md([
    "# Identificação do Aeropêndulo - Mínimos Quadrados (Avançado)\n",
    "Neste notebook, aplicaremos melhorias cruciais ao modelo ARX de Mínimos Quadrados:\n",
    "1. **Detrending**: Remoção da média (ponto de operação) dos sinais antes da estimação.\n",
    "2. **Condições Iniciais Corretas**: Utilização dos valores reais de teste para inicializar a simulação livre."
])

add_code([
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt"
])

add_md(["## 1. Carregamento e Preparação"])
add_code([
    "file_path = 'sinal3_swept_sine_malha_fechada.csv'\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "t = df['tempo_ms'].values / 1000.0\n",
    "u_raw = df['u_pct'].values\n",
    "y_raw = df['angulo_deg'].values\n",
    "\n",
    "Ts = np.mean(np.diff(t))\n",
    "print(f'Ts estimado: {Ts:.3f} s')"
])

add_md(["## 2. Divisão em Treino e Teste (Com Remoção de Média)\n", "Para modelos lineares ARX, é fundamental que a modelagem ocorra em variações em torno de zero."])
add_code([
    "N = len(u_raw)\n",
    "split_idx = int(0.7 * N)\n",
    "\n",
    "u_train_raw, y_train_raw = u_raw[:split_idx], y_raw[:split_idx]\n",
    "u_test_raw, y_test_raw = u_raw[split_idx:], y_raw[split_idx:]\n",
    "t_train, t_test = t[:split_idx], t[split_idx:]\n",
    "\n",
    "# REMOÇÃO DO PONTO DE OPERAÇÃO (Média do Treino)\n",
    "u_mean = np.mean(u_train_raw)\n",
    "y_mean = np.mean(y_train_raw)\n",
    "\n",
    "u_train = u_train_raw - u_mean\n",
    "y_train = y_train_raw - y_mean\n",
    "\n",
    "u_test = u_test_raw - u_mean\n",
    "y_test = y_test_raw - y_mean\n",
    "\n",
    "print(f'Média removida do u: {u_mean:.2f}')\n",
    "print(f'Média removida do y: {y_mean:.2f}')"
])

add_md(["## 3. Identificação do Modelo ARX (2ª Ordem)"])
add_code([
    "def construct_phi(y_data, u_data):\n",
    "    N_data = len(y_data)\n",
    "    y_k_minus_1 = -y_data[1:N_data-1]\n",
    "    y_k_minus_2 = -y_data[0:N_data-2]\n",
    "    u_k_minus_1 =  u_data[1:N_data-1]\n",
    "    u_k_minus_2 =  u_data[0:N_data-2]\n",
    "    \n",
    "    Phi = np.column_stack((y_k_minus_1, y_k_minus_2, u_k_minus_1, u_k_minus_2))\n",
    "    Y_target = y_data[2:N_data]\n",
    "    return Phi, Y_target\n",
    "\n",
    "Phi_TRAIN, Y_target_TRAIN = construct_phi(y_train, u_train)\n",
    "theta_hat, residuals, rank, singular_values = np.linalg.lstsq(Phi_TRAIN, Y_target_TRAIN, rcond=None)\n",
    "\n",
    "print('Parâmetros Estimados [a1, a2, b1, b2]:')\n",
    "print(theta_hat)"
])

add_md(["## 4. Simulação Livre (Free-Run) com Condições Iniciais Ajustadas"])
add_code([
    "def simulate_model(u_data, theta, y_initial_1, y_initial_2):\n",
    "    a1, a2, b1, b2 = theta\n",
    "    y_sim = np.zeros(len(u_data))\n",
    "    \n",
    "    # Setando as condições iniciais com os valores reais (sem média)\n",
    "    y_sim[0] = y_initial_1\n",
    "    y_sim[1] = y_initial_2\n",
    "    \n",
    "    for k in range(2, len(u_data)):\n",
    "        y_sim[k] = -a1 * y_sim[k-1] - a2 * y_sim[k-2] + b1 * u_data[k-1] + b2 * u_data[k-2]\n",
    "        \n",
    "    return y_sim\n",
    "\n",
    "# Simulando com as condições iniciais corretas de teste\n",
    "y_sim_test_centered = simulate_model(u_test, theta_hat, y_test[0], y_test[1])\n",
    "\n",
    "# Devolvendo o Ponto de Operação (Média) para o gráfico\n",
    "y_sim_test_real = y_sim_test_centered + y_mean\n",
    "\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(t_test, y_test_raw, label='Medido Real (Teste)')\n",
    "plt.plot(t_test, y_sim_test_real, label='Simulação Livre (Corrigida)', linestyle='--')\n",
    "plt.title('Fase de Teste: Simulação de Malha Aberta (Free Run) Melhorada')\n",
    "plt.xlabel('Tempo (s)')\n",
    "plt.ylabel('y (graus)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
])

with open('identificação-collab/2_batch_least_squares_melhorado.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
