import json

notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": text,
        "execution_count": None,
        "outputs": []
    })

add_md([
    "# Identificação do Aeropêndulo - Batch Least Squares\n",
    "Neste notebook, vamos utilizar os dados reais da bancada (sinal swept-sine ou multisine) para identificar a função de transferência discreta (ARX) do sistema usando Mínimos Quadrados em Lote."
])

add_code([
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import signal"
])

add_md([
    "## 1. Carregamento e Preparação dos Dados\n",
    "Para a identificação, os melhores dados são os que têm variação rica de frequências, como o **Swept-Sine** ou **Multisine**. A coleta semi-estática não excita a dinâmica adequadamente e pode gerar resultados ruins para a parte transitória. Portanto, recomendamos o arquivo `sinal3_swept_sine_malha_fechada.csv` ou os arquivos de multisine.\n",
    "\n",
    "**Nota:** Se estiver rodando no Google Colab, não esqueça de fazer o upload do arquivo CSV no painel à esquerda."
])

add_code([
    "# Carregando o arquivo CSV (Ajuste o caminho se necessário)\n",
    "file_path = 'sinal3_swept_sine_malha_fechada.csv'\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# Extraindo tempo, entrada (u) e saída (y)\n",
    "t = df['tempo_ms'].values / 1000.0  # converter para segundos\n",
    "u = df['u_pct'].values\n",
    "y = df['angulo_deg'].values\n",
    "\n",
    "# O período de amostragem Ts pode ser estimado pela média da diferença de tempo\n",
    "Ts = np.mean(np.diff(t))\n",
    "print(f'Período de amostragem estimado (Ts): {Ts:.3f} s')\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.subplot(2, 1, 1)\n",
    "plt.plot(t, u, label='Sinal de Controle (u_pct)')\n",
    "plt.ylabel('u (%)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.subplot(2, 1, 2)\n",
    "plt.plot(t, y, label='Ângulo (angulo_deg)', color='orange')\n",
    "plt.xlabel('Tempo (s)')\n",
    "plt.ylabel('y (graus)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.show()"
])

add_md([
    "## 2. Divisão em Treino e Teste\n",
    "Vamos separar a primeira parte dos dados para identificação (Treino) e o restante para validação cruzada (Teste)."
])

add_code([
    "N = len(u)\n",
    "split_idx = int(0.7 * N) # 70% para treino, 30% para teste\n",
    "\n",
    "u_train, y_train = u[:split_idx], y[:split_idx]\n",
    "u_test, y_test = u[split_idx:], y[split_idx:]\n",
    "t_train, t_test = t[:split_idx], t[split_idx:]\n",
    "\n",
    "print(f'Total de amostras: {N}')\n",
    "print(f'Amostras de Treino: {len(u_train)}')\n",
    "print(f'Amostras de Teste: {len(u_test)}')"
])

add_md([
    "## 3. Identificação do Sistema (Batch Least Squares)\n",
    "Vamos identificar um modelo ARX discreto de segunda ordem. A estrutura do modelo será:\n",
    "$$y(k) = -a_1 y(k-1) - a_2 y(k-2) + b_1 u(k-1) + b_2 u(k-2)$$\n",
    "\n",
    "A matriz regressora $\\Phi$ será composta por estes termos atrasados e os parâmetros $\\theta = [a_1, a_2, b_1, b_2]^T$."
])

add_code([
    "def construct_phi(y_data, u_data):\n",
    "    N_data = len(y_data)\n",
    "    \n",
    "    # Vetores deslocados no tempo\n",
    "    y_k_minus_1 = -y_data[1:N_data-1]\n",
    "    y_k_minus_2 = -y_data[0:N_data-2]\n",
    "    u_k_minus_1 =  u_data[1:N_data-1]\n",
    "    u_k_minus_2 =  u_data[0:N_data-2]\n",
    "    \n",
    "    # Montagem da Matriz Phi\n",
    "    Phi = np.column_stack((y_k_minus_1, y_k_minus_2, u_k_minus_1, u_k_minus_2))\n",
    "    \n",
    "    # O alvo y(k) correspondente\n",
    "    Y_target = y_data[2:N_data]\n",
    "    \n",
    "    return Phi, Y_target\n",
    "\n",
    "# Construindo a matriz de regressão para treino\n",
    "Phi_TRAIN, Y_target_TRAIN = construct_phi(y_train, u_train)\n",
    "\n",
    "# Estimando os parâmetros via Least Squares (Mínimos Quadrados)\n",
    "theta_hat, residuals, rank, singular_values = np.linalg.lstsq(Phi_TRAIN, Y_target_TRAIN, rcond=None)\n",
    "\n",
    "print('Parâmetros Estimados [a1, a2, b1, b2]:')\n",
    "print(theta_hat)"
])

add_md([
    "## 4. Avaliação (Predição One-Step-Ahead)\n",
    "A predição um-passo-à-frente (OSA) utiliza os valores reais passados para prever o valor atual. Isso avalia o quão bem a equação consegue calcular o próximo ponto."
])

add_code([
    "# Predição no conjunto de treino\n",
    "yhat_TRAIN = Phi_TRAIN @ theta_hat\n",
    "\n",
    "# Matriz Phi para teste\n",
    "Phi_TEST, Y_target_TEST = construct_phi(y_test, u_test)\n",
    "\n",
    "# Predição no conjunto de teste\n",
    "yhat_TEST = Phi_TEST @ theta_hat\n",
    "\n",
    "# Plot Treino\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(t_train[2:], Y_target_TRAIN, label='Medido (Treino)')\n",
    "plt.plot(t_train[2:], yhat_TRAIN, label='Predição OSA (Treino)', linestyle='--')\n",
    "plt.title('Fase de Treino: Saída e Predição One-Step-Ahead')\n",
    "plt.xlabel('Tempo (s)')\n",
    "plt.ylabel('y (graus)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "# Plot Teste\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(t_test[2:], Y_target_TEST, label='Medido (Teste)')\n",
    "plt.plot(t_test[2:], yhat_TEST, label='Predição OSA (Teste)', linestyle='--')\n",
    "plt.title('Fase de Teste: Saída e Predição One-Step-Ahead')\n",
    "plt.xlabel('Tempo (s)')\n",
    "plt.ylabel('y (graus)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
])

add_md([
    "## 5. Simulando a Malha Aberta Completa (Free-Run Simulation)\n",
    "Diferente do One-Step-Ahead, na simulação livre (free run) o modelo não recebe mais os valores reais de $y(k-1)$ e $y(k-2)$. Ele tem que usar as suas próprias estimativas passadas $\\hat{y}(k-1)$ e $\\hat{y}(k-2)$. Este é o teste definitivo da validade do modelo!"
])

add_code([
    "def simulate_model(u_data, theta):\n",
    "    a1, a2, b1, b2 = theta\n",
    "    y_sim = np.zeros(len(u_data))\n",
    "    \n",
    "    # Condição inicial (vamos supor 0, ou podemos pegar os 2 primeiros pontos reais)\n",
    "    y_sim[0] = 0.0\n",
    "    y_sim[1] = 0.0\n",
    "    \n",
    "    for k in range(2, len(u_data)):\n",
    "        # Usando os próprios y_sim passados (Simulação Livre)\n",
    "        y_sim[k] = -a1 * y_sim[k-1] - a2 * y_sim[k-2] + b1 * u_data[k-1] + b2 * u_data[k-2]\n",
    "        \n",
    "    return y_sim\n",
    "\n",
    "y_sim_test = simulate_model(u_test, theta_hat)\n",
    "\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(t_test, y_test, label='Medido Real (Teste)')\n",
    "plt.plot(t_test, y_sim_test, label='Simulação Livre (Free-Run)', linestyle='--')\n",
    "plt.title('Fase de Teste: Simulação de Malha Aberta (Free Run)')\n",
    "plt.xlabel('Tempo (s)')\n",
    "plt.ylabel('y (graus)')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
])

with open('identificação-collab/1_batch_least_squares_real_data.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook gerado em identificação-collab/1_batch_least_squares_real_data.ipynb")
