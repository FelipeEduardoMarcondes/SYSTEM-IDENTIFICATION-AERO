import json
import os

pasta_dados = r"c:\Users\mathe\OneDrive - Grupo Marista\PUCPR\15 Semestre\CONTROLE AVANÇADO\GITs\1-4 DRONE - GIT\SYSTEM-IDENTIFICATION-AERO\MPC_Hardware_Test\hil_PID\Dados para treinamento de Modelo"
arquivos = [
    "sinal1_semi_estatica_malha_fechada.csv",
    "sinal2_degraus_malha_fechada.csv",
    "sinal3_swept_sine_malha_fechada.csv",
    "sinal4_multiseno_malha_fechada.csv"
]

cells = []

# Célula 1: Imports
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# No Google Colab, se você fizer o upload direto, os arquivos ficarão na pasta raiz '/content'\n",
        "pasta_dados = '.'\n"
    ]
})

# Células para cada arquivo
for arquivo in arquivos:
    # Título Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"## Análise: `{arquivo}`\n"
        ]
    })
    
    # Código do Gráfico
    code = f"""arquivo = f"{{pasta_dados}}/{arquivo}"
df = pd.read_csv(arquivo)

plt.figure(figsize=(12, 5))
if 'Tempo_s' in df.columns:
    if 'Referencia' in df.columns:
        plt.plot(df['Tempo_s'], df['Referencia'], label='Referência', linestyle='--', color='black')
    if 'Saida_y' in df.columns:
        plt.plot(df['Tempo_s'], df['Saida_y'], label='Saída (y)', color='blue')
    if 'Controle_u' in df.columns:
        plt.plot(df['Tempo_s'], df['Controle_u'], label='Controle (u)', color='red', alpha=0.7)
    plt.xlabel('Tempo (s)')
else:
    for col in df.columns[1:]:
        plt.plot(df[df.columns[0]], df[col], label=col)
    plt.xlabel(df.columns[0])

plt.ylabel('Amplitude')
plt.title('{arquivo}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
    # É preciso tratar corretamente o final de linha (adicionando "\n" mas não na última se já tiver)
    lines = [line + "\n" for line in code.split("\n")]
    # Remover o último newline extra caso exista
    if lines[-1] == "\n":
        lines = lines[:-1]

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "plot_dados_colab.ipynb"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(os.path.join(pasta_dados, "plot_dados_colab.ipynb"), "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook gerado com sucesso!")
