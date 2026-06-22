import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Caminho para o diretório contendo os arquivos CSV
pasta_dados = r"c:\Users\mathe\OneDrive - Grupo Marista\PUCPR\15 Semestre\CONTROLE AVANÇADO\GITs\1-4 DRONE - GIT\SYSTEM-IDENTIFICATION-AERO\MPC_Hardware_Test\hil_PID\Dados para treinamento de Modelo"

# Lista todos os arquivos CSV no diretório
arquivos_csv = glob.glob(os.path.join(pasta_dados, "*.csv"))

# Se houver arquivos, cria uma figura com subplots (uma "célula" de gráfico para cada arquivo)
num_arquivos = len(arquivos_csv)

if num_arquivos > 0:
    fig, axes = plt.subplots(num_arquivos, 1, figsize=(12, 5 * num_arquivos))
    
    # Garante que axes seja iterável mesmo com apenas 1 arquivo
    if num_arquivos == 1:
        axes = [axes]
        
    for i, arquivo in enumerate(arquivos_csv):
        df = pd.read_csv(arquivo)
        nome_arquivo = os.path.basename(arquivo)
        ax = axes[i]
        
        # Plotando as colunas conhecidas: Tempo_s, Referencia, Saida_y, Controle_u
        if 'Tempo_s' in df.columns:
            if 'Referencia' in df.columns:
                ax.plot(df['Tempo_s'], df['Referencia'], label='Referência', linestyle='--', color='black')
            if 'Saida_y' in df.columns:
                ax.plot(df['Tempo_s'], df['Saida_y'], label='Saída (y)', color='blue')
            if 'Controle_u' in df.columns:
                ax.plot(df['Tempo_s'], df['Controle_u'], label='Controle (u)', color='red', alpha=0.7)
            ax.set_xlabel('Tempo (s)')
        else:
            # Caso os nomes das colunas sejam diferentes, plota todas contra a primeira
            for col in df.columns[1:]:
                ax.plot(df[df.columns[0]], df[col], label=col)
            ax.set_xlabel(df.columns[0])
            
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Gráfico: {nome_arquivo}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Nenhum arquivo CSV encontrado na pasta especificada.")
