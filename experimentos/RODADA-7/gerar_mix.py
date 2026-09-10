import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_and_mix(files, output_filename, plot_filename, threshold=10.0):
    if not files:
        print(f"Nenhum arquivo encontrado para {output_filename}")
        return
        
    print(f"\n--- Gerando {output_filename} ---")
    df_list = []
    current_time_ms = 0
    
    for f in sorted(files):
        if "MIX_" in f: continue
        
        df = pd.read_csv(f)
        u_col = 'u_pct' if 'u_pct' in df.columns else 'motor_percent'
        u_vals = df[u_col].values
        
        # Encontra o primeiro e o último índice onde u é maior que o threshold (exclui zeros no início e fim)
        active_indices = np.where(u_vals > threshold)[0]
        if len(active_indices) == 0:
            continue
            
        first_idx = active_indices[0]
        last_idx = active_indices[-1]
        
        # Corta o dataframe
        df_cut = df.iloc[first_idx:last_idx+1].copy()
        
        # Ajusta o tempo
        n_samples = len(df_cut)
        # Recalcula o tempo linearmente assumindo passo de 10ms (padrão do aeropendulo)
        new_time = np.arange(n_samples) * 10 + current_time_ms
        
        df_cut['tempo_ms'] = new_time
        current_time_ms = new_time[-1] + 10 # Atualiza o tempo para o próximo arquivo
        
        print(f"Adicionado {os.path.basename(f)} | Linhas: {n_samples}")
        df_list.append(df_cut)
        
    if df_list:
        df_mix = pd.concat(df_list, ignore_index=True)
        df_mix.to_csv(output_filename, index=False)
        print(f"Arquivo salvo: {output_filename} com {len(df_mix)} linhas.")
        
        # Gerar o gráfico de validação
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        t_sec = df_mix['tempo_ms'].values / 1000.0
        u_mix = df_mix[u_col].values
        y_mix = df_mix['angulo_deg'].values
        
        axes[0].plot(t_sec, y_mix, color='blue', lw=1)
        axes[0].set_title(f'Saída Y (Ângulo) - {output_filename}')
        axes[0].set_ylabel('Ângulo [°]')
        axes[0].grid(True)
        
        axes[1].plot(t_sec, u_mix, color='green', lw=1)
        axes[1].set_title(f'Entrada U (Controle) - {output_filename}')
        axes[1].set_xlabel('Tempo [s]')
        axes[1].set_ylabel('u [%]')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"Gráfico salvo: {plot_filename}")

if __name__ == "__main__":
    folder = r"C:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-7"
    os.chdir(folder)
    
    # Todos os arquivos que possuem -45- e não são png
    files_45 = [f for f in glob.glob("*-45-*.csv")]
    process_and_mix(files_45, "MIX_DC_45.csv", "plot_mix_45.png", threshold=20.0)
    
    # Todos os arquivos que possuem -60- e não são png
    files_60 = [f for f in glob.glob("*-60-*.csv")]
    # O threshold é 20.0 para garantir que pega quando passa do zero (em direção a 60)
    process_and_mix(files_60, "MIX_DC_60.csv", "plot_mix_60.png", threshold=20.0)
