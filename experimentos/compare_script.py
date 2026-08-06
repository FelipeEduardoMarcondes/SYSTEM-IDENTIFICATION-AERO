import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(file1, file2, title, output_name):
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Trim to shortest length just in case they differ slightly
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]

    # Calculate RMSE
    rmse_ang = np.sqrt(np.mean((df1['angulo_deg'] - df2['angulo_deg'])**2))
    rmse_u = np.sqrt(np.mean((df1['u_pct'] - df2['u_pct'])**2))

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"{title}\nRMSE Angulo: {rmse_ang:.2f} deg, RMSE U: {rmse_u:.2f} %")

    # Angulo
    axs[0].plot(df1['tempo_ms']/1000, df1['angulo_deg'], label='Hoje (08/06)', alpha=0.8)
    axs[0].plot(df2['tempo_ms']/1000, df2['angulo_deg'], label='Rodada-1 (07/31)', alpha=0.8, linestyle='--')
    axs[0].plot(df1['tempo_ms']/1000, df1['referencia'], label='Ref', alpha=0.5, color='gray', linestyle=':')
    axs[0].set_ylabel('Ângulo (deg)')
    axs[0].legend()
    axs[0].grid(True)

    # Controle
    axs[1].plot(df1['tempo_ms']/1000, df1['u_pct'], label='Hoje (08/06)', alpha=0.8)
    axs[1].plot(df2['tempo_ms']/1000, df2['u_pct'], label='Rodada-1 (07/31)', alpha=0.8, linestyle='--')
    axs[1].set_ylabel('Sinal de Controle u (%)')
    axs[1].set_xlabel('Tempo (s)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_name)
    print(f"Saved {output_name}")
    print(f"[{title}] RMSE Angulo: {rmse_ang:.2f} deg, RMSE U: {rmse_u:.2f} %")

if __name__ == "__main__":
    file1_today = r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\multi-seno-1_0806_16-16.csv"
    file1_old = r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-2\multi-seno-1_0804_19-03.csv"
    
    file2_today = r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\multi-seno-2_0806_16-19.csv"
    file2_old = r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-2\multi-seno-2_0804_19-28.csv"
    
    plot_comparison(file1_today, file1_old, "Comparação Multi-Seno 1 (Hoje vs Rodada 2)", r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\compare_multi_seno_1.png")
    plot_comparison(file2_today, file2_old, "Comparação Multi-Seno 2 (Hoje vs Rodada 2)", r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\compare_multi_seno_2.png")
