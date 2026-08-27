import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch, csd
import os

# Files
file_swept = r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\swept-sine-1_0819_19-02.csv"
file_chirp = r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-3\chirp-2_0807_17-09.csv"

def analyze_and_plot(filepath, name, out_prefix):
    print(f"Analisando {name}...")
    df = pd.read_csv(filepath)
    df['tempo_s'] = df['tempo_ms'] / 1000.0
    
    ref = df['referencia'].values
    ang = df['angulo_deg'].values
    u = df['u_pct'].values
    t = df['tempo_s'].values
    
    # 1. Análise no Tempo
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, ref, label='Referência', linestyle='--', color='black')
    plt.plot(t, ang, label='Ângulo Real', alpha=0.8, color='blue')
    plt.title(f'Comportamento no Tempo - {name}')
    plt.ylabel('Ângulo (deg)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    erro = ref - ang
    plt.plot(t, erro, label='Erro (Ref - Real)', color='red')
    plt.ylabel('Erro (deg)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, u, label='Sinal de Controle (u%)', color='green')
    plt.axhline(100, color='r', linestyle=':', alpha=0.5)
    plt.axhline(0, color='r', linestyle=':', alpha=0.5)
    plt.ylabel('u (%)')
    plt.xlabel('Tempo (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_tempo.png')
    plt.close()
    
    # 2. Espectro de Frequência (FFT)
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    n = len(t)
    f = np.fft.rfftfreq(n, d=dt)
    
    fft_ref = np.abs(np.fft.rfft(ref)) / (n/2)
    fft_ang = np.abs(np.fft.rfft(ang - np.mean(ang))) / (n/2)
    
    plt.figure(figsize=(10, 5))
    plt.plot(f, fft_ref, label='Espectro da Referência', linestyle='--', color='black')
    plt.plot(f, fft_ang, label='Espectro do Ângulo', alpha=0.8, color='blue')
    plt.title(f'Espectro de Frequências (FFT) - {name}')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude (deg)')
    plt.xlim(0, max(f)/4) # limit depending on fs
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{out_prefix}_fft.png')
    plt.close()
    
    # 3. Espectrograma (para ver a evolução temporal das frequências)
    ang_centered = ang - np.mean(ang)
    f_spec, t_spec, Sxx = spectrogram(ang_centered, fs=fs, nperseg=512, noverlap=256)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.title(f'Espectrograma do Ângulo Real - {name}')
    plt.ylabel('Frequência (Hz)')
    plt.xlabel('Tempo (s)')
    plt.ylim(0, max(f)/4)
    plt.colorbar(label='Potência (dB/Hz)')
    plt.savefig(f'{out_prefix}_espectrograma.png')
    plt.close()
    
    # 4. Função de Transferência em Malha Fechada (Magnitude - Bode)
    # H(f) = S_yx(f) / S_xx(f)
    f_welch, Pxx = welch(ref, fs=fs, nperseg=2048)
    f_welch, Pyx = csd(ref, ang_centered, fs=fs, nperseg=2048)
    H = Pyx / Pxx
    mag = np.abs(H)
    
    plt.figure(figsize=(10, 5))
    plt.plot(f_welch, 20 * np.log10(mag + 1e-10))
    plt.axhline(0, color='r', linestyle='--', alpha=0.5, label='0 dB (Ganho Unitário)')
    plt.title(f'Resposta em Frequência (Bode de Magnitude) - {name}')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.xlim(0, max(f)/4)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{out_prefix}_bode_mag.png')
    plt.close()

analyze_and_plot(file_swept, 'Swept-Sine 1', 'swept')
analyze_and_plot(file_chirp, 'Chirp 2', 'chirp')
print("Gráficos gerados com sucesso.")
