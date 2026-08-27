import pandas as pd
import numpy as np
from scipy.signal import welch, csd

def get_bandwidth(filepath):
    df = pd.read_csv(filepath)
    df['tempo_s'] = df['tempo_ms'] / 1000.0
    dt = np.mean(np.diff(df['tempo_s']))
    fs = 1.0 / dt
    
    ref = df['referencia'].values
    ang = df['angulo_deg'].values
    ang_centered = ang - np.mean(ang)
    
    f_welch, Pxx = welch(ref, fs=fs, nperseg=2048)
    f_welch, Pyx = csd(ref, ang_centered, fs=fs, nperseg=2048)
    H = Pyx / Pxx
    mag = np.abs(H)
    mag_db = 20 * np.log10(mag + 1e-10)
    
    # We find the frequency where it drops below -3dB
    for i in range(1, len(mag_db)):
        if mag_db[i] < -3.0:
            return f_welch[i]
    return f_welch[-1]

file_swept = r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\swept-sine-1_0819_19-02.csv"
file_chirp = r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-3\chirp-2_0807_17-09.csv"

print(f"Banda Passante (-3dB) Swept-Sine: {get_bandwidth(file_swept):.3f} Hz")
print(f"Banda Passante (-3dB) Chirp: {get_bandwidth(file_chirp):.3f} Hz")
