import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import os

Ts = 0.01

# Calculate exact size from mpc_v4.py by replicating its random state
np.random.seed(42)
dur_steps_45 = sum(int(round(np.random.uniform(2.0, 6.0) / Ts)) for _ in range(15))
dur_steps_60 = sum(int(round(np.random.uniform(2.0, 6.0) / Ts)) for _ in range(15))
steps_train = int(round(5.0/Ts)) + int(round(30.0/Ts)) + dur_steps_45 + int(round(5.0/Ts)) + int(round(30.0/Ts)) + dur_steps_60 + int(round(5.0/Ts))
tvec = np.arange(steps_train) * Ts

print(f"Total time length derived: {steps_train * Ts:.2f} s")

# Ref 1: Sequence of large steps
np.random.seed(101)
ref1 = np.zeros(steps_train)
t_accum = 0
current_val = 45.0
while t_accum < steps_train:
    dur = int(round(np.random.uniform(5.0, 15.0) / Ts))
    if t_accum + dur > steps_train:
        dur = steps_train - t_accum
    ref1[t_accum : t_accum + dur] = current_val
    current_val = np.random.uniform(20.0, 110.0)
    t_accum += dur

# Ref 2: Slower Multisine
ref2 = np.zeros(steps_train)
f_max = 0.1
df = 1.0 / (steps_train * Ts)
freqs = np.arange(df, f_max + 1e-9, df)
phases = np.random.uniform(0, 2 * np.pi, len(freqs))
ms = np.sum([np.sin(2*np.pi*f*tvec + ph) for f, ph in zip(freqs, phases)], axis=0)
ref2 = ms / np.max(np.abs(ms)) * 40.0 + 65.0

# Ref 3: Triangle / Ramp waves
ref3 = 60.0 + 30.0 * signal.sawtooth(2 * np.pi * 0.05 * tvec, 0.5)

# Save to CSV
dir_path = "c:/Users/vicio/Documents/AEROPENDULO/sysid/notebooks/identificação-collab/IDENT-HELON"
pd.DataFrame({'time': tvec, 'ref': ref1}).to_csv(f"{dir_path}/ref1_degraus.csv", index=False)
pd.DataFrame({'time': tvec, 'ref': ref2}).to_csv(f"{dir_path}/ref2_senoidal.csv", index=False)
pd.DataFrame({'time': tvec, 'ref': ref3}).to_csv(f"{dir_path}/ref3_rampas.csv", index=False)

print("Saved CSV files.")

# Plot to show the user
plt.figure(figsize=(12, 6))
plt.plot(tvec, ref1, label='Ref 1: Degraus Aleatórios')
plt.plot(tvec, ref2, label='Ref 2: Senoidal Suave')
plt.plot(tvec, ref3, label='Ref 3: Rampas')
plt.legend()
plt.title(f'Referências Geradas (Mesma faixa de tempo: {steps_train*Ts:.1f}s)')
plt.xlabel('Tempo (s)')
plt.ylabel('Ângulo (deg)')
plt.grid(True)
plt.savefig(f"{dir_path}/referencias_geradas.png")
print("Saved plot.")
