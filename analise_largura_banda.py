"""
Análise de Largura de Banda do Aeropêndulo
===========================================
Determina a frequência máxima para a qual o sistema é controlável/modelável.

Métodos:
1. Coerência (Cxy) entre entrada (referência) e saída (ângulo) - usando chirps/swept-sines
2. ETFE (Empirical Transfer Function Estimate) - magnitude da resposta em frequência
3. Análise do erro no domínio do tempo com janela deslizante para mapear onde o tracking se degrada
4. PSD (Power Spectral Density) do sinal de saída vs referência

Autor: Análise automática
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('dark_background')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'savefig.facecolor': '#1a1a2e',
    'grid.alpha': 0.3,
})

OUTPUT_DIR = Path(r"c:\Users\vicio\Documents\AEROPENDULO\analise_bandwidth")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. Carregar dados dos ensaios relevantes (chirps e swept-sine)
# ============================================================================

chirp_files = {
    'chirp-1 (R3, 16:32)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-3\chirp-1_0807_16-32.csv",
    'chirp-1 (R3, 16:34)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-3\chirp-1_0807_16-34.csv",
    'chirp-2 (R3, 17:07)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-3\chirp-2_0807_17-07.csv",
    'chirp-2 (R3, 17:09)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-3\chirp-2_0807_17-09.csv",
    'swept-sine-1 (19:02)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\swept-sine-1_0819_19-02.csv",
}

# Também carregar multi-seno para comparação
multisine_files = {
    'multi-seno-1 (R3, 16:57)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-3\multi-seno-1_0807_16-57.csv",
    'multi-seno-1 (R3, 17:01)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-3\multi-seno-1_0807_17-01.csv",
    'multi-seno-2 (R3, 17:13)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\RODADA-3\multi-seno-2_0807_17-13.csv",
    'multi-seno-1 (19:23)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\multi-seno-1_0819_19-23.csv",
    'multi-seno-2 (18:59)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\multi-seno-2_0819_18-59.csv",
    'multi-seno-2 (19:11)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\multi-seno-2_0819_19-11.csv",
    'multi-seno-3 (19:19)': r"c:\Users\vicio\Documents\AEROPENDULO\experimentos\multi-seno-3_0819_19-19.csv",
}

def load_experiment(filepath):
    """Carrega CSV do experimento e retorna dados processados."""
    df = pd.read_csv(filepath)
    t = df['tempo_ms'].values / 1000.0  # ms -> s
    y = df['angulo_deg'].values          # graus
    u = df['u_pct'].values               # % PWM
    ref = df['referencia'].values         # graus
    
    # Calcular taxa de amostragem
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    
    return t, y, u, ref, fs, dt


def find_active_region(ref, t, threshold=2.0):
    """Encontra a região ativa do ensaio (onde a referência muda)."""
    # Encontrar onde a referência começa a mudar significativamente
    ref_diff = np.abs(np.diff(ref))
    active = ref_diff > threshold
    
    if not np.any(active):
        # Tentar com threshold menor
        ref_range = np.max(ref) - np.min(ref)
        active = ref_diff > ref_range * 0.01
    
    if not np.any(active):
        return 0, len(t) - 1
    
    start_idx = max(0, np.argmax(active) - 10)
    end_idx = min(len(t) - 1, len(active) - 1 - np.argmax(active[::-1]) + 10)
    
    return start_idx, end_idx


def compute_coherence_and_etfe(ref, y, fs, nperseg=None):
    """
    Calcula coerência e ETFE entre referência e saída.
    
    A coerência Cxy^2 indica quão linearmente correlacionados são os sinais
    em cada frequência. Valores < 0.5 indicam que o modelo linear não é confiável.
    """
    N = len(ref)
    if nperseg is None:
        nperseg = min(N // 4, 1024)
    
    # Remover média
    ref_zm = ref - np.mean(ref)
    y_zm = y - np.mean(y)
    
    # Coerência
    f_coh, Cxy = signal.coherence(ref_zm, y_zm, fs=fs, nperseg=nperseg, 
                                   noverlap=nperseg//2)
    
    # ETFE - Empirical Transfer Function Estimate
    f_tf, Pxy = signal.csd(ref_zm, y_zm, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    _, Pxx = signal.welch(ref_zm, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    # H(f) = Pxy / Pxx
    H = Pxy / (Pxx + 1e-12)
    H_mag = np.abs(H)
    H_phase = np.angle(H, deg=True)
    
    # PSD da saída e entrada
    f_psd, Pyy = signal.welch(y_zm, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    return f_coh, Cxy, f_tf, H_mag, H_phase, f_psd, Pxx, Pyy


def compute_tracking_error_by_window(t, ref, y, window_duration=2.0):
    """
    Calcula o erro RMS do tracking por janelas temporais.
    Mapeia em qual momento (e portanto em qual frequência para chirps) o tracking piora.
    """
    dt = np.median(np.diff(t))
    window_samples = int(window_duration / dt)
    
    n_windows = len(t) // window_samples
    t_windows = []
    rmse_windows = []
    amplitude_ratio = []
    
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        
        ref_w = ref[start:end]
        y_w = y[start:end]
        
        # Erro RMS
        error = y_w - ref_w
        rmse = np.sqrt(np.mean(error**2))
        
        # Razão de amplitudes (quanto o sistema amplifica ou atenua)
        ref_amp = np.std(ref_w)
        y_amp = np.std(y_w)
        ratio = y_amp / (ref_amp + 1e-6) if ref_amp > 0.5 else np.nan
        
        t_windows.append(t[start + window_samples // 2])
        rmse_windows.append(rmse)
        amplitude_ratio.append(ratio)
    
    return np.array(t_windows), np.array(rmse_windows), np.array(amplitude_ratio)


def estimate_chirp_frequency_at_time(t, t_start_chirp, t_end_chirp, f_start, f_end):
    """Estima a frequência instantânea de um chirp linear em função do tempo."""
    duration = t_end_chirp - t_start_chirp
    t_rel = (t - t_start_chirp) / duration
    t_rel = np.clip(t_rel, 0, 1)
    f_inst = f_start + (f_end - f_start) * t_rel
    return f_inst


# ============================================================================
# 2. Análise principal
# ============================================================================

print("=" * 70)
print("ANÁLISE DE LARGURA DE BANDA DO AEROPÊNDULO")
print("=" * 70)

# ------- Análise dos Chirps -------
fig_coh, axes_coh = plt.subplots(len(chirp_files), 2, figsize=(16, 4 * len(chirp_files)))
fig_coh.suptitle('Coerência e Magnitude da Resposta em Frequência\n(Ensaios de Chirp / Swept-Sine)',
                 fontsize=15, fontweight='bold', color='white', y=0.98)

coherence_thresholds = {}

for idx, (name, filepath) in enumerate(chirp_files.items()):
    print(f"\n--- {name} ---")
    t, y, u, ref, fs, dt = load_experiment(filepath)
    print(f"  Fs = {fs:.1f} Hz, duração = {t[-1]:.1f} s, N = {len(t)}")
    
    # Encontrar região ativa do chirp
    start_i, end_i = find_active_region(ref, t)
    t_act = t[start_i:end_i]
    y_act = y[start_i:end_i]
    ref_act = ref[start_i:end_i]
    u_act = u[start_i:end_i]
    
    print(f"  Região ativa: {t_act[0]:.1f}s - {t_act[-1]:.1f}s ({len(t_act)} amostras)")
    
    # Calcular coerência e ETFE
    nperseg = min(len(t_act) // 6, 512)
    f_coh, Cxy, f_tf, H_mag, H_phase, f_psd, Pxx, Pyy = \
        compute_coherence_and_etfe(ref_act, y_act, fs, nperseg=nperseg)
    
    # Encontrar frequência onde coerência cai abaixo de 0.5
    coh_above_05 = f_coh[Cxy >= 0.5]
    if len(coh_above_05) > 0:
        f_max_coh = coh_above_05[-1]
    else:
        f_max_coh = 0
    
    # Encontrar frequência onde coerência cai abaixo de 0.7 (mais conservador)
    coh_above_07 = f_coh[Cxy >= 0.7]
    if len(coh_above_07) > 0:
        f_max_coh_07 = coh_above_07[-1]
    else:
        f_max_coh_07 = 0
    
    coherence_thresholds[name] = {
        'f_coh_0.5': f_max_coh,
        'f_coh_0.7': f_max_coh_07,
    }
    
    print(f"  f_max (coerência >= 0.5): {f_max_coh:.3f} Hz ({f_max_coh*2*np.pi:.2f} rad/s)")
    print(f"  f_max (coerência >= 0.7): {f_max_coh_07:.3f} Hz ({f_max_coh_07*2*np.pi:.2f} rad/s)")
    
    # Plot coerência
    ax1 = axes_coh[idx, 0]
    ax1.plot(f_coh, Cxy, 'c-', linewidth=1.5, label='Coerência $C_{xy}^2$')
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Limiar 0.5')
    ax1.axhline(0.7, color='orange', linestyle='--', alpha=0.7, label='Limiar 0.7')
    if f_max_coh > 0:
        ax1.axvline(f_max_coh, color='red', linestyle=':', alpha=0.5)
    if f_max_coh_07 > 0:
        ax1.axvline(f_max_coh_07, color='orange', linestyle=':', alpha=0.5)
    ax1.set_xlim([0, min(fs/2, 10)])
    ax1.set_ylim([0, 1.05])
    ax1.set_xlabel('Frequência (Hz)')
    ax1.set_ylabel('Coerência $C_{xy}^2$')
    ax1.set_title(f'{name}')
    ax1.legend(fontsize=8)
    ax1.grid(True)
    
    # Plot magnitude ETFE
    ax2 = axes_coh[idx, 1]
    ax2.semilogy(f_tf, H_mag, 'm-', linewidth=1.5, label='|H(f)|')
    if f_max_coh > 0:
        ax2.axvline(f_max_coh, color='red', linestyle=':', alpha=0.7, label=f'f_max(0.5)={f_max_coh:.2f} Hz')
    if f_max_coh_07 > 0:
        ax2.axvline(f_max_coh_07, color='orange', linestyle=':', alpha=0.7, label=f'f_max(0.7)={f_max_coh_07:.2f} Hz')
    ax2.set_xlim([0, min(fs/2, 10)])
    ax2.set_xlabel('Frequência (Hz)')
    ax2.set_ylabel('|H(f)| (Magnitude)')
    ax2.set_title(f'{name} - ETFE')
    ax2.legend(fontsize=8)
    ax2.grid(True)

fig_coh.tight_layout(rect=[0, 0, 1, 0.96])
fig_coh.savefig(OUTPUT_DIR / 'coerencia_etfe_chirps.png', dpi=150, bbox_inches='tight')
print(f"\nSalvo: {OUTPUT_DIR / 'coerencia_etfe_chirps.png'}")


# ============================================================================
# 3. Análise do Erro por Janela Temporal (mapeando para frequência nos chirps)
# ============================================================================

fig_err, axes_err = plt.subplots(len(chirp_files), 1, figsize=(16, 3.5 * len(chirp_files)))
fig_err.suptitle('Erro RMS por Janela Temporal (Ensaios Chirp/Swept-Sine)',
                 fontsize=15, fontweight='bold', color='white', y=0.98)

for idx, (name, filepath) in enumerate(chirp_files.items()):
    t, y, u, ref, fs, dt = load_experiment(filepath)
    
    # Calcular erro por janela
    t_w, rmse_w, amp_ratio = compute_tracking_error_by_window(t, ref, y, window_duration=2.0)
    
    ax = axes_err[idx]
    
    # Plot erro RMS
    color1 = '#00d4ff'
    ax.plot(t_w, rmse_w, 'o-', color=color1, markersize=4, linewidth=1.5, label='RMSE (°)')
    ax.set_ylabel('RMSE (°)', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    
    # Segundo eixo para razão de amplitude
    ax2 = ax.twinx()
    color2 = '#ff6b6b'
    valid = ~np.isnan(amp_ratio)
    ax2.plot(t_w[valid], amp_ratio[valid], 's-', color=color2, markersize=3, 
             linewidth=1, alpha=0.8, label='Amp ratio')
    ax2.axhline(1.0, color='yellow', linestyle='--', alpha=0.4)
    ax2.set_ylabel('Razão Amplitude (y/ref)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax.set_xlabel('Tempo (s)')
    ax.set_title(f'{name} - Degradação do Tracking')
    ax.grid(True, alpha=0.3)
    
    # Legenda combinada
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

fig_err.tight_layout(rect=[0, 0, 1, 0.96])
fig_err.savefig(OUTPUT_DIR / 'erro_por_janela.png', dpi=150, bbox_inches='tight')
print(f"Salvo: {OUTPUT_DIR / 'erro_por_janela.png'}")


# ============================================================================
# 4. Análise combinada: PSD da referência vs saída (todos os ensaios)
# ============================================================================

fig_psd, axes_psd = plt.subplots(2, 1, figsize=(14, 10))
fig_psd.suptitle('PSD Comparativa: Referência vs Saída\n(Todos os Ensaios de Chirp e Multi-Seno)',
                 fontsize=14, fontweight='bold', color='white')

colors_chirp = plt.cm.cool(np.linspace(0.2, 0.8, len(chirp_files)))
colors_multi = plt.cm.autumn(np.linspace(0.2, 0.8, len(multisine_files)))

# PSD Chirps
ax_psd1 = axes_psd[0]
for idx, (name, filepath) in enumerate(chirp_files.items()):
    t, y, u, ref, fs, dt = load_experiment(filepath)
    start_i, end_i = find_active_region(ref, t)
    ref_act = ref[start_i:end_i] - np.mean(ref[start_i:end_i])
    y_act = y[start_i:end_i] - np.mean(y[start_i:end_i])
    
    nperseg = min(len(ref_act) // 4, 512)
    f, Pref = signal.welch(ref_act, fs=fs, nperseg=nperseg)
    _, Py = signal.welch(y_act, fs=fs, nperseg=nperseg)
    
    ax_psd1.semilogy(f, Pref, '--', color=colors_chirp[idx], alpha=0.6, linewidth=1)
    ax_psd1.semilogy(f, Py, '-', color=colors_chirp[idx], alpha=0.9, linewidth=1.5, label=name)

ax_psd1.set_xlim([0, 8])
ax_psd1.set_xlabel('Frequência (Hz)')
ax_psd1.set_ylabel('PSD (°²/Hz)')
ax_psd1.set_title('Chirps / Swept-Sine (tracejado=ref, sólido=saída)')
ax_psd1.legend(fontsize=8, loc='upper right')
ax_psd1.grid(True)

# PSD Multi-senos
ax_psd2 = axes_psd[1]
for idx, (name, filepath) in enumerate(multisine_files.items()):
    t, y, u, ref, fs, dt = load_experiment(filepath)
    start_i, end_i = find_active_region(ref, t)
    ref_act = ref[start_i:end_i] - np.mean(ref[start_i:end_i])
    y_act = y[start_i:end_i] - np.mean(y[start_i:end_i])
    
    nperseg = min(len(ref_act) // 4, 512)
    f, Pref = signal.welch(ref_act, fs=fs, nperseg=nperseg)
    _, Py = signal.welch(y_act, fs=fs, nperseg=nperseg)
    
    ax_psd2.semilogy(f, Pref, '--', color=colors_multi[idx], alpha=0.6, linewidth=1)
    ax_psd2.semilogy(f, Py, '-', color=colors_multi[idx], alpha=0.9, linewidth=1.5, label=name)

ax_psd2.set_xlim([0, 8])
ax_psd2.set_xlabel('Frequência (Hz)')
ax_psd2.set_ylabel('PSD (°²/Hz)')
ax_psd2.set_title('Multi-Senos (tracejado=ref, sólido=saída)')
ax_psd2.legend(fontsize=8, loc='upper right')
ax_psd2.grid(True)

fig_psd.tight_layout()
fig_psd.savefig(OUTPUT_DIR / 'psd_comparativa.png', dpi=150, bbox_inches='tight')
print(f"Salvo: {OUTPUT_DIR / 'psd_comparativa.png'}")


# ============================================================================
# 5. Diagrama de Bode empírico (ETFE médio de todos os chirps)
# ============================================================================

fig_bode, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig_bode.suptitle('Diagrama de Bode Empírico (ETFE)\nMédia de todos os ensaios chirp/swept-sine',
                  fontsize=14, fontweight='bold', color='white')

all_H_mag = []
all_H_phase = []
all_f = None

for name, filepath in chirp_files.items():
    t, y, u, ref, fs, dt = load_experiment(filepath)
    start_i, end_i = find_active_region(ref, t)
    ref_act = ref[start_i:end_i]
    y_act = y[start_i:end_i]
    
    nperseg = min(len(ref_act) // 6, 512)
    f_coh, Cxy, f_tf, H_mag, H_phase, _, _, _ = \
        compute_coherence_and_etfe(ref_act, y_act, fs, nperseg=nperseg)
    
    ax_mag.semilogy(f_tf, H_mag, '-', alpha=0.4, linewidth=1, label=name)
    ax_ph.plot(f_tf, H_phase, '-', alpha=0.4, linewidth=1, label=name)
    
    # Interpolar para média
    if all_f is None:
        all_f = f_tf
    else:
        all_f = f_tf  # Assumir mesmo grid
    all_H_mag.append(H_mag)
    all_H_phase.append(H_phase)

# Média
if all_H_mag:
    min_len = min(len(h) for h in all_H_mag)
    H_mag_mean = np.mean([h[:min_len] for h in all_H_mag], axis=0)
    H_phase_mean = np.mean([h[:min_len] for h in all_H_phase], axis=0)
    f_plot = all_f[:min_len]
    
    ax_mag.semilogy(f_plot, H_mag_mean, 'w-', linewidth=2.5, label='Média', zorder=10)
    ax_ph.plot(f_plot, H_phase_mean, 'w-', linewidth=2.5, label='Média', zorder=10)
    
    # Marcar -3dB
    H_mag_db = 20 * np.log10(H_mag_mean + 1e-12)
    H_mag_db_ref = H_mag_db[1:10].mean()  # Ganho DC aproximado
    mask_3db = H_mag_db < (H_mag_db_ref - 3)
    if np.any(mask_3db):
        f_3db = f_plot[np.argmax(mask_3db)]
        ax_mag.axvline(f_3db, color='#ff6b6b', linestyle='--', linewidth=2,
                       label=f'-3dB: {f_3db:.2f} Hz ({f_3db*2*np.pi:.1f} rad/s)')
        ax_ph.axvline(f_3db, color='#ff6b6b', linestyle='--', linewidth=2)
        print(f"\n>>> Frequência de -3dB (ETFE médio): {f_3db:.3f} Hz ({f_3db*2*np.pi:.2f} rad/s)")
    
    # Marcar -6dB
    mask_6db = H_mag_db < (H_mag_db_ref - 6)
    if np.any(mask_6db):
        f_6db = f_plot[np.argmax(mask_6db)]
        ax_mag.axvline(f_6db, color='#ffa500', linestyle=':', linewidth=1.5,
                       label=f'-6dB: {f_6db:.2f} Hz ({f_6db*2*np.pi:.1f} rad/s)')
        print(f">>> Frequência de -6dB (ETFE médio): {f_6db:.3f} Hz ({f_6db*2*np.pi:.2f} rad/s)")

ax_mag.set_xlim([0.01, min(fs/2, 10)])
ax_mag.set_ylabel('|H(f)| (Magnitude)')
ax_mag.set_title('Magnitude')
ax_mag.legend(fontsize=8, loc='upper right')
ax_mag.grid(True)

ax_ph.set_xlim([0.01, min(fs/2, 10)])
ax_ph.set_xlabel('Frequência (Hz)')
ax_ph.set_ylabel('Fase (°)')
ax_ph.set_title('Fase')
ax_ph.axhline(-90, color='gray', linestyle=':', alpha=0.5)
ax_ph.axhline(-180, color='gray', linestyle=':', alpha=0.5)
ax_ph.legend(fontsize=8, loc='upper right')
ax_ph.grid(True)

fig_bode.tight_layout()
fig_bode.savefig(OUTPUT_DIR / 'bode_empirico.png', dpi=150, bbox_inches='tight')
print(f"Salvo: {OUTPUT_DIR / 'bode_empirico.png'}")


# ============================================================================
# 6. Análise da coerência MALHA ABERTA: u_pct -> angulo_deg
# ============================================================================

print("\n" + "=" * 70)
print("ANÁLISE DE COERÊNCIA EM MALHA ABERTA (u -> y)")
print("=" * 70)

fig_oa, axes_oa = plt.subplots(len(chirp_files), 2, figsize=(16, 4 * len(chirp_files)))
fig_oa.suptitle('Coerência e ETFE em Malha Aberta (u→y)\n(Essencial para identificação)',
                fontsize=15, fontweight='bold', color='white', y=0.98)

oa_bandwidth_results = {}

for idx, (name, filepath) in enumerate(chirp_files.items()):
    t, y, u, ref, fs, dt = load_experiment(filepath)
    start_i, end_i = find_active_region(ref, t)
    
    y_act = y[start_i:end_i]
    u_act = u[start_i:end_i]
    
    nperseg = min(len(y_act) // 6, 512)
    
    # Coerência u -> y
    u_zm = u_act - np.mean(u_act)
    y_zm = y_act - np.mean(y_act)
    
    f_coh, Cxy = signal.coherence(u_zm, y_zm, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    
    # ETFE u -> y
    f_tf, Puy = signal.csd(u_zm, y_zm, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    _, Puu = signal.welch(u_zm, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    H_uy = Puy / (Puu + 1e-12)
    H_uy_mag = np.abs(H_uy)
    
    # Frequências limite
    coh_above_05 = f_coh[Cxy >= 0.5]
    coh_above_07 = f_coh[Cxy >= 0.7]
    f_max_05 = coh_above_05[-1] if len(coh_above_05) > 0 else 0
    f_max_07 = coh_above_07[-1] if len(coh_above_07) > 0 else 0
    
    oa_bandwidth_results[name] = {'f_coh_0.5': f_max_05, 'f_coh_0.7': f_max_07}
    
    print(f"\n  {name}:")
    print(f"    f_max (Cxy >= 0.5): {f_max_05:.3f} Hz ({f_max_05*2*np.pi:.2f} rad/s)")
    print(f"    f_max (Cxy >= 0.7): {f_max_07:.3f} Hz ({f_max_07*2*np.pi:.2f} rad/s)")
    
    # Plot
    ax1 = axes_oa[idx, 0]
    ax1.plot(f_coh, Cxy, 'lime', linewidth=1.5)
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    ax1.axhline(0.7, color='orange', linestyle='--', alpha=0.7)
    if f_max_05 > 0:
        ax1.axvline(f_max_05, color='red', linestyle=':', alpha=0.5,
                    label=f'Cxy≥0.5: {f_max_05:.2f} Hz')
    if f_max_07 > 0:
        ax1.axvline(f_max_07, color='orange', linestyle=':', alpha=0.5,
                    label=f'Cxy≥0.7: {f_max_07:.2f} Hz')
    ax1.set_xlim([0, min(fs/2, 10)])
    ax1.set_ylim([0, 1.05])
    ax1.set_xlabel('Frequência (Hz)')
    ax1.set_ylabel('Coerência $C_{uy}^2$')
    ax1.set_title(f'{name} - Coerência u→y')
    ax1.legend(fontsize=8)
    ax1.grid(True)
    
    ax2 = axes_oa[idx, 1]
    ax2.semilogy(f_tf, H_uy_mag, 'lime', linewidth=1.5)
    if f_max_05 > 0:
        ax2.axvline(f_max_05, color='red', linestyle=':', alpha=0.7)
    if f_max_07 > 0:
        ax2.axvline(f_max_07, color='orange', linestyle=':', alpha=0.7)
    ax2.set_xlim([0, min(fs/2, 10)])
    ax2.set_xlabel('Frequência (Hz)')
    ax2.set_ylabel('|H(f)| u→y')
    ax2.set_title(f'{name} - ETFE u→y')
    ax2.grid(True)

fig_oa.tight_layout(rect=[0, 0, 1, 0.96])
fig_oa.savefig(OUTPUT_DIR / 'coerencia_malha_aberta.png', dpi=150, bbox_inches='tight')
print(f"\nSalvo: {OUTPUT_DIR / 'coerencia_malha_aberta.png'}")


# ============================================================================
# 7. Resumo Final
# ============================================================================

print("\n" + "=" * 70)
print("RESUMO: FREQUÊNCIA MÁXIMA PARA MODELAGEM")
print("=" * 70)

print("\n--- Coerência ref→y (malha fechada) ---")
for name, vals in coherence_thresholds.items():
    print(f"  {name}:")
    print(f"    Cxy ≥ 0.5: {vals['f_coh_0.5']:.3f} Hz ({vals['f_coh_0.5']*2*np.pi:.2f} rad/s)")
    print(f"    Cxy ≥ 0.7: {vals['f_coh_0.7']:.3f} Hz ({vals['f_coh_0.7']*2*np.pi:.2f} rad/s)")

print("\n--- Coerência u→y (malha aberta) ---")
for name, vals in oa_bandwidth_results.items():
    print(f"  {name}:")
    print(f"    Cxy ≥ 0.5: {vals['f_coh_0.5']:.3f} Hz ({vals['f_coh_0.5']*2*np.pi:.2f} rad/s)")
    print(f"    Cxy ≥ 0.7: {vals['f_coh_0.7']:.3f} Hz ({vals['f_coh_0.7']*2*np.pi:.2f} rad/s)")

# Valores médios
all_f05_cl = [v['f_coh_0.5'] for v in coherence_thresholds.values() if v['f_coh_0.5'] > 0]
all_f07_cl = [v['f_coh_0.7'] for v in coherence_thresholds.values() if v['f_coh_0.7'] > 0]
all_f05_oa = [v['f_coh_0.5'] for v in oa_bandwidth_results.values() if v['f_coh_0.5'] > 0]
all_f07_oa = [v['f_coh_0.7'] for v in oa_bandwidth_results.values() if v['f_coh_0.7'] > 0]

print("\n" + "=" * 70)
print(">>> RECOMENDAÇÃO FINAL <<<")
print("=" * 70)

if all_f05_oa:
    f_rec_min = np.min(all_f05_oa)
    f_rec_median = np.median(all_f05_oa)
    f_rec_max = np.max(all_f05_oa)
    print(f"\nCoerência u→y ≥ 0.5 (mín/med/máx): {f_rec_min:.2f} / {f_rec_median:.2f} / {f_rec_max:.2f} Hz")
    print(f"                                   {f_rec_min*2*np.pi:.1f} / {f_rec_median*2*np.pi:.1f} / {f_rec_max*2*np.pi:.1f} rad/s")

if all_f07_oa:
    f_rec_min7 = np.min(all_f07_oa)
    f_rec_median7 = np.median(all_f07_oa)
    f_rec_max7 = np.max(all_f07_oa)
    print(f"\nCoerência u→y ≥ 0.7 (mín/med/máx): {f_rec_min7:.2f} / {f_rec_median7:.2f} / {f_rec_max7:.2f} Hz")
    print(f"                                   {f_rec_min7*2*np.pi:.1f} / {f_rec_median7*2*np.pi:.1f} / {f_rec_max7*2*np.pi:.1f} rad/s")

if all_f05_cl:
    print(f"\nCoerência ref→y ≥ 0.5 (mediana): {np.median(all_f05_cl):.2f} Hz ({np.median(all_f05_cl)*2*np.pi:.1f} rad/s)")
if all_f07_cl:
    print(f"Coerência ref→y ≥ 0.7 (mediana): {np.median(all_f07_cl):.2f} Hz ({np.median(all_f07_cl)*2*np.pi:.1f} rad/s)")

print("\n" + "-" * 70)
print("INTERPRETAÇÃO:")
print("-" * 70)
print("""
Para identificação de sistemas:
  - Faixa SEGURA para modelar: até a freq onde coerência u→y ≥ 0.7
  - Faixa LIMITE para modelar:  até a freq onde coerência u→y ≥ 0.5
  - Acima disso: ruído domina, modelo linear não é confiável

Para projeto de controlador:
  - Largura de banda do controlador deve ser ≤ freq de coerência 0.7
  - Margem de segurança: usar ~70-80% da freq limite
  
Observação visual dos chirps:
  - O tracking degrada visivelmente quando a frequência do chirp aumenta
  - Na swept-sine, o erro explode e o controle satura nas frequências mais altas
  - Isso confirma que o aeropêndulo tem banda passante limitada
""")

plt.close('all')
print("\n=== ANÁLISE CONCLUÍDA ===")
print(f"Gráficos salvos em: {OUTPUT_DIR}")
