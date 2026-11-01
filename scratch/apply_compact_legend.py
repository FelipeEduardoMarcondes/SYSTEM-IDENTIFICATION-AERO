import os
import re

narx_files = [
    'sysid/notebooks/identificação-collab/IDENT-HELON/narx.py',
    'fluxograma-tcc/04_modelagem_dinamica/1_narx_polinomial/narx.py',
    'fluxograma-tcc/modelagem-narx/narx/narx.py'
]

new_plot_single_validation = r'''def plot_single_validation(res, filename_base):
    """Gera gráfico individual de validação com 2 subplots (Ângulo e Controle) com legenda compacta e headroom."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.3), sharex=True,
                                   gridspec_kw={'height_ratios': [2.2, 1.2]})
    
    t = res['t']
    y_true = res['y_true']
    y_fr = res['y_fr']
    u = res['u']
    
    # Subplot Superior: Ângulo
    ax1.plot(t, y_true, color='#111111', lw=1.1, label=r'Medido $y(k)$')
    ax1.plot(t, y_fr, color='#d62728', ls='--', lw=1.3, label=r'NARX Free-Run $\hat{y}(k)$')
    if not np.all(np.isnan(res['ref'])):
        ax1.plot(t, res['ref'], color='#7f7f7f', ls=':', lw=0.9, alpha=0.7, label=r'Referência $r(k)$')
        
    ax1.set_ylabel(r'Ângulo $\theta$ [$^\circ$]')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Legenda compacta (tamanho reduzido para não sobrepor a onda)
    ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.92,
               borderpad=0.25, labelspacing=0.2, handlelength=1.2, handletextpad=0.35)
    ax1.set_xlim(0, t[-1])
    
    # Margem superior (headroom) para a onda não sobrepor a legenda nem as métricas
    y_peak = max(float(np.max(y_true)), float(np.max(y_fr)))
    y_valley = min(float(np.min(y_true)), float(np.min(y_fr)))
    margin_top = max(16.0, (y_peak - y_valley) * 0.20)
    margin_bottom = max(5.0, (y_peak - y_valley) * 0.05)
    ax1.set_ylim(y_valley - margin_bottom, y_peak + margin_top)
    
    # Caixa com métricas compacta
    metric_text = f"RMSE = {res['rmse']:.3f}°\nMAE = {res['mae']:.3f}°\nFIT = {res['fit']:.1f}%"
    ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=7.5,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.85', alpha=0.92))
    
    # Subplot Inferior: Controle
    ax2.plot(t, u, color='#1f77b4', lw=1.0, label=r'Excitação $u(k)$')
    ax2.set_xlabel('Tempo [s]')
    ax2.set_ylabel(r'Controle $u$ [%]')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.92,
               borderpad=0.25, labelspacing=0.2, handlelength=1.2)
    
    u_peak = float(np.max(u))
    u_valley = float(np.min(u))
    ax2.set_ylim(u_valley - 4.0, u_peak + max(12.0, (u_peak - u_valley) * 0.22))
    
    plt.tight_layout()
    save_academic_plot(fig, filename_base)
    plt.close(fig)'''

for path in narx_files:
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'def plot_single_validation\(res, filename_base\):[\s\S]*?plt\.close\(fig\)'
    content_new, count = re.subn(pattern, lambda m: new_plot_single_validation, content)
    if count > 0:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content_new)
        print(f'Successfully updated plot_single_validation in {path}')
    else:
        print(f'Warning: pattern not matched in {path}')

print('Script update completed!')
