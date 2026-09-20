"""
test_freerun_extended.py — Teste estendido de modelos NODE

Auto-detecta todos os arquivos .pth na pasta de resultados e avalia
cada modelo em FREE-RUN e CHUNK-FIT contra datasets de teste variados.

Free-run com proteção: divide a integração em segmentos e aborta se
o modelo divergir, evitando que o solver trave.
"""
import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from node_v9 import (
    carregar_lista, avalia_chunks, plot_chunk_fit, device,
    PhysicsODE_AeroBaseline, PhysicsODE_AeroCoulomb,
    PhysicsODE_AeroTustin, PhysicsODE_AeroHibrido,
)
from torchdiffeq import odeint

# ──────────────────────────────────────────────────────────────────────
# Mapeamento: substring no nome do .pth → classe do modelo
# ──────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = [
    ("AeroBaseline",  PhysicsODE_AeroBaseline),
    ("AeroCoulomb",   PhysicsODE_AeroCoulomb),
    ("AeroTustin",    PhysicsODE_AeroTustin),
    ("AeroHibrido",   PhysicsODE_AeroHibrido),
]


def infer_model_class(pth_name):
    for keyword, ModelClass in MODEL_REGISTRY:
        if keyword in pth_name:
            return ModelClass
    return None


# ──────────────────────────────────────────────────────────────────────
# Free-Run SEGURO — integra em segmentos e aborta se divergir
# ──────────────────────────────────────────────────────────────────────
ANGLE_LIMIT_RAD = 10.0  # ~573° — se ultrapassar, modelo divergiu

def avalia_free_run_safe(model, datasets, integrator='rk4', segment_steps=500):
    """Free-run com proteção: integra em segmentos e aborta se divergir.
    
    Em vez de integrar 10000+ passos de uma vez (que trava o solver se
    o modelo divergir), integra em segmentos de `segment_steps` passos.
    Após cada segmento, verifica se houve divergência (NaN ou ângulo fora
    dos limites físicos). Se divergiu, preenche o resto com NaN e segue.
    """
    model.eval()
    resultados = []
    with torch.no_grad():
        for ds in datasets:
            t_t = ds['t']
            u_t = ds['u']
            x_t = ds['x']
            model.t_series = t_t
            model.u_series = u_t

            N = len(t_t)
            dt = float(t_t[1] - t_t[0])
            y_pred_full = np.full(N, np.nan)
            y_real = x_t[:, 0].cpu().numpy() * (180 / np.pi)

            # Condição inicial real
            x_curr = x_t[0].unsqueeze(0)
            divergiu = False

            # Integrar em segmentos
            seg_start = 0
            while seg_start < N - 1 and not divergiu:
                seg_end = min(seg_start + segment_steps, N)
                seg_len = seg_end - seg_start
                t_seg = torch.arange(0, seg_len * dt, dt, device=t_t.device)[:seg_len]

                model.batch_start_times = t_t[seg_start].reshape(1, 1)

                try:
                    pred = odeint(model, x_curr, t_seg, method=integrator).squeeze(1)
                except Exception:
                    divergiu = True
                    break

                # Checar divergência
                theta_pred = pred[:, 0]
                if torch.any(torch.isnan(theta_pred)) or torch.any(torch.abs(theta_pred) > ANGLE_LIMIT_RAD):
                    # Salvar até onde deu e marcar divergência
                    valid_mask = (~torch.isnan(theta_pred)) & (torch.abs(theta_pred) <= ANGLE_LIMIT_RAD)
                    valid_idx = torch.where(valid_mask)[0]
                    if len(valid_idx) > 0:
                        last_valid = valid_idx[-1].item() + 1
                        y_pred_full[seg_start:seg_start + last_valid] = \
                            theta_pred[:last_valid].cpu().numpy() * (180 / np.pi)
                    divergiu = True
                    break

                y_pred_full[seg_start:seg_end] = theta_pred.cpu().numpy() * (180 / np.pi)
                # Usar último estado como CI do próximo segmento
                x_curr = pred[-1:, :]
                seg_start = seg_end

            # Calcular métricas só na parte válida
            mask = ~np.isnan(y_pred_full)
            if np.sum(mask) > 10:
                y_r = y_real[mask]
                y_p = y_pred_full[mask]
                rmse = float(np.sqrt(np.mean((y_p - y_r) ** 2)))
                ss_res = np.sum((y_r - y_p) ** 2)
                ss_tot = np.sum((y_r - np.mean(y_r)) ** 2)
                r2 = float(1 - ss_res / (ss_tot + 1e-12))
                fit = float((1 - np.linalg.norm(y_p - y_r) /
                             (np.linalg.norm(y_r - np.mean(y_r)) + 1e-12)) * 100)
            else:
                rmse, r2, fit = 999.0, -999.0, -999.0

            cobertura = np.sum(mask) / N * 100
            resultados.append({
                'name': ds['name'], 'rmse': rmse, 'r2': r2, 'fit': fit,
                'y_real': y_real, 'y_pred': y_pred_full,
                't': t_t.cpu().numpy(),
                'divergiu': divergiu, 'cobertura': cobertura,
            })
    return resultados


def plot_free_run_safe(resultado, titulo, out_path=None):
    """Plot de free-run com indicação visual de divergência."""
    n = len(resultado)
    cols = min(3, n)
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    fig.suptitle(titulo, fontsize=13)
    axs = np.array(axs).reshape(rows, cols) if n > 1 else np.array([[axs]])

    for i, r in enumerate(resultado):
        ax = axs[i // cols, i % cols]
        t = r['t']
        ax.plot(t, r['y_real'], 'k', lw=1.2, label='Real')

        y_pred = r['y_pred']
        mask = ~np.isnan(y_pred)
        t_valid = t[mask]
        y_valid = y_pred[mask]
        
        color = 'darkorange' if r['divergiu'] else 'r'
        ls = '-' if r['divergiu'] else '--'
        ax.plot(t_valid, y_valid, color=color, ls=ls, lw=1.0, label='Pred')

        if r['divergiu']:
            # Marcar ponto de divergência
            if len(t_valid) > 0:
                ax.axvline(t_valid[-1], color='red', lw=1.5, ls='--', alpha=0.7)
                ax.text(t_valid[-1], ax.get_ylim()[1] * 0.95, ' DIVERGIU',
                        color='red', fontsize=7, fontweight='bold', va='top')

        div_tag = " ⚠DIVERGIU" if r['divergiu'] else ""
        ax.set_title(
            f"{r['name']}{div_tag}\n"
            f"RMSE={r['rmse']:.2f}° R²={r['r2']:.3f} FIT={r['fit']:.1f}% "
            f"Cob={r['cobertura']:.0f}%",
            fontsize=8)
        ax.set_xlabel('Tempo (s)', fontsize=7)
        ax.set_ylabel('Ângulo (°)', fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.4)

    for i in range(n, rows * cols):
        axs[i // cols, i % cols].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if out_path:
        plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Datasets de teste estendidos (rodadas 2 a 5)
# ──────────────────────────────────────────────────────────────────────
EXTENDED_TESTS = {
    "APRBS": [
        "RODADA-4/aprbs-2_0819_18-51.csv",
        "RODADA-5/aprbs-1_0827_17-19.csv",
        "RODADA-5/aprbs-2_0827_17-25.csv",
        "RODADA-5/aprbs-3_0827_17-28.csv"
    ],
    "MultiSeno": [
        "RODADA-2/multi-seno-1_0804_19-03.csv",
        "RODADA-4/multi-seno-1_0819_19-23.csv",
        "RODADA-5/multi-seno-2_0827_17-37.csv",
        "RODADA-5/multi-seno-3_0827_17-40.csv"
    ],
    "Varredura": [
        "RODADA-2/chirp-1_0804_19-17.csv",
        "RODADA-3/chirp-1_0807_16-32.csv",
        "RODADA-3/chirp-1_0807_16-34.csv",
        "RODADA-5/swept-sine-1_0827_17-58.csv"
    ],
    "Degraus": [
        "RODADA-2/seq-degraus-1_0804_19-09.csv",
        "RODADA-3/seq-degraus-1_0807_16-38.csv",
        "RODADA-5/seq-degraus-3_0827_17-52.csv",
        "RODADA-5/seq-degraus-4_0827_17-55.csv"
    ]
}

# ──────────────────────────────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────────────────────────────
RESULT_DIR = "resultados_v9_20260916_215831"
OUT_DIR = f"{RESULT_DIR}_extended_test"
K_STEPS = 400
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Auto-detectar modelos
# ──────────────────────────────────────────────────────────────────────
pth_files = sorted(glob.glob(os.path.join(RESULT_DIR, "model_*.pth")))
if not pth_files:
    print(f"Nenhum arquivo .pth encontrado em {RESULT_DIR}/")
    exit(1)

print(f"Encontrados {len(pth_files)} modelos em {RESULT_DIR}/:")
for f in pth_files:
    print(f"  • {os.path.basename(f)}")

# ──────────────────────────────────────────────────────────────────────
# Carregar datasets de teste
# ──────────────────────────────────────────────────────────────────────
print("\nCarregando datasets estendidos...")
test_por_tipo = {tipo: carregar_lista(arqs, to_device=device)
                 for tipo, arqs in EXTENDED_TESTS.items()}

# ──────────────────────────────────────────────────────────────────────
# Avaliar cada modelo
# ──────────────────────────────────────────────────────────────────────
resumo_freerun = {}
resumo_chunk = {}

for pth_path in pth_files:
    pth_name = os.path.basename(pth_path)
    tag = pth_name.replace("model_", "").replace(".pth", "")

    ModelClass = infer_model_class(pth_name)
    if ModelClass is None:
        print(f"\n⚠ Classe não identificada: {pth_name}")
        continue

    print(f"\n{'='*60}")
    print(f"  Avaliando: {tag}  ({ModelClass.__name__})")
    print(f"{'='*60}")

    model = ModelClass().to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    model.eval()

    # ── 1) FREE-RUN (com proteção) ──
    print(f"\n  --- Free-Run (segmentado, com proteção) ---")
    todos_freerun = []
    por_tipo_fr = {}
    for test_tipo, test_ds in test_por_tipo.items():
        res = avalia_free_run_safe(model, test_ds, segment_steps=500)
        todos_freerun.extend(res)
        rmse_m = np.mean([r['rmse'] for r in res])
        r2_m   = np.mean([r['r2']   for r in res])
        fit_m  = np.mean([r['fit']  for r in res])
        n_div  = sum(1 for r in res if r['divergiu'])
        por_tipo_fr[test_tipo] = {'rmse': rmse_m, 'r2': r2_m, 'fit': fit_m}
        div_str = f" ({n_div}/{len(res)} divergiram)" if n_div > 0 else ""
        print(f"  [{test_tipo:<10}] RMSE={rmse_m:>8.2f}°  R²={r2_m:>8.3f}  FIT={fit_m:>6.1f}%{div_str}")

    rmse_g = np.mean([v['rmse'] for v in por_tipo_fr.values()])
    r2_g   = np.mean([v['r2']   for v in por_tipo_fr.values()])
    fit_g  = np.mean([v['fit']  for v in por_tipo_fr.values()])
    n_div_total = sum(1 for r in todos_freerun if r['divergiu'])
    print(f"  {'MÉDIA':<10}  RMSE={rmse_g:>8.2f}°  R²={r2_g:>8.3f}  FIT={fit_g:>6.1f}%  (div: {n_div_total}/{len(todos_freerun)})")
    resumo_freerun[tag] = {'rmse': rmse_g, 'r2': r2_g, 'fit': fit_g, 'divergencias': n_div_total}

    plot_free_run_safe(todos_freerun, f"{tag} — Free-Run Extended",
                       out_path=f"{OUT_DIR}/freerun_{tag}_extended.png")

    # ── 2) CHUNK-FIT (k=400) ──
    print(f"\n  --- Chunk Fit (k={K_STEPS}) ---")
    todos_chunk = []
    por_tipo_ch = {}
    for test_tipo, test_ds in test_por_tipo.items():
        res = avalia_chunks(model, test_ds, k_steps=K_STEPS)
        todos_chunk.extend(res)
        rmse_m = np.mean([r['rmse'] for r in res])
        r2_m   = np.mean([r['r2']   for r in res])
        fit_m  = np.mean([r['fit']  for r in res])
        por_tipo_ch[test_tipo] = {'rmse': rmse_m, 'r2': r2_m, 'fit': fit_m}
        print(f"  [{test_tipo:<10}] RMSE={rmse_m:>8.2f}°  R²={r2_m:>8.3f}  FIT={fit_m:>6.1f}%")

    rmse_g = np.mean([v['rmse'] for v in por_tipo_ch.values()])
    r2_g   = np.mean([v['r2']   for v in por_tipo_ch.values()])
    fit_g  = np.mean([v['fit']  for v in por_tipo_ch.values()])
    print(f"  {'MÉDIA':<10}  RMSE={rmse_g:>8.2f}°  R²={r2_g:>8.3f}  FIT={fit_g:>6.1f}%")
    resumo_chunk[tag] = {'rmse': rmse_g, 'r2': r2_g, 'fit': fit_g}

    plot_chunk_fit(todos_chunk, f"{tag} — Chunk Fit Extended (k={K_STEPS})",
                   out_path=f"{OUT_DIR}/chunkfit_{tag}_extended.png")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ──────────────────────────────────────────────────────────────────────
# Resumo final comparativo
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  RESUMO COMPARATIVO")
print(f"{'='*70}")
print(f"\n  {'Modelo':<30}  {'Free-Run RMSE':>13}  {'Chunk RMSE':>10}  {'Chunk FIT':>10}")
print(f"  {'─'*70}")
for tag in resumo_freerun:
    fr = resumo_freerun[tag]
    ch = resumo_chunk.get(tag, {'rmse': 0, 'fit': 0})
    div_str = f" ({fr['divergencias']}div)" if fr['divergencias'] > 0 else ""
    print(f"  {tag:<30}  {fr['rmse']:>8.2f}°{div_str:<5}  {ch['rmse']:>8.2f}°  {ch['fit']:>8.1f}%")

print(f"\n  Gráficos salvos em {OUT_DIR}/")
