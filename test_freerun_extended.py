import os
import torch
import numpy as np
from node_v7 import (
    MODELOS, EXCITACOES, carregar_lista, avalia_free_run, plot_free_run, device
)

# Definir um conjunto estendido de testes das rodadas 2 a 5
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

RESULT_DIR = "resultados_v7_20260906_224337"
OUT_DIR = "resultados_v7_20260906_224337_extended_test"
os.makedirs(OUT_DIR, exist_ok=True)

print("Carregando datasets estendidos...")
test_por_tipo = {tipo: carregar_lista(arqs) for tipo, arqs in EXTENDED_TESTS.items()}

mod_nomes = list(MODELOS.keys())
exc_nomes = list(EXCITACOES.keys())

for mod_nome, ModelClass in MODELOS.items():
    for exc_nome in exc_nomes:
        tag = f"{mod_nome}_{exc_nome}"
        pth_file = os.path.join(RESULT_DIR, f"model_{tag}.pth")
        
        if not os.path.exists(pth_file):
            print(f"Skipping {tag}, model not found.")
            continue
            
        print(f"\n======================================")
        print(f" Avaliando {tag}")
        print(f"======================================")
        
        # Load model
        model = ModelClass().to(device)
        model.load_state_dict(torch.load(pth_file, map_location=device))
        model.eval()
        
        todos_res = []
        
        for test_tipo, test_ds in test_por_tipo.items():
            res = avalia_free_run(model, test_ds)
            todos_res.extend(res)
            rmse_m = np.mean([r['rmse'] for r in res])
            r2_m   = np.mean([r['r2']   for r in res])
            fit_m  = np.mean([r['fit']  for r in res])
            
            print(f"  [TESTE {test_tipo:<10}] RMSE={rmse_m:>8.2f}° R²={r2_m:>8.3f} FIT={fit_m:>6.1f}%")

        # Free-run plot
        plot_free_run(todos_res, f"{tag} - Extended Test", out_path=f"{OUT_DIR}/freerun_{tag}_extended.png")
        
        # Liberar memória
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"\nTeste concluído. Gráficos salvos em {OUT_DIR}/")
