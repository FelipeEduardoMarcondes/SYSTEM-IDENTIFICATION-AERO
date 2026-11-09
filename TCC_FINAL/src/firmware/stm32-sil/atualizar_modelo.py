"""
Script utilitário para atualizar o modelo neural X-CUBE-AI no firmware STM32.
Executa diretamente o gerador oficial da ST (stedgeai.exe), sem precisar abrir
o STM32CubeMX nem sofrer com sobrescrita de arquivos do projeto.

Uso:
    python firmware/stm32-sil/atualizar_modelo.py --compression high
    python firmware/stm32-sil/atualizar_modelo.py --compression none
    python firmware/stm32-sil/atualizar_modelo.py -m caminho/meu_modelo.onnx --compression high
"""

import os
import sys
import subprocess
import argparse

STEDGEAI_EXE = r"C:/Users/vicio/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/windows/stedgeai.exe"
DEFAULT_MODEL = r"data/experimentos_novos/EXP001_otimizacao_narx/3_sil_stm32/ann_model.onnx"
OUTPUT_DIR = r"firmware/stm32-sil/X-CUBE-AI/App"

def main():
    parser = argparse.ArgumentParser(description="Atualiza modelo X-CUBE-AI no firmware STM32")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Caminho do modelo ONNX")
    parser.add_argument("-c", "--compression", default="high", choices=["none", "low", "medium", "high"], help="Nivel de compressao")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    output_dir = os.path.abspath(OUTPUT_DIR)

    if not os.path.exists(STEDGEAI_EXE):
        print(f"[ERRO] Executavel stedgeai nao encontrado em: {STEDGEAI_EXE}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[ERRO] Modelo ONNX nao encontrado em: {model_path}")
        sys.exit(1)

    print(f"\n==================================================")
    print(f"  Gerando codigo C otimizado via ST Edge AI Core  ")
    print(f"==================================================")
    print(f"  Modelo:      {model_path}")
    print(f"  Compressao:  {args.compression}")
    print(f"  Destino:     {output_dir}")
    print(f"--------------------------------------------------")

    cmd = [
        STEDGEAI_EXE,
        "generate",
        "--target", "stm32f4",
        "--name", "network",
        "-m", model_path,
        "--compression", args.compression,
        "--output", output_dir,
        "--optimization", "balanced",
        "--verbosity", "1",
        "--no-report"
    ]

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("\n[ERRO] Falha na geracao do codigo pelo stedgeai!")
        sys.exit(res.returncode)

    # Garante que os includes em app_x-cube-ai.h estejam corretos caso tenham sido afetados
    h_file = os.path.join(output_dir, "app_x-cube-ai.h")
    if os.path.exists(h_file):
        with open(h_file, "r", encoding="utf-8") as f:
            content = f.read()
        if '#include ".h"' in content:
            content = content.replace('#include ".h"', '#include "network.h"')
            content = content.replace('#include "_data.h"', '#include "network_data.h"')
            with open(h_file, "w", encoding="utf-8") as f:
                f.write(content)

    print("\n[SUCESSO] Codigo C do modelo gerado com sucesso!")
    print("Agora basta dar 'Build' ou 'Run' no STM32CubeIDE!")

if __name__ == "__main__":
    main()
