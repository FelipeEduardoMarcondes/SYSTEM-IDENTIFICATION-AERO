"""
novo_experimento.py — Cria a estrutura de pastas para um novo experimento.

Uso:
    python novo_experimento.py

Fluxo interativo:
    1. Pede o nome/descrição do experimento
    2. Pede o tipo de modelo
    3. Auto-incrementa o ID (EXP001, EXP002, ...)
    4. Cria a pasta com subpastas e README/config.json preenchidos
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
EXPERIMENTS_DIR = ROOT / "data" / "experimentos_novos"


def get_next_exp_id() -> int:
    """Encontra o próximo ID de experimento disponível."""
    if not EXPERIMENTS_DIR.exists():
        return 1

    max_id = 0
    for item in EXPERIMENTS_DIR.iterdir():
        if item.is_dir():
            match = re.match(r"EXP(\d+)", item.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def criar_experimento(exp_id: int, nome: str, modelo_tipo: str, modelo_script: str):
    """Cria a estrutura completa do experimento."""

    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", nome.lower().strip())
    folder_name = f"EXP{exp_id:03d}_{slug}"
    exp_dir = EXPERIMENTS_DIR / folder_name

    if exp_dir.exists():
        print(f"  ❌ Pasta {exp_dir} já existe!")
        return

    # Criar subpastas
    subdirs = [
        "1_identificacao",
        "2_mpc_python",
        "3_sil_stm32",
        "4_aeropendulo",
    ]
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Criar README.md
    data_hoje = datetime.now().strftime("%Y-%m-%d")
    readme = f"""# {folder_name}

| Campo           | Valor                                    |
|-----------------|------------------------------------------|
| **Data início** | {data_hoje}                              |
| **Modelo**      | {modelo_tipo} ({modelo_script})          |
| **Dados treino**| [Preencher]                              |
| **Dados teste** | [Preencher]                              |
| **Status**      | 🔄 Em andamento                          |

## Resultados Resumidos

| Plataforma     | RMSE (°) | R²     | Notas            |
|----------------|----------|--------|------------------|
| Python (ident) |          |        |                  |
| Python (MPC)   |          |        |                  |
| STM32 SIL      |          |        |                  |
| Aeropêndulo    |          |        |                  |

## Checklist

- [ ] Treinar modelo (`1_identificacao/`)
- [ ] Salvar checkpoint + métricas
- [ ] Simular MPC no Python (`2_mpc_python/`)
- [ ] Exportar pesos para C (`3_sil_stm32/ann_weights.h`)
- [ ] Rodar SIL na STM32 (`3_sil_stm32/`)
- [ ] Comparar Python vs STM
- [ ] Testar no aeropêndulo real (`4_aeropendulo/`)
- [ ] Preencher tabela de resultados acima

## Observações

- [Anotar decisões, problemas, insights aqui]
"""

    # Criar config.json
    config = {
        "experimento_id": f"EXP{exp_id:03d}",
        "nome": nome,
        "data_criacao": data_hoje,
        "modelo": {
            "tipo": modelo_tipo,
            "script": modelo_script,
            "versao": "",
            "n_parametros": 0,
        },
        "treinamento": {
            "epocas": 2000,
            "lr_inicial": 0.015,
            "scheduler": "CosineAnnealing",
            "seed": 0,
            "batch_size": 1024,
            "k_min": 20,
            "k_max": 400,
        },
        "dados": {
            "treino": [],
            "validacao": [],
            "teste": [],
        },
        "coletas_usadas": [],
    }

    # Criar .gitkeep nos subdiretórios para que o git os rastreie
    for subdir in subdirs:
        gitkeep = exp_dir / subdir / ".gitkeep"
        gitkeep.write_text("", encoding="utf-8")

    # Salvar README e config
    (exp_dir / "README.md").write_text(readme, encoding="utf-8")
    (exp_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print(f"  ✅ Experimento criado: {exp_dir}")
    print()
    print(f"  Estrutura:")
    print(f"  {folder_name}/")
    print(f"  ├── README.md")
    print(f"  ├── config.json")
    for i, subdir in enumerate(subdirs):
        prefix = "└──" if i == len(subdirs) - 1 else "├──"
        print(f"  {prefix} {subdir}/")
    print()
    print(f"  Próximos passos:")
    print(f"  1. Edite config.json com os dados de treino/teste")
    print(f"  2. Rode o treinamento e salve resultados em 1_identificacao/")
    print(f"  3. Vá avançando nas etapas 2 → 3 → 4")


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           Criar Novo Experimento                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Auto-incrementar ID
    exp_id = get_next_exp_id()
    print(f"  Próximo ID disponível: EXP{exp_id:03d}")
    print()

    # Perguntar nome
    nome = input("  Nome/descrição curta do experimento:\n  > ").strip()
    if not nome:
        print("  ❌ Nome não pode ser vazio!")
        return

    # Perguntar modelo
    print()
    print("  Modelos disponíveis:")
    modelos = [
        ("PhysicsODE_Asymmetric (NODE)", "sysid/scripts/node_v9.py"),
        ("NARX", "sysid/scripts/narx_v6.py"),
        ("NODE + Motor Dynamics", "sysid/scripts/node_v3_motor_dyn.py"),
        ("GRU", "sysid/scripts/identificacao_gru.py"),
        ("Hammerstein", "sysid/scripts/identificacao_hammerstein.py"),
        ("Outro (digitar)", ""),
    ]
    for i, (m_nome, _) in enumerate(modelos, 1):
        print(f"    {i}. {m_nome}")

    print()
    escolha = input("  Escolha o modelo (número): ").strip()

    try:
        idx = int(escolha) - 1
        if idx < 0 or idx >= len(modelos):
            raise ValueError()
    except ValueError:
        print("  ❌ Opção inválida!")
        return

    if idx == len(modelos) - 1:
        modelo_tipo = input("  Tipo do modelo: ").strip()
        modelo_script = input("  Caminho do script: ").strip()
    else:
        modelo_tipo, modelo_script = modelos[idx]

    print()
    print(f"  Criando EXP{exp_id:03d}_{nome}...")
    criar_experimento(exp_id, nome, modelo_tipo, modelo_script)


if __name__ == "__main__":
    main()
