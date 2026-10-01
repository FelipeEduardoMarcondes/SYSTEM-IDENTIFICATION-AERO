"""
migrar_estrutura.py — Migração da estrutura de pastas para o novo formato.

Uso:
    python migrar_estrutura.py              # Modo DRY-RUN (só mostra o que faria)
    python migrar_estrutura.py --execute    # Executa de verdade (COPIA, não move)

O que faz:
    1. Cria data/coletas/ com as RODADAs renomeadas (RODADA-N_YYYYMMDD)
    2. Cria data/sinais_referencia/ com os CSVs templates soltos
    3. Cria data/experimentos/ com estrutura de EXP### (template vazio)
    4. Gera README.md para cada RODADA em data/coletas/
    5. NÃO apaga a estrutura antiga — você faz isso manualmente após validar

Após migrar, a estrutura antiga fica intacta. Só depois que você validar
que tudo está certo, apague manualmente as pastas antigas.
"""

import os
import shutil
import sys
import json
from pathlib import Path
from datetime import datetime

# ─── Configuração ────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
DATA = ROOT / "data"

DRY_RUN = "--execute" not in sys.argv

# Mapeamento: RODADA antiga → (nome novo, data, descrição)
RODADAS_COLETA = {
    "RODADA-1": {
        "nome": "RODADA-1_20260731",
        "data": "31/07/2026 e 03/08/2026",
        "descricao": "Multi-seno, swept-sine, chirp (piloto). Primeiras coletas exploratórias.",
        "uso_protocolo": "Não utilizada no protocolo v6",
    },
    "RODADA-2": {
        "nome": "RODADA-2_20260804",
        "data": "04/08/2026",
        "descricao": "Chirp, multi-seno, seq-degraus. Sessão completa de coleta.",
        "uso_protocolo": "Teste (avaliação final)",
    },
    "RODADA-3": {
        "nome": "RODADA-3_20260807",
        "data": "07/08/2026",
        "descricao": "Chirp, seq-degraus, multi-seno, curva semi-estática, seq-degraus-APRBS.",
        "uso_protocolo": "Validação (chirp-1_16-32) + Treino Exp E (chirps)",
    },
    "RODADA-4": {
        "nome": "RODADA-4_20260819",
        "data": "19/08/2026",
        "descricao": "APRBS, multi-seno, swept-sine, degraus, escada-atrito.",
        "uso_protocolo": "Teste (avaliação final)",
    },
    "RODADA-5": {
        "nome": "RODADA-5_20260827",
        "data": "27/08/2026",
        "descricao": "APRBS, multi-seno, seq-degraus, swept-sine. Sessão principal de treino.",
        "uso_protocolo": "Treino (Exp A–D)",
    },
    "RODADA-6": {
        "nome": "RODADA-6_20260904",
        "data": "04/09/2026",
        "descricao": "APRBS (45/60), chirp (45/60), multi-seno (45), degraus. Coleta com variação de amplitude.",
        "uso_protocolo": "Dados adicionais para protocolo com variação de ponto de operação",
    },
    "RODADA-7": {
        "nome": "RODADA-7_20260904",
        "data": "04-05/09/2026",
        "descricao": "Coleta completa: APRBS, chirp, multi-seno, seq-degraus (amplitude 45 e 60). Inclui MIX concatenados e dados de treino.",
        "uso_protocolo": "Treino/validação com dados amplos (protocolo NARX/NODE)",
    },
    "MALHA-ABERTA": {
        "nome": "MALHA-ABERTA_20260917",
        "data": "17/09/2026",
        "descricao": "Respostas em malha aberta para vários duty-cycles (-67% a +50%). Identificação estática e de atrito.",
        "uso_protocolo": "Identificação de curva estática e parâmetros de atrito",
    },
}

# CSVs soltos na raiz de coleta_dados que são templates/sinais de referência
CSVS_SOLTOS_COLETA = [
    "chirp-60-amp50_0908_23-09.csv", "chirp-60-amp50_0908_23-09.png",
    "chirp-60-amp50_0923_17-15.csv", "chirp-60-amp50_0923_17-15.png",
    "chirp_0601_19-15.csv", "chirp_0601_19-15.png",
    "chirp_0601_19-16.csv", "chirp_0601_19-16.png",
    "chirp_0601_19-23.csv", "chirp_0601_19-23.png",
    "chirp_0601_19-28.csv", "chirp_0601_19-28.png",
    "chirp_0601_19-39.csv", "chirp_0601_19-39.png",
    "chirp_0601_19-43.csv", "chirp_0601_19-43.png",
    "chirp_0608_18-44.csv", "chirp_0608_18-44.png",
    "degraus_0908_23-01.csv", "degraus_0908_23-01.png",
    "degraus_0925_17-26.csv", "degraus_0925_17-26.png",
    "multi-seno-60-030Hz_0908_23-13.csv", "multi-seno-60-030Hz_0908_23-13.png",
    "multisine_0601_19-37.csv", "multisine_0601_19-37.png",
    "multisine_0608_18-53.csv", "multisine_0608_18-53.png",
    "parametros_chirp.csv",
    "sinal3_swept_sine_malha_fechada_ref_y_0909_12-51.csv",
    "sinal3_swept_sine_malha_fechada_ref_y_0909_12-51.png",
    "step_35_open.csv", "step_39_open.csv",
    "sysid_multiseno_validation_ref_0909_13-16.csv",
    "sysid_multiseno_validation_ref_0909_13-16.png",
    "sysid_multiseno_validation_ref_0909_14-42.csv",
    "sysid_multiseno_validation_ref_0909_14-42.png",
    "sysid_multiseno_validation_ref_0920_19-11.csv",
    "sysid_multiseno_validation_ref_0920_19-11.png",
    "varios_steps.csv",
]

# CSVs soltos na raiz de controle/ que são sinais de referência
CSVS_SOLTOS_CONTROLE = [
    "adapt_signals.py",
    "aprbs-60-1.csv",
    "chirp-60-amp50.csv",
    "multi-seno-60-030Hz.csv",
    "referencia_mpc.csv",
    "simulacao_mpc.csv",
    "sinal1_semi_estatica_malha_fechada_ref_y.csv",
    "sinal3_swept_sine_malha_fechada_ref_y.csv",
    "sinal4_multiseno_malha_fechada.csv",
    "sysid_multiseno_validation_ref.csv",
    "wave_mpc_treino.csv",
]


# ─── Funções auxiliares ──────────────────────────────────────────────────────

def log_action(action: str, src: str = "", dst: str = ""):
    """Imprime ação de forma legível."""
    prefix = "[DRY-RUN]" if DRY_RUN else "[EXEC]"
    if src and dst:
        print(f"  {prefix} {action}: {src} → {dst}")
    elif src:
        print(f"  {prefix} {action}: {src}")
    else:
        print(f"  {prefix} {action}")


def mkdir(path: Path):
    """Cria diretório (se não dry-run)."""
    log_action("MKDIR", str(path))
    if not DRY_RUN:
        path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path):
    """Copia arquivo (se não dry-run)."""
    log_action("COPY", str(src), str(dst))
    if not DRY_RUN:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_dir(src: Path, dst: Path):
    """Copia diretório recursivamente (se não dry-run)."""
    log_action("COPY_DIR", str(src), str(dst))
    if not DRY_RUN:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def write_file(path: Path, content: str):
    """Escreve arquivo de texto (se não dry-run)."""
    log_action("WRITE", str(path))
    if not DRY_RUN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


# ─── Etapa 1: Criar data/coletas/ ───────────────────────────────────────────

def migrar_coletas():
    """Copia as RODADAs de coleta_dados para data/coletas/ com nomes padronizados."""
    print("\n" + "=" * 70)
    print("ETAPA 1: Migrar coletas para data/coletas/")
    print("=" * 70)

    src_base = DATA / "experimentos" / "coleta_dados"
    dst_base = DATA / "coletas"

    mkdir(dst_base)

    for rodada_old, info in RODADAS_COLETA.items():
        src = src_base / rodada_old
        dst = dst_base / info["nome"]

        if not src.exists():
            print(f"  [SKIP] {src} não existe")
            continue

        copy_dir(src, dst)

        # Gerar README para a rodada
        readme = f"""# {info['nome']}

| Campo | Valor |
|-------|-------|
| **Data da coleta** | {info['data']} |
| **Descrição** | {info['descricao']} |
| **Uso no protocolo** | {info['uso_protocolo']} |

## Arquivos

| Arquivo | Tamanho |
|---------|---------|
"""
        if src.exists():
            for f in sorted(src.iterdir()):
                if f.is_file():
                    size_kb = f.stat().st_size / 1024
                    readme += f"| `{f.name}` | {size_kb:.1f} KB |\n"

        readme += """
## Notas

> ⚠️ **Dados brutos — NÃO modificar.** Qualquer processamento deve ser feito
> nos scripts de pré-processamento, nunca nos arquivos originais.
"""
        write_file(dst / "README.md", readme)

    # Copiar CSVs soltos para uma subpasta "AVULSOS" dentro de coletas
    print("\n  Copiando CSVs soltos de coleta_dados/...")
    dst_avulsos = dst_base / "AVULSOS_pre-rodadas"
    mkdir(dst_avulsos)
    for fname in CSVS_SOLTOS_COLETA:
        src_file = src_base / fname
        if src_file.exists():
            copy_file(src_file, dst_avulsos / fname)
        else:
            print(f"  [SKIP] {src_file} não existe")


# ─── Etapa 2: Criar data/sinais_referencia/ ─────────────────────────────────

def migrar_sinais_referencia():
    """Move os sinais de referência/templates para data/sinais_referencia/."""
    print("\n" + "=" * 70)
    print("ETAPA 2: Criar data/sinais_referencia/")
    print("=" * 70)

    dst_base = DATA / "sinais_referencia"
    mkdir(dst_base)

    src_controle = DATA / "controle"
    for fname in CSVS_SOLTOS_CONTROLE:
        src = src_controle / fname
        if src.exists():
            copy_file(src, dst_base / fname)
        else:
            print(f"  [SKIP] {src} não existe")

    # README
    readme = """# Sinais de Referência

Sinais de entrada pré-definidos (templates) usados para coleta de dados
e referência do MPC. Estes são os **sinais que a interface envia para o
aeropêndulo**, não os dados coletados.

| Arquivo | Descrição |
|---------|-----------|
| `aprbs-60-1.csv` | Template APRBS, amplitude 60% |
| `chirp-60-amp50.csv` | Template Chirp, amplitude 50% |
| `multi-seno-60-030Hz.csv` | Template Multi-seno, fmax 0.30 Hz |
| `referencia_mpc.csv` | Referência para teste de MPC |
| `wave_mpc_treino.csv` | Onda de referência para treino MPC |
| `simulacao_mpc.csv` | Saída de simulação MPC (referência) |
| `sinal1_semi_estatica_malha_fechada_ref_y.csv` | Curva semi-estática, malha fechada |
| `sinal3_swept_sine_malha_fechada_ref_y.csv` | Swept-sine, malha fechada |
| `sysid_multiseno_validation_ref.csv` | Multi-seno para validação |
| `adapt_signals.py` | Script para adaptar/gerar sinais |
"""
    write_file(dst_base / "README.md", readme)


# ─── Etapa 3: Migrar testes_rede (SIL + simulação) ──────────────────────────

def migrar_testes_rede():
    """Copia os resultados de testes_rede para dentro do template de experimento."""
    print("\n" + "=" * 70)
    print("ETAPA 3: Migrar testes_rede existentes para EXP_LEGADO")
    print("=" * 70)

    src_sil = DATA / "experimentos" / "testes_rede" / "sil_stm32"
    src_sim = DATA / "experimentos" / "testes_rede" / "simulacao_python"

    # Criar um experimento legado para agrupar os resultados antigos
    exp_legado = DATA / "experimentos_novos" / "EXP000_legado_narx-mpc"
    mkdir(exp_legado)

    # README do legado
    readme = """# EXP000 — Legado NARX-MPC

| Campo           | Valor                                             |
|-----------------|---------------------------------------------------|
| **Data início** | ~2026-09-20                                       |
| **Modelo**      | NARX (ann_mpc.pth / narx_model.json)              |
| **Dados treino**| Vários (pré-reorganização)                        |
| **Status**      | 📦 Legado — migrado da estrutura antiga            |

## Notas

Este experimento agrupa os resultados da estrutura antiga (`testes_rede/`)
que não tinham rastreabilidade clara. Os dados foram copiados como estavam.

Para novos experimentos, use o script `novo_experimento.py`.
"""
    write_file(exp_legado / "README.md", readme)

    # Copiar SIL
    if src_sil.exists():
        dst_sil = exp_legado / "3_sil_stm32"
        copy_dir(src_sil, dst_sil)

    # Copiar simulação Python
    if src_sim.exists():
        dst_sim = exp_legado / "2_mpc_python"
        copy_dir(src_sim, dst_sim)


# ─── Etapa 4: Migrar controle/RODADA-N ──────────────────────────────────────

def migrar_controle():
    """Copia os resultados de controle/ (rodadas MPC no hardware)."""
    print("\n" + "=" * 70)
    print("ETAPA 4: Migrar data/controle/RODADA-N para EXP_LEGADO")
    print("=" * 70)

    src_ctrl = DATA / "controle"
    exp_legado = DATA / "experimentos_novos" / "EXP000_legado_narx-mpc"

    # As rodadas de controle vão como testes no aeropêndulo
    dst_real = exp_legado / "4_aeropendulo"
    mkdir(dst_real)

    for item in sorted(src_ctrl.iterdir()):
        if item.is_dir() and item.name.startswith("RODADA"):
            copy_dir(item, dst_real / item.name)


# ─── Etapa 5: Criar estrutura de template ───────────────────────────────────

def criar_template_experimento():
    """Cria um template EXP_TEMPLATE para referência."""
    print("\n" + "=" * 70)
    print("ETAPA 5: Criar template de novo experimento")
    print("=" * 70)

    template = DATA / "experimentos_novos" / "_TEMPLATE"

    for subdir in [
        "1_identificacao",
        "2_mpc_python",
        "3_sil_stm32",
        "4_aeropendulo",
    ]:
        mkdir(template / subdir)

    readme = """# EXP### — [NOME DO MODELO] [descrição curta]

| Campo           | Valor                                    |
|-----------------|------------------------------------------|
| **Data início** | YYYY-MM-DD                               |
| **Modelo**      | [Tipo] ([arquivo do script])             |
| **Dados treino**| [Quais RODADAs/arquivos]                 |
| **Dados teste** | [Quais RODADAs/arquivos]                 |
| **Status**      | 🔄 Em andamento / ✅ Completo até etapa N |

## Resultados Resumidos

| Plataforma     | RMSE (°) | R²     | Notas            |
|----------------|----------|--------|------------------|
| Python (ident) |          |        |                  |
| Python (MPC)   |          |        |                  |
| STM32 SIL      |          |        |                  |
| Aeropêndulo    |          |        |                  |

## Observações

- [Anotar decisões, problemas, insights]
"""
    write_file(template / "README.md", readme)

    config = {
        "experimento_id": "EXP###",
        "data_criacao": "YYYY-MM-DD",
        "modelo": {
            "tipo": "ex: PhysicsODE_Asymmetric",
            "script": "ex: sysid/scripts/node_v9.py",
            "versao": "ex: v9",
            "n_parametros": 0,
        },
        "treinamento": {
            "epocas": 2000,
            "lr_inicial": 0.015,
            "scheduler": "CosineAnnealing",
            "seed": 0,
            "batch_size": 1024,
        },
        "dados": {
            "treino": [],
            "validacao": [],
            "teste": [],
        },
        "coletas_usadas": [],
    }
    config_path = template / "config.json"
    log_action("WRITE", str(config_path))
    if not DRY_RUN:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if DRY_RUN:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  MODO DRY-RUN — Nenhuma alteração será feita no disco.     ║")
        print("║  Para executar de verdade, rode com --execute              ║")
        print("╚══════════════════════════════════════════════════════════════╝")
    else:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  MODO EXECUÇÃO — Arquivos serão COPIADOS (não movidos).    ║")
        print("║  A estrutura antiga permanece intacta.                     ║")
        print("╚══════════════════════════════════════════════════════════════╝")

    print(f"\n  Raiz do projeto: {ROOT}")
    print(f"  Pasta de dados:  {DATA}")

    migrar_coletas()
    migrar_sinais_referencia()
    migrar_testes_rede()
    migrar_controle()
    criar_template_experimento()

    print("\n" + "=" * 70)
    print("RESUMO DA NOVA ESTRUTURA")
    print("=" * 70)
    print("""
    data/
    ├── coletas/                          ← Dados brutos (imutáveis)
    │   ├── RODADA-1_20260731/
    │   ├── RODADA-2_20260804/
    │   ├── RODADA-3_20260807/
    │   ├── RODADA-4_20260819/
    │   ├── RODADA-5_20260827/
    │   ├── RODADA-6_20260904/
    │   ├── RODADA-7_20260904/
    │   ├── MALHA-ABERTA_20260917/
    │   └── AVULSOS_pre-rodadas/
    │
    ├── sinais_referencia/                ← Templates de sinais
    │
    ├── experimentos_novos/               ← Nova estrutura de experimentos
    │   ├── _TEMPLATE/                    ← Copie para criar novo EXP
    │   │   ├── README.md
    │   │   ├── config.json
    │   │   ├── 1_identificacao/
    │   │   ├── 2_mpc_python/
    │   │   ├── 3_sil_stm32/
    │   │   └── 4_aeropendulo/
    │   │
    │   └── EXP000_legado_narx-mpc/       ← Dados antigos migrados
    │       ├── README.md
    │       ├── 2_mpc_python/             ← (antigo simulacao_python/)
    │       ├── 3_sil_stm32/             ← (antigo sil_stm32/)
    │       └── 4_aeropendulo/           ← (antigo controle/RODADA-N/)
    │
    ├── experimentos/                     ← ANTIGA (manter até validar)
    └── controle/                         ← ANTIGA (manter até validar)
    """)

    if DRY_RUN:
        print("  ⚡ Para executar: python migrar_estrutura.py --execute")
    else:
        print("  ✅ Migração concluída!")
        print("  ℹ️  A estrutura antiga NÃO foi apagada.")
        print("     Valide a nova estrutura e apague manualmente quando estiver seguro.")
        print()
        print("  Próximo passo: use 'python novo_experimento.py' para criar novos EXPs.")


if __name__ == "__main__":
    main()
