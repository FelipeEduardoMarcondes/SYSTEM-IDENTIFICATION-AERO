# Sinais de Referência

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
