# Aeropêndulo — Sistema de Aquisição v2

TCC: NMPC Aproximado por Redes Neurais em Hardware Embarcado  
Firmware: Arduino (ICM-20948 + ESC BLDC)  
Python: aquisição, geração de sinais, visualização

---

## Estrutura do repositório

```
aeropendulo/
├── aeropendulo_firmware.ino     Firmware Arduino
└── python/
    ├── main.py          Ponto de entrada — menu interativo
    ├── config.py        Constantes globais (BAUD, FS, limites)
    ├── signals.py       Geradores de sinais (chirp, multisine, degraus)
    ├── serial_comm.py   Comunicação serial + protocolos SEQ e WAVE
    ├── acquisition.py   Loop de coleta, threads de parada e streaming
    ├── live_plot.py     Visualizador em tempo real
    ├── storage.py       Salvar/carregar CSV, plot estático
    └── presets.py       Sequências de degraus predefinidas
```

---

## Instalação

```bash
pip install pandas matplotlib numpy pyserial keyboard
```

---

## Uso rápido

```bash
# Menu interativo
python python/main.py

# Plotar CSV direto
python python/main.py dados/chirp_20260530_143000.csv
```

---

## Protocolos firmware ↔ Python

### Protocolo SEQ — degraus discretos

Usado para sequências de degraus com timing interno no Arduino.  
Python envia o comando antes do START; durante a coleta o Arduino avança
os degraus sozinho pelo tempo interno, sem depender de UART para timing.

```
PC  → SEQ=0:45,15000:30,30000:60\n
ARD → # SEQ_OK n=3
PC  → START\n
ARD → # EXP_START
ARD → tempo_ms,angulo_deg,u_pct,referencia
ARD → 0,44.8,12.3,45.00
    ...
PC  → STOP\n
ARD → # PARADO
```

### Protocolo WAVE — sinais contínuos (chirp, multi-seno)

O sinal completo é pré-carregado no buffer interno do Arduino **antes** do
START. Durante a aquisição, o firmware lê a próxima amostra a cada
interrupção de timer (100 Hz determinístico) — sem nenhuma dependência
de UART no loop de controle.

```
PC  → WAVE=4200\n               (total de amostras)
ARD → # WAVE_OK n=4200
PC  → DATA=45.00,45.12,...\n    (até 150 valores por pacote)
ARD → # DATA_ACK 0
PC  → DATA=...\n
ARD → # DATA_ACK 1
    ...  (todos os blocos)
PC  → DATA_END\n
ARD → # WAVE_READY n=4200
PC  → START\n
ARD → # EXP_START
ARD → tempo_ms,angulo_deg,u_pct,referencia
    ...
PC  → STOP\n
ARD → # PARADO
```

**Por que WAVE resolve o problema do chirp?**  
O chirp muda de valor a cada 10 ms. Enviar via UART com `time.sleep` causa
jitter de ±5–20 ms por limitações do sistema operacional, distorcendo o
sinal. Com WAVE, o Arduino executa `waveBuf[waveIdx++]` dentro da ISR do
timer — timing determinístico de hardware.

---

## Geração dos sinais (signals.py)

```python
from signals import chirp, multisine, degraus_aleatorios, exportar_csv

# Chirp: 0.025 Hz → 0.5 Hz, 40 s, amplitude ±30 deg em torno de 45 deg
t, u = chirp(duracao=40, amp=30, dc=45, f_ini=0.025, f_fim=0.5)

# Multi-seno persistentemente excitante
t, u = multisine(duracao=40, amp=30, dc=45, f_max=0.5, seed=0)

# Sequência de degraus aleatórios (para validação)
t, u = degraus_aleatorios(duracao=60, amp=30, dc=45, t_degrau=4, seed=0)

# Exportar para CSV (compatível com MATLAB e com carregar_sequencia_csv)
exportar_csv(t, u, "sinal_chirp.csv")
```

---

## Limites do buffer WAVE

O Arduino Mega tem ~8 KB de SRAM. Com `float` (4 bytes) e `WAVE_MAX=6100`:

```
6100 × 4 bytes = 24.4 KB   →  ESTOURO no Mega
```

**Solução para Mega:** reduza `WAVE_MAX` para `1500` (~15 s) e divida
experimentos longos em blocos, ou use o Arduino Due / STM32 que têm
mais SRAM.

Para o STM32F446RE (128 KB SRAM):
```
6100 × 4 bytes = 24.4 KB   →  OK (sobram ~100 KB para o restante)
```

Ajuste `WAVE_MAX` no firmware e `WAVE_CHUNK` em `config.py` conforme
o microcontrolador alvo.

---

## Parâmetros importantes (config.py)

| Parâmetro    | Padrão  | Descrição                                      |
|--------------|---------|------------------------------------------------|
| `BAUD`       | 500 000 | Taxa serial — deve bater com o firmware        |
| `FS`         | 100.0   | Frequência de amostragem (Hz)                  |
| `WAVE_CHUNK` | 150     | Amostras por pacote DATA=                      |
| `LIVE_JANELA_S` | 30   | Janela deslizante no live plot (s)             |
| `REF_MIN/MAX`| 0/135   | Limites de referência angular (graus)          |

---

## Fluxo completo do TCC

```
Etapa 1-2: Degraus (modo SEQ)  →  PID preliminar, validação rápida
Etapa 3:   Chirp   (modo WAVE) →  caracterização em frequência
Etapa 3:   Multi-seno (WAVE)   →  identificação (excitação persistente)
Etapa 3:   Degraus aleatórios  →  validação do modelo identificado
Etapa 4+:  Degraus (SEQ/WAVE)  →  ensaios PID final e MLP embarcada
```
