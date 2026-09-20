# HIL — PID Gain Scheduling (1/4-drone)

Controlador PID com **gain scheduling** (projetado em `pid_gain_scheduling_14_drone.ipynb`)
rodando em malha fechada com o Arduino via serial. Os parâmetros do projeto e o
modelo físico (`phys_14_drone.ipynb`) estão **hardcoded**.

## Arquivos (pacote autocontido)

- `hil_pid_gain_scheduling.py` — o controlador. Roda contra o emulador (em software ou no Arduino) ou contra a bancada real.
- `arduino_emulator/arduino_emulator.ino` — emulador do *plant*: roda o **modelo físico** dentro de um Arduino (sem IMU/ESC).
- `arduino_real_slave/arduino_real_slave.ino` — escravo da bancada física (IMU + ESC).
- `resultado_esperado_emulador.png` — gráfico de referência do que o emulador deve produzir.

## Modelo e projeto (hardcoded)

Modelo: `theta_ddot = c1·sin(theta) + c2·u·|u| + c3·theta_dot`, com
`c1=-8.29151533`, `c2=0.00418887`, `c3=-1.43673669`, `Ts=0.05 s`.

PID por *pole placement* (polos `-2, -2.5, -3`): `Kd` e `Ki` constantes; `Kp`
agendado pela saída e feedforward `v_eq` agendado pela referência. A entrada física
sai por `u = sign(v)·sqrt(|v|)`.

Referência: escada **0 → 180 graus, de 10 em 10**, 5 s por degrau (`SWEEP_DOWN=True`
para subir e descer).

# ▶ ORDEM DE TESTE — SEMPRE EMULADOR PRIMEIRO, EQUIPAMENTO DEPOIS

> O **mesmo** `hil_pid_gain_scheduling.py` roda em todas as etapas. Só muda de onde
> vêm os sinais. Não pule para a bancada sem antes passar pela FASE 1.

Pré-requisito: `pip install numpy matplotlib pyserial`

---

## FASE 1 — EMULADOR (sem risco, faça PRIMEIRO)

### 1A. Emulador em software (não precisa de Arduino)

No topo de `hil_pid_gain_scheduling.py` deixe:

```python
USE_EMULATOR = True
```

Rode:

```bash
python3 hil_pid_gain_scheduling.py
```

A serial é substituída pelo `FakeSerialArduino`, que roda o **mesmo modelo físico**
do `arduino_emulator.ino`. Gera `hil_gain_scheduling_EMU.csv` e `.png`.

✅ **Critério de aprovação:** o gráfico gerado deve bater com
`resultado_esperado_emulador.png` (saída segue a escada 0→180 sem instabilizar).

> `REALTIME=False` (padrão) roda o mais rápido possível; o resultado numérico é
> idêntico ao tempo real. Coloque `True` para sentir a duração real (~95 s).

### 1B. Emulador no Arduino (valida o link serial real, ainda sem motor)

1. Flash de `arduino_emulator/arduino_emulator.ino` no Arduino.
2. No `.py`: `USE_EMULATOR = False` e ajuste `SERIAL_PORT` (ex.: `COM8`, `/dev/ttyUSB0`).
3. `python3 hil_pid_gain_scheduling.py`.

✅ **Critério de aprovação:** resultado equivalente ao da etapa 1A (o Arduino está
rodando o mesmo modelo físico, agora pela serial de verdade). Gera `..._HW.csv`/`.png`.

---

## FASE 2 — EQUIPAMENTO (bancada real, só depois da FASE 1 OK)

⚠️ **Segurança:** bancada fixada, hélice livre/desobstruída, ninguém na frente do
plano de rotação, mão pronta no botão de desligar. O script pede ENTER antes de
ligar o motor e envia `u=0` no fim.

1. Flash de `arduino_real_slave/arduino_real_slave.ino` no Arduino da bancada.
2. No `.py`: `USE_EMULATOR = False` e `SERIAL_PORT` correto.
3. `python3 hil_pid_gain_scheduling.py` — aguarda 6 s a calibração do IMU e pede
   ENTER para ligar o motor.
4. Acompanhe; ao fim gera `hil_gain_scheduling_HW.csv` e `.png`.

✅ **Critério de aprovação:** a saída acompanha a referência em escada de forma
estável. Guardem o `.csv`/`.png` para compararmos.

## ⚠️ Pontos de atenção antes do hardware físico

- **Faixa de `u`**: projeto e firmware já casados em `±100 %`. No `arduino_real_slave`,
  `pctParaUs` mapeia `±100 %` para `1000..2000 us` (faixa cheia do ESC).
- **Convenção de ângulo**: o `arduino_real_slave` reporta `ângulo + 90°`; o modelo
  usa `theta=0` no equilíbrio "para baixo" (faixa 0..180). Garanta que ambos meçam
  0 na mesma posição física.
