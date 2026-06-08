/*
  aeropendulo_firmware.ino
  ========================
  Firmware do aeropêndulo — compatível com o sistema de aquisição Python v2.
  Protocolos suportados
  ---------------------
  SEQ=t1_ms:ref1,t2_ms:ref2,...    Sequência de degraus (timing interno)
  WAVE=<n>                          Anuncia n amostras de sinal contínuo
  DATA=v0,v1,...,vk                 Bloco de amostras (até WAVE_CHUNK por pacote)
  DATA_END                          Encerra o carregamento do buffer WAVE
  CHIRP=Amp,Fmax,T0,DC,PadS         Gera chirp dinamicamente (Amp, Fmax em Hz, T0 em s, offset DC, pad em s)
  R=<graus>                         Referência instantânea (legado / modo SEQ)
  START                             Inicia experimento
  STOP                              Para imediatamente
  RECAL                             Recalibra giroscópio
  FREE                              Coleta com motor desligado
*/

#include <Wire.h>
#include <Servo.h>
#include "ICM_20948.h"

// ── Pinos e limites do ESC ────────────────────────────────────────────────────
#define WIRE_PORT     Wire
#define AD0_VAL       0
#define ESC_PIN       4
#define ESC_MIN       1000
#define ESC_MAX       2000
#define ESC_NEUTRO    1500

// ── Temporização ──────────────────────────────────────────────────────────────
#define ODR_DIV       10
#define INTERVALO_US  10000UL        // 10 ms = 100 Hz

// ── Capacidade dos buffers ────────────────────────────────────────────────────
#define MAX_STEPS     50             // degraus no modo SEQ
#define WAVE_MAX      1500           // amostras WAVE (~15 s a 100 Hz)

// ── Ganhos PID ────────────────────────────────────────────────────────────────
const float Kp = 0.5793f;
const float Ki = 0.6647f;
const float Kd = 0.2f;
const float Ts = INTERVALO_US / 1e6f;

// ── IMU e ESC ─────────────────────────────────────────────────────────────────
ICM_20948_I2C myICM;
Servo esc;
// ── Filtro complementar ───────────────────────────────────────────────────────
float anguloFiltrado = 0.0f;
float gyroBias       = 0.0f;
// ── Estado do controlador PID ─────────────────────────────────────────────────
float e_1 = 0.0f;
float u_i = 0.0f;
float y_1 = 0.0f;
// ── Referência corrente ───────────────────────────────────────────────────────
float r = 40.0f;

// ── Buffer SEQ (degraus) ──────────────────────────────────────────────────────
struct Degrau {
  unsigned long t_ms;
  float ref;
};
Degrau degraus[MAX_STEPS];
int    nDegraus  = 0;
int    idxDegrau = 0;
// ── Buffer WAVE (sinal contínuo) ──────────────────────────────────────────────
float        waveBuf[WAVE_MAX];
int          waveLen       = 0;
int          waveTotalEsp  = 0;
int          waveIdx       = 0;
int          waveBloco     = 0;
bool         waveAtivo     = false;

// ── Parâmetros do Chirp ───────────────────────────────────────────────────────
bool  chirpAtivo  = false;
float chirp_Amp   = 0.0f;
float chirp_T0    = 0.0f;
float chirp_DC    = 0.0f;
float chirp_PadS  = 0.0f;   // s — DC antes e depois do chirp ativo
float chirp_a     = 0.0f;
float chirp_b     = 0.0f;

// ── Temporização ──────────────────────────────────────────────────────────────
unsigned long proximaAmostra = 0;
unsigned long ultimoDadoUs   = 0;
unsigned long tempoInicio    = 0;

// ── Estados da máquina ────────────────────────────────────────────────────────
enum Estado { IDLE, LOADING_WAVE, RUNNING, FREE_RUN };
Estado estado = IDLE;

// ── Buffer de recepção serial ─────────────────────────────────────────────────
String rxBuffer = "";
// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

int pctParaUs(float pct) {
  pct = constrain(pct, -80.0f, 80.0f);
  return (int)(1500.0f - 5.0f * pct);
}

float anguloAcelAtual() {
  unsigned long limite = millis() + 200;
  while (millis() < limite) {
    if (myICM.dataReady()) {
      myICM.getAGMT();
      return atan2f(-myICM.accZ(), myICM.accX()) * 180.0f / PI;
    }
  }
  return 0.0f;
}

void configurarIMU() {
  ICM_20948_smplrt_t smplrt;
  smplrt.g = ODR_DIV;
  smplrt.a = ODR_DIV;
  myICM.setSampleRate(ICM_20948_Internal_Gyr, smplrt);
  myICM.setSampleRate(ICM_20948_Internal_Acc, smplrt);

  ICM_20948_dlpcfg_t dlp;
  dlp.g = 4;
  dlp.a = 4;
  myICM.setDLPFcfg(ICM_20948_Internal_Gyr | ICM_20948_Internal_Acc, dlp);
  myICM.enableDLPF(ICM_20948_Internal_Gyr, true);
  myICM.enableDLPF(ICM_20948_Internal_Acc, true);
}

bool reiniciarIMU() {
  myICM.begin(WIRE_PORT, AD0_VAL);
  if (myICM.status != ICM_20948_Stat_Ok) return false;
  configurarIMU();
  return true;
}

void calibrarGiroscopio() {
  Serial.println("# CAL_START");
  delay(1000);
  float soma = 0.0f;
  for (int i = 0; i < 500;) {
    if (myICM.dataReady()) {
      myICM.getAGMT();
      soma += myICM.gyrY();
      i++;
      delay(2);
    }
  }
  gyroBias = soma / 500.0f;
  Serial.print("# CAL_BIAS ");
  Serial.println(gyroBias, 4);
  Serial.println("# CAL_OK");
}

void inicializarESC() {
  esc.attach(ESC_PIN, ESC_MIN, ESC_MAX);
  esc.writeMicroseconds(ESC_NEUTRO);
  delay(3000);
  Serial.println("# ESC_OK");
}

void resetarControlador() {
  e_1 = 0.0f;
  u_i = 0.0f;
  y_1 = 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsers de comandos
// ─────────────────────────────────────────────────────────────────────────────

void parsearSEQ(const String& cmd) {
  nDegraus = 0;
  String dados = cmd.substring(4);
  int pos = 0;

  while (pos < (int)dados.length() && nDegraus < MAX_STEPS) {
    int sep   = dados.indexOf(',', pos);
    String parte = (sep == -1) ? dados.substring(pos) : dados.substring(pos, sep);
    int col = parte.indexOf(':');
    if (col != -1) {
      degraus[nDegraus].t_ms = (unsigned long)parte.substring(0, col).toInt();
      degraus[nDegraus].ref  = parte.substring(col + 1).toFloat();
      nDegraus++;
    }
    if (sep == -1) break;
    pos = sep + 1;
  }

  Serial.print("# SEQ_OK n=");
  Serial.println(nDegraus);
  if (nDegraus > 0) r = degraus[0].ref;
}

void iniciarWAVE(const String& cmd) {
  waveTotalEsp = cmd.substring(5).toInt();
  waveLen      = 0;
  waveBloco    = 0;

  if (waveTotalEsp <= 0 || waveTotalEsp > WAVE_MAX) {
    Serial.print("# WAVE_ERR total=");
    Serial.print(waveTotalEsp);
    Serial.print(" limite=");
    Serial.println(WAVE_MAX);
    return;
  }

  Serial.print("# WAVE_OK n=");
  Serial.println(waveTotalEsp);
  estado = LOADING_WAVE;
}

void processarDATA(const String& cmd) {
  String dados = cmd.substring(5);
  int pos = 0;
  while (pos < (int)dados.length() && waveLen < WAVE_MAX) {
    int sep = dados.indexOf(',', pos);
    String tok = (sep == -1) ? dados.substring(pos) : dados.substring(pos, sep);
    tok.trim();
    if (tok.length() > 0) {
      waveBuf[waveLen++] = tok.toFloat();
    }
    if (sep == -1) break;
    pos = sep + 1;
  }

  Serial.print("# DATA_ACK ");
  Serial.println(waveBloco++);
}

void finalizarWAVE() {
  Serial.print("# WAVE_READY n=");
  Serial.println(waveLen);
  waveAtivo = true;
  estado    = IDLE;
}

void parsearCHIRP(const String& cmd) {
  // Payload: CHIRP=amp,fmax,t0,dc,pad_s
  String dados = cmd.substring(6);

  int p1 = dados.indexOf(',');
  int p2 = dados.indexOf(',', p1 + 1);
  int p3 = dados.indexOf(',', p2 + 1);
  int p4 = dados.indexOf(',', p3 + 1);   // opcional — pad_s

  if (p1 == -1 || p2 == -1 || p3 == -1) {
    Serial.println("# CHIRP_ERR");
    return;
  }

  chirp_Amp     = dados.substring(0, p1).toFloat();
  float f_max   = dados.substring(p1 + 1, p2).toFloat();
  chirp_T0      = dados.substring(p2 + 1, p3).toFloat();
  chirp_DC      = (p4 == -1) ? dados.substring(p3 + 1).toFloat()
                              : dados.substring(p3 + 1, p4).toFloat();
  chirp_PadS    = (p4 == -1) ? 0.0f : dados.substring(p4 + 1).toFloat();

  float f0 = 1.0f / chirp_T0;
  float k1 = 1.0f;
  float k2 = f_max / f0;

  chirp_a = PI * (k2 - k1) * (f0 * f0);
  chirp_b = 2.0f * PI * k1 * f0;

  chirpAtivo = true;
  waveAtivo  = false;
  nDegraus   = 0;

  Serial.println("# CHIRP_OK");
}

// ─────────────────────────────────────────────────────────────────────────────
// Leitura não-bloqueante da serial
// ─────────────────────────────────────────────────────────────────────────────

String lerComandoSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      String cmd = rxBuffer;
      rxBuffer = "";
      cmd.trim();
      return cmd;
    }
    rxBuffer += c;
  }
  return "";
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(500000);
  while (!Serial);

  WIRE_PORT.begin();
  WIRE_PORT.setClock(400000);
  WIRE_PORT.setWireTimeout(3000, true);

  myICM.begin(WIRE_PORT, AD0_VAL);
  if (myICM.status != ICM_20948_Stat_Ok) {
    Serial.println("# ERRO_IMU");
    while (1);
  }

  configurarIMU();
  inicializarESC();
  calibrarGiroscopio();

  Serial.println("# PRONTO");
  Serial.println("# Comandos: START | STOP | RECAL | FREE");
  Serial.println("# SEQ=t1:r1,...  |  R=<graus>");
  Serial.println("# WAVE=<n> | DATA=v0,... | DATA_END");
  Serial.println("# CHIRP=Amp,Fmax,T0,DC,PadS");
  estado = IDLE;
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop principal
// ─────────────────────────────────────────────────────────────────────────────

void loop() {

  String cmd = lerComandoSerial();
  if (cmd.length() > 0) {

    if (cmd == "STOP") {
      esc.writeMicroseconds(ESC_NEUTRO);
      resetarControlador();
      waveAtivo  = false;
      chirpAtivo = false;
      waveIdx    = 0;
      nDegraus   = 0;
      estado     = IDLE;
      Serial.println("# PARADO");
      return;
    }

    if (cmd == "RECAL" && estado == IDLE) {
      calibrarGiroscopio();
      Serial.println("# PRONTO");
      return;
    }

    if (cmd.startsWith("R=")) {
      r = cmd.substring(2).toFloat();
      return;
    }

    if (cmd.startsWith("SEQ=") && estado == IDLE) {
      waveAtivo  = false;
      chirpAtivo = false;
      parsearSEQ(cmd);
      return;
    }

    if (cmd.startsWith("WAVE=") && estado == IDLE) {
      nDegraus   = 0;
      chirpAtivo = false;
      iniciarWAVE(cmd);
      return;
    }

    if (cmd.startsWith("DATA=") && estado == LOADING_WAVE) {
      processarDATA(cmd);
      return;
    }

    if (cmd == "DATA_END" && estado == LOADING_WAVE) {
      finalizarWAVE();
      return;
    }

    if (cmd.startsWith("CHIRP=") && estado == IDLE) {
      parsearCHIRP(cmd);
      return;
    }

    if (cmd == "START" && estado == IDLE) {
      anguloFiltrado = anguloAcelAtual();
      resetarControlador();
      tempoInicio    = millis();
      proximaAmostra = micros() + INTERVALO_US;
      ultimoDadoUs   = micros();
      idxDegrau      = 0;
      waveIdx        = 0;

      if (chirpAtivo) {
        r = chirp_DC; // Inicia na origem (offset)
      } else if (waveAtivo) {
        r = waveBuf[0];
      } else if (nDegraus > 0) {
        r = degraus[0].ref;
      }

      estado = RUNNING;
      Serial.println("# EXP_START");
      Serial.println("tempo_ms,angulo_deg,u_pct,referencia");
      return;
    }

    if (cmd == "FREE" && estado == IDLE) {
      anguloFiltrado = anguloAcelAtual();
      resetarControlador();
      tempoInicio    = millis();
      proximaAmostra = micros() + INTERVALO_US;
      ultimoDadoUs   = micros();

      estado = FREE_RUN;
      Serial.println("# EXP_START");
      Serial.println("tempo_ms,angulo_deg,u_pct,referencia");
      return;
    }
  }

  if (estado == LOADING_WAVE) return;

  if (estado == FREE_RUN) {
    unsigned long agoraMicros = micros();
    if (agoraMicros < proximaAmostra) return;
    if (!myICM.dataReady()) {
      if (micros() - ultimoDadoUs > 50000UL) {
        reiniciarIMU();
        ultimoDadoUs = micros();
      }
      return;
    }

    ultimoDadoUs   = micros();
    proximaAmostra = agoraMicros + INTERVALO_US;
    myICM.getAGMT();

    const float dt   = Ts;
    float anguloAcel = atan2f(-myICM.accZ(), myICM.accX()) * 180.0f / PI;
    float gy         = myICM.gyrY() - gyroBias;
    anguloFiltrado   = 0.98f * (anguloFiltrado + (-gy) * dt) + 0.02f * anguloAcel;

    Serial.print(millis() - tempoInicio);
    Serial.print(",");
    Serial.print(anguloFiltrado + 90.0f, 2);
    Serial.print(",");
    Serial.print("0.00,");
    Serial.println("0.00");
    return;
  }

  if (estado == IDLE) return;

  unsigned long agoraMicros = micros();
  if (agoraMicros < proximaAmostra) return;
  
  if (!myICM.dataReady()) {
    if (micros() - ultimoDadoUs > 50000UL) {
      Serial.println("# IMU_REINIT");
      reiniciarIMU();
      ultimoDadoUs = micros();
    }
    return;
  }

  ultimoDadoUs   = micros();
  proximaAmostra = agoraMicros + INTERVALO_US;
  myICM.getAGMT();

  unsigned long t_exp = millis() - tempoInicio;

  // ── Avanço de referência ─────────────────────────────────────────────────
  if (chirpAtivo) {
    float t_sec   = t_exp / 1000.0f;
    float t_fim   = chirp_PadS + chirp_T0;
    if (t_sec < chirp_PadS) {
      // Fase 1: pad inicial — sistema estabiliza no ponto de operação
      r = chirp_DC;
    } else if (t_sec < t_fim) {
      // Fase 2: chirp ativo — fase calculada a partir do início da excitação
      float t_local = t_sec - chirp_PadS;
      r = chirp_Amp * sinf((chirp_a * t_local + chirp_b) * t_local) + chirp_DC;
    } else {
      // Fase 3: pad final — retorno ao ponto de operação
      r = chirp_DC;
    }
  } else if (waveAtivo) {
    if (waveIdx < waveLen) {
      r = waveBuf[waveIdx++];
    }
  } else if (nDegraus > 0) {
    while (idxDegrau + 1 < nDegraus &&
           t_exp >= degraus[idxDegrau + 1].t_ms) {
      idxDegrau++;
      r = degraus[idxDegrau].ref;
      Serial.print("# STEP_CHANGE idx=");
      Serial.print(idxDegrau);
      Serial.print(" t=");
      Serial.print(t_exp);
      Serial.print("ms ref=");
      Serial.println(r, 2);
    }
  }

  // ── Filtro complementar ──────────────────────────────────────────────────
  const float dt   = Ts;
  float anguloAcel = atan2f(-myICM.accZ(), myICM.accX()) * 180.0f / PI;
  float gy         = myICM.gyrY() - gyroBias;
  anguloFiltrado   = 0.98f * (anguloFiltrado + (-gy) * dt) + 0.02f * anguloAcel;
  
  // ── Lei de controle PID ──────────────────────────────────────────────────
  float y_med = anguloFiltrado + 90.0f;
  float e     = r - y_med;

  float u_p = Kp * e;
  u_i       = u_i + Ki * (Ts / 2.0f) * (e + e_1);
  float u_d = -(Kd / Ts) * (y_med - y_1);
  float u   = u_p + u_i + u_d;
  
  // Anti-windup
  float u_sat = constrain(u, -80.0f, 80.0f);
  if (u != u_sat) {
    u_i -= (u - u_sat);
  }
  u = u_sat;

  esc.writeMicroseconds(pctParaUs(u));
  e_1 = e;
  y_1 = y_med;

  // ── Saída CSV ─────────────────────────────────────────────────────────────
  Serial.print(t_exp);
  Serial.print(",");
  Serial.print(y_med, 2);
  Serial.print(",");
  Serial.print(u, 2);
  Serial.print(",");
  Serial.println(r, 2);
}