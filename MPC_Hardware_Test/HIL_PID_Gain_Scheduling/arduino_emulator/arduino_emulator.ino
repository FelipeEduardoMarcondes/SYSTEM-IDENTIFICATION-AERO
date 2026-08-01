// ─────────────────────────────────────────────────────────────────────────────
// Arduino plant EMULATOR — modelo FÍSICO do 1/4-drone (hardcoded)
// ─────────────────────────────────────────────────────────────────────────────
// Substitui o antigo emulador NARX. Mantém EXATAMENTE o mesmo protocolo serial do
// arduino_real_slave, de modo que o mesmo script de PID no PC funciona com qualquer
// um dos dois (basta trocar a porta):
//
//   PC      -> Arduino:  "<u>\n"      u  = ação de controle em PORCENTAGEM
//   Arduino -> PC:       "<y>\n"      y  = ângulo do pêndulo em GRAUS
//
// Modelo contínuo identificado em phys_14_drone.ipynb (mesmos c1,c2,c3 usados no
// projeto do gain scheduling, pid_gain_scheduling_14_drone.ipynb):
//
//   theta_ddot = c1*sin(theta) + c2*u*|u| + c3*theta_dot        [theta em rad]
//
//   - c1*sin(theta) : torque gravitacional / inércia
//   - c2*u*|u|      : empuxo da hélice / inércia (força do motor ~ u^2, com sinal)
//   - c3*theta_dot  : atrito viscoso / inércia
//
// Integrado com UM passo RK4 de Ts = 0.05 s por amostra (mesmo integrador da
// identificação). A saída theta (rad) é convertida para graus antes de enviar.
//
// OBS p/ migração ao hardware real:
//   * O arduino_real_slave reporta o ângulo com offset de +90 graus e satura u em
//     [-100, 100] (pctParaUs). Aqui usamos a convenção do MODELO: theta = 0 é o
//     equilíbrio "para baixo", faixa de operação 0..180 graus, sem offset. Garanta
//     que a convenção de ângulo e a faixa de u batam antes do teste físico.
// ─────────────────────────────────────────────────────────────────────────────

const float C1 = -8.29151533f;
const float C2 =  0.00418887f;
const float C3 = -1.43673669f;

const float TS      = 0.05f;            // tempo de amostragem [s] (20 Hz)
const float RAD2DEG = 57.2957795131f;

// ── Estado do modelo ──────────────────────────────────────────────────────────
float theta    = 0.0f;   // rad
float thetaDot = 0.0f;   // rad/s

// ── Buffer de recepção serial ─────────────────────────────────────────────────
String rxBuffer = "";

// theta_ddot = f(theta, theta_dot, u)
float accel(float th, float thd, float u) {
  return C1 * sinf(th) + C2 * u * fabsf(u) + C3 * thd;
}

// Um passo RK4 de tamanho fixo TS com entrada u em ZOH (segura-e-mantém)
void stepPlant(float u) {
  float k1th = thetaDot;
  float k1td = accel(theta, thetaDot, u);

  float k2th = thetaDot + 0.5f * TS * k1td;
  float k2td = accel(theta + 0.5f * TS * k1th, thetaDot + 0.5f * TS * k1td, u);

  float k3th = thetaDot + 0.5f * TS * k2td;
  float k3td = accel(theta + 0.5f * TS * k2th, thetaDot + 0.5f * TS * k2td, u);

  float k4th = thetaDot + TS * k3td;
  float k4td = accel(theta + TS * k3th, thetaDot + TS * k3td, u);

  theta    += (TS / 6.0f) * (k1th + 2.0f * k2th + 2.0f * k3th + k4th);
  thetaDot += (TS / 6.0f) * (k1td + 2.0f * k2td + 2.0f * k3td + k4td);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // aguarda a porta serial conectar
  }
}

void loop() {
  // Leitura não-bloqueante: a cada linha "<u>\n" recebida, dá um passo no modelo
  // e responde com o ângulo em graus — espelhando o arduino_real_slave.
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      rxBuffer.trim();
      if (rxBuffer.length() > 0) {
        float u = rxBuffer.toFloat();
        stepPlant(u);
        Serial.println(theta * RAD2DEG, 2);   // resposta: ângulo em graus
      }
      rxBuffer = "";
    } else {
      rxBuffer += c;
    }
  }
}
