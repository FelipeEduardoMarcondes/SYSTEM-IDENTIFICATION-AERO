#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// --- Configuração LoRa ---
// Pinos padrão para Heltec WiFi LoRa 32 V2. 
// Ajuste se for TTGO (geralmente SS=18, RST=14, DI0=26) ou Heltec V3.
#define SS      18
#define RST     14
#define DI0     26
#define BAND    915E6 // Frequência Brasil

// --- Configuração OLED ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RST 16
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RST);

int pacotesRecebidos = 0;
int lastRssi = 0;

void setup() {
  Serial.begin(500000);
  while (!Serial);

  // Inicializa I2C para o OLED (Pinos 4 e 15 para Heltec V2. Para TTGO use Wire.begin(21, 22))
  Wire.begin(4, 15);
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { 
    Serial.println(F("OLED falhou"));
  }
  display.clearDisplay();
  display.setTextColor(WHITE);
  display.setTextSize(1);
  display.setCursor(0,0);
  display.println("LoRa Gateway RX");
  display.display();

  // Inicializa LoRa
  LoRa.setPins(SS, RST, DI0);
  if (!LoRa.begin(BAND)) {
    Serial.println("LoRa falhou!");
    display.println("Erro no LoRa!");
    display.display();
    while (1);
  }
  
  // Opcional: Para maior alcance (e menor velocidade), descomente:
  // LoRa.setSpreadingFactor(10); 
  // LoRa.setSignalBandwidth(125E3);

  display.println("Iniciado! Aguardando");
  display.display();
  Serial.println("# LORA_RX_READY");
}

void loop() {
  // 1. Ler da Serial (Comandos do PC/Python) e enviar para o Dummy via LoRa
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() > 0) {
      LoRa.beginPacket();
      LoRa.print(cmd);
      LoRa.endPacket();
      
      // Atualiza o display com o comando enviado
      display.fillRect(0, 32, 128, 32, BLACK);
      display.setCursor(0, 32);
      display.print("-> PC: ");
      display.println(cmd);
      display.display();
    }
  }

  // 2. Ler do LoRa (Dados CSV do Dummy) e enviar para o PC/Python
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    String rxData = "";
    while (LoRa.available()) {
      rxData += (char)LoRa.read();
    }
    
    lastRssi = LoRa.packetRssi();
    pacotesRecebidos++;

    // Envia o dado EXATAMENTE como chegou para o Python entender
    Serial.println(rxData);

    // Atualiza o OLED (limitado para não causar muito delay)
    if (pacotesRecebidos % 5 == 0 || rxData.startsWith("#")) {
      display.fillRect(0, 16, 128, 48, BLACK);
      display.setCursor(0, 16);
      display.print("RSSI: "); display.print(lastRssi); display.println(" dBm");
      display.print("Pkts RX: "); display.println(pacotesRecebidos);
      display.print("Dado: "); display.println(rxData.substring(0, min(15, (int)rxData.length())));
      display.display();
    }
  }
}
