// Scanner de I2C -- roda uma vez, lista quem responde em cada barramento
// disponivel. Usa pra descobrir o endereço/bus real da IMU onboard do
// XIAO nRF52840 Sense quando 0x6A no Wire padrao da NACK.

#include <Wire.h>

void escanear(TwoWire &bus, const char *nomeBus) {
  Serial.print("--- Escaneando ");
  Serial.print(nomeBus);
  Serial.println(" ---");
  int encontrados = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    bus.beginTransmission(addr);
    uint8_t erro = bus.endTransmission();
    if (erro == 0) {
      Serial.print("  dispositivo em 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
      encontrados++;
    }
  }
  if (encontrados == 0) Serial.println("  nenhum dispositivo respondeu.");
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 5000) delay(10);

  // Liga a IMU onboard antes de escanear -- sem isso ela nao responde
  // em endereco nenhum.
  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);
  digitalWrite(PIN_LSM6DS3TR_C_POWER, HIGH);
  delay(100);

  Serial.println("IMU alimentada. Escaneando barramentos I2C...\n");

  Wire.begin();
  escanear(Wire, "Wire (pinos externos D4/D5)");

#if WIRE_INTERFACES_COUNT > 1
  Wire1.begin();
  escanear(Wire1, "Wire1 (bus interno)");
#else
  Serial.println("Este core nao expoe um Wire1 separado.");
#endif

  Serial.println("Fim do scan.");
}

void loop() {}
