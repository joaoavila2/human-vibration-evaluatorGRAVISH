// Firmware de STREAMING ao vivo -- nao grava no SD, so manda amostra por
// amostra pela Serial, pro app desenhar o FFT em tempo real. Separado do
// firmware de gravacao (VCI_nRF52840_accel_SD_v2.ino) de proposito: nao
// arrisca o que ja esta testado e funcionando la.
//
// Formato de cada amostra (14 bytes, little-endian, sem padding):
//   uint16  sync   sempre 0x55AA -- deixa o lado PC resincronizar se
//                  conectar no meio do stream ou perder byte
//   float   x, y, z   aceleracao em g

#include <LSM6DS3.h>
#include <Wire.h>

LSM6DS3 myIMU(I2C_MODE, 0x6A);

const uint16_t ACCEL_RANGE_G = 8;
const uint16_t SAMPLE_RATE_HZ = 833;
const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;
uint32_t nextSample;

const uint16_t SYNC_MARKER = 0x55AA;

#pragma pack(push, 1)
struct Amostra {
    uint16_t sync;
    float x, y, z;
};
#pragma pack(pop)

void setup() {
    Serial.begin(921600);
    unsigned long t0 = millis();
    while (!Serial && (millis() - t0) < 3000) delay(10);

    myIMU.settings.accelEnabled = 1;
    myIMU.settings.accelRange = ACCEL_RANGE_G;
    myIMU.settings.accelSampleRate = SAMPLE_RATE_HZ;
    myIMU.settings.gyroEnabled = 0;

    if (myIMU.begin() != 0) {
        if (Serial) Serial.println("ERRO: falha ao iniciar a IMU");
        while (1) delay(1000);
    }

    nextSample = micros();
}

void loop() {
    uint32_t now = micros();
    if ((int32_t)(now - nextSample) >= 0) {
        nextSample += SAMPLE_PERIOD_US;

        Amostra a;
        a.sync = SYNC_MARKER;
        a.x = myIMU.readFloatAccelX();
        a.y = myIMU.readFloatAccelY();
        a.z = myIMU.readFloatAccelZ();

        Serial.write((const uint8_t*)&a, sizeof(a));
    }
}
