#include <Wire.h>

#include "lstmCell.h"
#include "denseLayer.h"
#include "weights.h"

const int numClasses = 5;
const char* classes[numClasses] = {
  "default",
  "wave",
  "circle_cw",
  "circle_ccw",
  "rotate"
};

int inputSize =  6;
int hiddenSize =  10;

float output[numClasses];
LSTMCell lstmCell(inputSize, hiddenSize);
DenseLayer denseLayer(inputSize, numClasses);

typedef struct {
    int16_t AcX;
    int16_t AcY;
    int16_t AcZ;
    int16_t GyX;
    int16_t GyY;
    int16_t GyZ;
} GyroValues;

const int MPU_addr=0x68;
GyroValues latestValues;
float preprocessedValues[6];

void setupGyroscope() {
  Wire.begin();
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0);     // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);
}

GyroValues readGyro(GyroValues *values) {
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_addr,14,true); 

  values->AcX = Wire.read()<<8|Wire.read();  
  values->AcY = Wire.read()<<8|Wire.read();
  values->AcZ = Wire.read()<<8|Wire.read();
  Wire.read()<<8|Wire.read(); // Discard the temperature value
  values->GyX = Wire.read()<<8|Wire.read();
  values->GyY = Wire.read()<<8|Wire.read();
  values->GyZ = Wire.read()<<8|Wire.read();
}

void preprocessGyroData(const GyroValues &values) {
  // TODO: replace with actual preprocessing
  preprocessedValues[0] = 0.1 * values.AcX;
  preprocessedValues[1] = 0.1 * values.AcY;
  preprocessedValues[2] = 0.1 * values.AcZ;
  preprocessedValues[3] = 0.1 * values.GyX;
  preprocessedValues[4] = 0.1 * values.GyY;
  preprocessedValues[5] = 0.1 * values.GyZ;
}

int runInference(const float input[]) {
    lstmCell.forward(input);
    denseLayer.forward(lstmCell.hiddenState, output);
}

void setup() {
  Serial.begin(9600);
  setupGyroscope();
}

void loop() {
  readGyro(&latestValues);
  preprocessGyroData(latestValues);
  int classIndex = runInference(preprocessedValues);
  Serial.println(classes[classIndex]);
}
