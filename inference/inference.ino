#include <Wire.h>

#include "dataBuffer.h"
#include "denseLayer.h"
#include "lstmCell.h"
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
int seqLen = 10;

DataBuffer buffer;
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

const int MPU_addr = 0x68;
GyroValues latestValues;
float preprocessedValues[6];

int predictionsToClass(const float arr[numClasses]) {
    int largestIndex = 0;
    float largestValue = arr[0];

    for (int i = 1; i < numClasses; i++) {
        if (arr[i] > largestValue) {
            largestIndex = i;
            largestValue = arr[i];
        }
    }

    return largestIndex;
}

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
  preprocessedValues[0] = (float)(values.AcX+24064)/(float)50236;
  preprocessedValues[1] = (float)(values.AcY+9304)/(float)30308;
  preprocessedValues[2] = (float)(values.AcZ+9712)/(float)42479;
  preprocessedValues[3] = (float)(values.GyX+32039)/(float)53771;
  preprocessedValues[4] = (float)(values.GyY+32768)/(float)65535;
  preprocessedValues[5] = (float)(values.GyZ+32768)/(float)65535;
}

int runInference() {
  for (int i = 0; i < seqLen; i++) {
    lstmCell.forward(buffer.access(i));
  }
    denseLayer.forward(lstmCell.hiddenState, output);
  return predictionsToClass(output);

void setup() {
  Serial.begin(9600);
  setupGyroscope();
}

void loop() {
  readGyro(&latestValues);
  preprocessGyroData(latestValues);
  buffer.push(preprocessedValues);
  int classIndex = runInference();
  Serial.println(classes[classIndex]);
}
