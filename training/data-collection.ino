#include<Wire.h>
#include <Keypad.h>


typedef struct {
    int16_t AcX;
    int16_t AcY;
    int16_t AcZ;
    int16_t GyX;
    int16_t GyY;
    int16_t GyZ;
} GyroValues;

const byte ROWS = 4;
const byte COLS = 4; 

const char hexaKeys[ROWS][COLS] = {
  {'1','2','3','A'},
  {'4','5','6','B'},
  {'7','8','9','C'},
  {'*','0','#','D'}
};

const int numClasses = 4;
const char* classes[numClasses] = {
  "wave",
  "circle_cw",
  "circle_ccw",
  "rotate"
};

byte rowPins[ROWS] = {9, 8, 7, 6};
byte colPins[COLS] = {5, 4, 3, 2}; 

const int MPU_addr=0x68;
GyroValues latestValues;
Keypad keypad = Keypad(makeKeymap(hexaKeys), rowPins, colPins, ROWS, COLS); 

void setup(){
  Wire.begin();
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0);     // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);
  Serial.begin(9600);
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

void printGyroValues(GyroValues *values) {
  Serial.print(latestValues.AcX);Serial.print(";");
  Serial.print(latestValues.AcY);Serial.print(";");
  Serial.print(latestValues.AcZ);Serial.print(";");
  Serial.print(latestValues.GyX);Serial.print(";");
  Serial.print(latestValues.GyY);Serial.print(";");
  Serial.print(latestValues.GyZ);Serial.println("");
}


void loop(){
  char keyPressed = keypad.getKey();
  if (keyPressed){
      int classIndex = atoi(&keyPressed);
      if (classIndex > 0 && classIndex <= numClasses) {
        Serial.print(keyPressed);
        Serial.print(';');
        Serial.print(classes[classIndex - 1]);
        Serial.println(";sample start");
      } else if (keyPressed == '#') {
        Serial.println("sample end");
      }
  }

  readGyro(&latestValues);
  printGyroValues(&latestValues);
  delay(200);
}