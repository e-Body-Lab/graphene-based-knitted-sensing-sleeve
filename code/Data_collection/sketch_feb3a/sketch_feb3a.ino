//This script reads IMU and resistance data
//Data transmitted through type C cable
#include <LSM6DS3.h>
#include <Wire.h>
#include <MadgwickAHRS.h>
Madgwick filter;
LSM6DS3 myIMU(I2C_MODE, 0x6A);    
float ax, ay, az, gx, gy, gz;
float roll, pitch, heading;
int a=0;
int analogPin0 = 0;
int analogPin1 = 1;
int analogPin2 = 2;
int analogPin3 = 3;
int analogPin4 = 4;
int raw0 = 0;
int raw1 = 1;
int raw2 = 2;
int raw3 = 3;
int raw4 = 4;

unsigned long current_time;
unsigned long previous_time;
void setup() {
  analogReadResolution(12);
  Serial.begin(115200);
  pinMode(6,INPUT);
  Serial.println("start setup");
  //pinMode(3,OUTPUT);
  while (!Serial);
  //Call .begin() to configure the IMUs
  if (myIMU.begin() != 0) {
    Serial.println("Device error");
  } else {
    Serial.println("aX,aY,aZ,gX,gY,gZ");
  }
  filter.begin(25);
}

void loop() {
  int i=digitalRead(6); 
  // Serial.println(1);
  // if (a==0){ 
    // if (i==HIGH){  
      // delay(50);
      previous_time = millis();

      while (1) {
      current_time = millis();  
      if(current_time - previous_time >= 10){
      previous_time = current_time;
      raw0 = analogRead(analogPin0);
      raw1 = analogRead(analogPin1);
      raw2 = analogRead(analogPin2);
      raw3 = analogRead(analogPin3);
      raw4 = analogRead(analogPin4);
      Serial.println(-1);
      Serial.println(raw0);
      Serial.println(raw1);
      Serial.println(raw2);
      Serial.println(raw3);
      Serial.println(raw4);
      ax = myIMU.readFloatAccelX();
      ay = myIMU.readFloatAccelY();
      az = myIMU.readFloatAccelZ();
      gx = myIMU.readFloatGyroX();
      gy = myIMU.readFloatGyroY();
      gz = myIMU.readFloatGyroZ();
      Serial.println(ax);
      Serial.println(ay);
      Serial.println(az);
      Serial.println(gx);
      Serial.println(gy);
      Serial.println(gz);
      // delay(9.5);
      i=digitalRead(6);
      // if (i==HIGH) {
      //   break;
      //   }
      // }
      a++;

      }
      // delay(5);

    }

}