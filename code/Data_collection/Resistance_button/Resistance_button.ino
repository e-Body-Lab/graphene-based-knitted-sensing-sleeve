#include <LSM6DS3.h>
#include <Wire.h>
#include <MadgwickAHRS.h>
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
void setup() {
  Serial.begin(9600);
  pinMode(6,INPUT);
}

void loop() {
  int i=digitalRead(6);  
  if (a==0){  
    if (i==HIGH){  
      delay(50); 
      while (1) {
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
      delay(9.5);
      i=digitalRead(6);
      if (i==HIGH) {
        break;
        }
      }
      a++;
    }
  }
  else
  if(a==1){  
    if(i==HIGH){
      delay(50);
      a--;
    }
  }
}
