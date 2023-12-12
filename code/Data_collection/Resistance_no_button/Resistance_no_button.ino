#include <LSM6DS3.h>
#include <Wire.h>
#include <MadgwickAHRS.h>

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

long previousMillis = 0;  // last timechecked, in ms

void setup() {
  Serial.begin(9600);

  pinMode(LED_BUILTIN, OUTPUT);
  Serial.println("Test Starts");
}
void loop() {
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
}