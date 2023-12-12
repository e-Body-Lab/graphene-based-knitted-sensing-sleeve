// Reads resistance data and save to microsd card
#include <SD.h>
#include <SPI.h>

File myFile;

int pinCS = 7; 
int analogPin0 = 0;
int analogPin1 = 1;
int analogPin2 = 2;
int analogPin3 = 3;
int analogPin4 = 4;
int raw0 = 0;
int raw1 = 0;
int raw2 = 0;
int raw3 = 0;
int raw4 = 0;
void setup() {
    
  Serial.begin(9600);
  pinMode(pinCS, OUTPUT);
  if (SD.begin())
  {
    Serial.println("SD card is ready to use.");
  } else
  {
    Serial.println("SD card initialization failed");
    return;
  }
  myFile = SD.open("test.txt", FILE_WRITE);
  if (myFile){
    myFile.println("New test");
    myFile.close();
  }
  
}
void loop() {
  myFile = SD.open("test.txt", FILE_WRITE);
  if (myFile) {
    raw0 = analogRead(analogPin0);
    raw1 = analogRead(analogPin1);
    raw2 = analogRead(analogPin2);
    raw3 = analogRead(analogPin3);
    raw4 = analogRead(analogPin4);
    myFile.println(raw0);
    myFile.println(raw1);
    myFile.println(raw2);
    myFile.println(raw3);
    myFile.println(raw4);
    Serial.println("New reading, R1-R5:");
    Serial.println(raw0);
    Serial.println(raw1);
    Serial.println(raw2);
    Serial.println(raw3);
    Serial.println(raw4);
    myFile.close();
  }
  else {
    Serial.println("error opening test.txt");
  }

}
