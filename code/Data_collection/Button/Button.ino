/*
  xiao-sense-input
  Reads in from analog inputs and IMU then outputs the readings via serial.
  A button press starts and stops the data stream.
*/
#include <LSM6DS3.h>
#include <Wire.h>
#include <MadgwickAHRS.h>

Madgwick filter;                // filter for IMU calibration
LSM6DS3 myIMU(I2C_MODE, 0x6A);  // IMU

float ax, ay, az, gx, gy, gz;  // IMU data

const int buttonPin = 6;    // the number of the pushbutton pin
int printState = LOW;      // the current state of printing via serial
int buttonState;            // the current reading from the input pin
int lastButtonState = LOW;  // the previous reading from the input pin

// the following variables are unsigned longs because the time, measured in
// milliseconds, will quickly become a bigger number than can be stored in an int.
unsigned long lastDebounceTime = 0;  // the last time the output pin was toggled
unsigned long debounceDelay = 500;    // the debounce time; increase if the output flickers

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);

  // set initial LED state
  digitalWrite(LED_BUILTIN, printState);

  // begin serial communication
  Serial.begin(9600);

  while (!Serial)
    ;
  //Call .begin() to configure the IMUs
  if (myIMU.begin() != 0) {
    Serial.println("Device error");
  } else {
    Serial.println("aX,aY,aZ,gX,gY,gZ");
  }
  filter.begin(25);
}

void loop() {
  // read the state of the switch into a local variable:
  int reading = digitalRead(buttonPin);

  // check to see if you just pressed the button
  // (i.e. the input went from LOW to HIGH), and you've waited long enough
  // since the last press to ignore any noise:

  // If the switch changed, due to noise or pressing:
  if (reading != lastButtonState) {
    // reset the debouncing timer
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    // whatever the reading is at, it's been there for longer than the debounce
    // delay, so take it as the actual current state:

    // if the button state has changed:
    if (reading != buttonState) {
      buttonState = reading;

      // only toggle the LED if the new button state is HIGH
      if (buttonState == HIGH) {
        printState = !printState;
      }
    }
  }

  // set the LED:
  digitalWrite(LED_BUILTIN, !printState);

  if (printState) {
    // read and print sensor data
    readData();
  }

  // save the reading. Next time through the loop, it'll be the lastButtonState:
  lastButtonState = reading;
}

void readData() {
  // read in analog inputs
  int sensor0 = analogRead(0);
  int sensor1 = analogRead(1);
  int sensor2 = analogRead(2);
  int sensor3 = analogRead(3);
  int sensor4 = analogRead(4);

  // read in IMU data
  ax = myIMU.readFloatAccelX();
  ay = myIMU.readFloatAccelY();
  az = myIMU.readFloatAccelZ();
  gx = myIMU.readFloatGyroX();
  gy = myIMU.readFloatGyroY();
  gz = myIMU.readFloatGyroZ();

  // send out the data
  Serial.print(sensor0);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.print(sensor1);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.print(sensor2);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.print(sensor3);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.print(sensor4);  // prints a label
  Serial.print("\t"); // prints a tab

  Serial.print(ax);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.print(ay);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.print(az);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.print(gx);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.print(gy);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.print(gz);  // prints a label
  Serial.print("\t"); // prints a tab
  Serial.println();
}