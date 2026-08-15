#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000); 
  pwm.setPWMFreq(50);  
}
void loop() {
  if (Serial.available()) {

    int targetAngle = Serial.readStringUntil('\n').toInt(); 

    int pulse = map(targetAngle, 0, 180, 150, 600);
  
    pwm.setPWM(0, 0, pulse); 
  }
}