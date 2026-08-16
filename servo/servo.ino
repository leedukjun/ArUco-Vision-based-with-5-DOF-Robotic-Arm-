#include <Adafruit_PWMServoDriver.h>
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setPWMFreq(50); // Tần số chuẩn cho Servo
}

void loop() {
  if (Serial.available()) {
    // Điều khiển servo đang cắm ở cổng số 0
    pwm.writeMicroseconds(0, Serial.parseInt()); 
  }
}