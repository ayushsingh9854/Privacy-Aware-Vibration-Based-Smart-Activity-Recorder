#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified();

void setup(void)
{
  Serial.begin(9600);

  if(!accel.begin())
  {
    Serial.println("No ADXL345 detected. Check wiring!");
    while(1);
  }
}

void loop(void)
{
  sensors_event_t event;
  accel.getEvent(&event);

  Serial.print(event.acceleration.x); Serial.print(",");
  Serial.print(event.acceleration.y); Serial.print(",");
  Serial.print(event.acceleration.z); Serial.println("");
  

  delay(10);
}