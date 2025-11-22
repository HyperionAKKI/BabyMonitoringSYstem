
*int soundSensorPin = 2; 
#include <ezLED.h>
ezLED led(7);
void setup() {
    Serial.begin(9600); 
    pinMode(soundSensorPin, INPUT);
    pinMode(4,OUTPUT);

}

void loop() {
  led.loop();
    int period = pulseIn(soundSensorPin, HIGH); 

   
    int frequency = 1000000 / period;

    if (frequency >= 200 && frequency <= 4000) {
        Serial.print("Frequency (Hz): ");
        Serial.println(frequency);
        led.turnON();
        led.turnOFF(50);
        digitalWrite(4,1);
        delay(300);
        digitalWrite(4,0);

    }

    delay(100); 
}
