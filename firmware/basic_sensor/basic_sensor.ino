#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

SparkFun_VL53L5CX myImager;
VL53L5CX_ResultsData measurementData; 

void setup() {
  Serial.begin(115200);
  delay(1000); 

  Serial.println("Starting...");

  Wire.begin(6, 7); // Ensure these are your correct pins!
  Wire.setClock(400000); 

  if (myImager.begin(0x29, Wire) == false) {
    Serial.println("Sensor NOT found. Please Unplug/Replug USB and try again.");
    while (1);
  }

  Serial.println("Sensor Ready! Reading distances...");
  myImager.setResolution(8 * 8); 
  myImager.setRangingFrequency(10); 
  myImager.startRanging();
}

void loop() {
  if (myImager.isDataReady() == true) {
    if (myImager.getRangingData(&measurementData)) {
      // This loop prints RAW NUMBERS, which Python needs
      for (int y = 0; y <= 7; y++) {
        for (int x = 0; x <= 7; x++) {
          Serial.print(measurementData.distance_mm[(y * 8) + x]);
          Serial.print("\t"); 
        }
        Serial.println(); 
      }
      Serial.println(); 
    }
  }
}