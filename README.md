Baby Monitoring System 👶🔔
A smart baby monitoring solution designed to assist hearing-impaired parents by detecting baby cries and providing immediate alerts through vibration and LED indicators.



🧠 Project Overview
This system employs a sound sensor calibrated to the frequency range of baby cries. Upon detection, it activates a vibration motor and an LED light to notify parents who may not hear the baby's cries.

What is Happening->

Sound Sensor detect frequency of baby crying(babies cry on a fixed scale of frequencies).

Noise suppression using damping scale on noise sensor.

The sound sensor then responds by sending the frequency on Serial Monitor.

A LED is blinked and a motor is activated.



🎯 Features
Sound Detection: Monitors ambient noise to identify baby cries.

Vibration Alert: Activates a vibration motor to provide tactile feedback.

Visual Indicator: Illuminates an LED to offer a visual cue.

Real-Time Response: Ensures immediate alerts upon detection.



🛠️ Hardware Components
Arduino-compatible microcontroller (e.g., Arduino Uno)

Sound sensor module (e.g., KY-038)

Vibration motor module

LED indicator

Resistors and connecting wires

Breadboard or PCB for circuit assembly
Medium

🧩 Software
Firmware: Written in C++ using the Arduino IDE.

Main Script: BabyMonitoringSystem.ino

Video Demonstration: BabyMonitoringSystemVideo.mp4



📂 Repository Structure



BabyMonitoringSYstem/
├── BabyMonitoringSystem.ino        # Arduino source code
├── BabyMonitoringSystemVideo.mp4   # Demonstration video
└── .gitattributes                  # Git attributes configuration
🚀 Getting Started
Clone the Repository:

git clone https://github.com/HyperionAKKI/BabyMonitoringSYstem.git

2. Set Up Hardware: Assemble the components as per the schematic.
3. Upload Firmware: Use the Arduino IDE to upload BabyMonitoringSystem.ino to your microcontroller.
4. Test the System: Simulate baby cries to ensure the vibration and LED activate accordingly.



📹 Demonstration
For a visual demonstration of the system in action, refer to the BabyMonitoringSystemVideo.mp4 file included in the repository.

📄 License
This project is open-source and available under the MIT License.

🙏 Acknowledgments
Developed by HyperionAKKI to provide an accessible solution for hearing-impaired parents

