# Human Detection and Alert System

This project is a real-time human detection and alert system using a YOLOv8 model. The system captures video from a camera, detects humans, and sends an email alert with an image when a human is detected.

## Features

- Real-time human detection using YOLOv8.
- Email alert with an attached image upon detecting a human.
- Adjustable email interval to prevent flooding.
- Simple and effective corner rectangle visualization for detected humans.

## Requirements

- Python 3.7+
- OpenCV
- cvzone
- smtplib
- ultralytics
# Code Overview

## HumanDetection.py

This script initializes the camera, loads the YOLOv8 model, and processes the video stream in real-time to detect humans. When a human is detected, an email with an attached image is sent to the specified recipient.

### Key Sections of the Code

1. **Camera Initialization**: Sets up the camera with the specified resolution.
2. **Model Loading**: Loads the YOLOv8 model.
3. **Human Detection**: Processes the video stream and detects humans using the YOLOv8 model.
4. **Email Alert**: Sends an email with an image attachment when a human is detected.
5. **Email Interval**: Prevents email flooding by setting a minimum interval between email alerts.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or additions you would like to make.

### Acknowledgements

- YOLOv8 by Ultralytics
- OpenCV
- cvzone
