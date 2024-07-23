from ultralytics import YOLO
import cv2
import cvzone
import smtplib
import math
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import time

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, 416)  # width
cap.set(4, 416)  # height

# Load the YOLO model
model = YOLO('yolov8l.pt')

# Class names from the model (only keeping "person")
classNames = ["person"]

# Email settings
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_email_password'
RECIPIENT_ADDRESS = 'recipient_email@gmail.com'


def send_email_alert(image_path):
    msg = MIMEMultipart()
    msg['Subject'] = 'Alert: Human Detection'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_ADDRESS

    text = MIMEText("Human detected. See attached image.")
    msg.attach(text)

    with open(image_path, 'rb') as f:
        img_data = f.read()

    image = MIMEImage(img_data, name=os.path.basename(image_path))
    msg.attach(image)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(msg['From'], [msg['To']], msg.as_string())

# Initialize email delay
last_email_time = time.time()
email_interval = 60  # Time interval between emails in seconds

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    human_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if classNames[cls] == 'person':
                human_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)

    if human_detected:
        current_time = time.time()
        if current_time - last_email_time > email_interval:
            image_path = 'detected.jpg'
            cv2.imwrite(image_path, img)
            send_email_alert(image_path)
            print("Alarm! Human detected.")
            last_email_time = current_time

    cv2.imshow("IMAGE", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
