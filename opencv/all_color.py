import cv2
import numpy as np
import time

#IP_CAMERA_URL = "http://192.168.248.241:8080/video"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to connect to phone camera!")
    exit()

COLOR_RANGES = {
    "Red1": [(0, 100, 100), (10, 255, 255)],
    "Red2": [(170, 100, 100), (180, 255, 255)],
    "Blue": [(90, 150, 50), (130, 255, 255)],
    "Green": [(40, 40, 40), (80, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)],
    "Orange": [(10, 100, 100), (20, 255, 255)],
    "Purple": [(130, 50, 50), (160, 255, 255)]
}

STABILITY_TIME = 1.0  
last_detected_color = None
last_detection_time = time.time()

def preprocess_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def detect_color(frame, hsv, color_name, lower, upper, min_area=2000):
    mask = preprocess_mask(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            color_bgr = (0, 0, 255) if "Red" in color_name else \
                        (255, 0, 0) if color_name == "Blue" else \
                        (0, 255, 0) if color_name == "Green" else \
                        (0, 255, 255) if color_name == "Yellow" else \
                        (0, 165, 255) if color_name == "Orange" else \
                        (128, 0, 128)  # Purple
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
            cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame!")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_color = None

    for color, (lower, upper) in COLOR_RANGES.items():
        if detect_color(frame, hsv, color, lower, upper):
            detected_color = "Red" if "Red" in color else color
            break

    if detected_color != last_detected_color:
        last_detected_color = detected_color
        last_detection_time = time.time()

    if detected_color and (time.time() - last_detection_time) >= STABILITY_TIME:
        print(f"Detected Color: {detected_color}")

    cv2.imshow("Color Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

