import cv2
import numpy as np
import rospy
from std_msgs.msg import Int32
import time

cap = cv2.VideoCapture(2)  # USB Webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffering lag

COLOR_RANGES = {
    "Red1": [(0, 100, 100), (10, 255, 255)],
    "Red2": [(170, 100, 100), (180, 255, 255)],
    "Blue": [(90, 150, 50), (130, 255, 255)]
}

STABILITY_TIME = 0.5  
last_detected_color = None
last_detection_time = time.time()
servo_angle = None

def preprocess_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.medianBlur(mask, 5)  # Reduce noise
    return mask

def detect_color(frame, hsv, color_name, lower, upper, min_area=1500):
    mask = preprocess_mask(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            color_bgr = (0, 0, 255) if "Red" in color_name else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
            cv2.putText(frame, color_name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return True
    return False

rospy.init_node("color_detector")
servo_pub = rospy.Publisher("/servo_angle_cmd", Int32, queue_size=1)

frame_skip = 2  # Skip every 2nd frame for better performance
frame_count = 0

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame!")
        continue

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frame to improve FPS

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_color = None

    if detect_color(frame, hsv, "Red", *COLOR_RANGES["Red1"]) or detect_color(frame, hsv, "Red", *COLOR_RANGES["Red2"]):
        detected_color = "Red"
    elif detect_color(frame, hsv, "Blue", *COLOR_RANGES["Blue"]):
        detected_color = "Blue"

    if detected_color != last_detected_color:
        last_detected_color = detected_color
        last_detection_time = time.time()

    if detected_color and (time.time() - last_detection_time) >= STABILITY_TIME:
        new_angle = 180 if detected_color == "Red" else 90
        if new_angle != servo_angle:
            servo_pub.publish(new_angle)
            servo_angle = new_angle
            print(f"Published: {servo_angle}° (Detected: {detected_color})")

    cv2.imshow("Color Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

