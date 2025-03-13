import cv2
import numpy as np
import threading
import queue

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

COLOR_RANGES = {
    "Red1": [(0, 100, 100), (10, 255, 255)],
    "Red2": [(170, 100, 100), (180, 255, 255)],
    "Blue": [(90, 150, 50), (130, 255, 255)]
}

frame_queue = queue.Queue(maxsize=2)
processing = True

def capture_frames():
    while processing:
        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)

thread_capture = threading.Thread(target=capture_frames, daemon=True)
thread_capture.start()

def detect_color(frame, hsv, color_name, lower, upper, min_area=1500):
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            color_bgr = (0, 0, 255) if "Red" in color_name else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
            cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    return mask

while True:
    if frame_queue.empty():
        continue

    frame = frame_queue.get()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_red1 = detect_color(frame, hsv, "Red", *COLOR_RANGES["Red1"])
    mask_red2 = detect_color(frame, hsv, "Red", *COLOR_RANGES["Red2"])
    mask_blue = detect_color(frame, hsv, "Blue", *COLOR_RANGES["Blue"])

    cv2.imshow("USB Webcam Feed", frame)
    cv2.imshow("Red Mask", mask_red1 + mask_red2)
    cv2.imshow("Blue Mask", mask_blue)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

processing = False
cap.release()
cv2.destroyAllWindows()

