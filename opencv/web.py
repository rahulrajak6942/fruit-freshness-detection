import threading

def process_frame():
    global frame
    while True:
        ret, temp_frame = cap.read()
        if not ret:
            break
        frame = temp_frame.copy()

frame = None
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

