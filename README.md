# **Fruit Freshness & Industrial Conveyor Belt Automation using ROS**  

This project integrates **AI-based vision detection** with a **ROS-controlled conveyor belt system**. It can be used for **fruit freshness detection**, **product sorting**, **quality control**, and **expiry-based rejection** in industries like **food processing, pharmaceuticals, and manufacturing**.  

🚀 **Key Technologies Used:**  
✅ **Python** – Core programming language.  
✅ **OpenCV** – Image processing and real-time analysis.  
✅ **TensorFlow/PyTorch** – Deep learning for object classification.  
✅ **YOLO** – Fast and efficient object detection.  
✅ **ROS** – Controls the motorized conveyor belt.  

---

## **Applications**  
🔹 **Fruit Freshness Detection** – Moves fresh fruits forward, removes spoiled ones.  
🔹 **Industrial Sorting** – Classifies products based on labels, expiry dates, and defects.  
🔹 **Automated Quality Control** – Detects damages, irregularities, and missing components.  
🔹 **Smart Manufacturing** – Controls conveyor movement based on product type.  

---

## **System Architecture**  

### **1. Object Detection Node (YOLO & OpenCV)**  
- Uses a **camera** to capture images of products on the conveyor belt.  
- YOLO (You Only Look Once) **detects freshness, quality, or defects**.  
- The detection result is **published to a ROS topic `/detection_status`**.  

### **2. Conveyor Belt Motor Control Node**  
- Subscribes to **`/detection_status`** topic.  
- If a product is **fresh/valid**, the **conveyor moves forward**.  
- If a product is **spoiled/expired/defective**, the **conveyor stops or diverts**.  

---

## **Installation & Setup**  

### **1. Install ROS (if not installed)**  
Follow the official ROS Noetic installation guide:  
🔗 [ROS Noetic Installation](http://wiki.ros.org/noetic/Installation/Ubuntu)  

### **2. Clone This Repository**  
```bash
git clone https://github.com/rahulrajak6942/fruit-freshness-detection.git
cd fruit-freshness-detection
```

### **3. Create a ROS Package**  
```bash
cd ~/catkin_ws/src
catkin_create_pkg conveyor_belt std_msgs rospy
```

### **4. Copy the Scripts**  
```bash
cp -r fruit-freshness-detection/conveyer_belt ~/catkin_ws/src/conveyor_belt/
```

### **5. Install Dependencies**  
```bash
pip install opencv-python numpy torch torchvision
```

### **6. Build the Package**  
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

---

## **Running the System**  

### **Step 1: Start ROS Master**  
```bash
roscore
```

### **Step 2: Run the Object Detection Node (YOLO + OpenCV)**  
```bash
rosrun conveyor_belt object_detection.py
```

### **Step 3: Run the Conveyor Belt Motor Controller**  
```bash
rosrun conveyor_belt motor_control.py
```

---

## **Code Explanation**  

### **1. Object Detection Node (YOLO + OpenCV)**
This script detects freshness, expiry, or defects and publishes the result.  

```python
import rospy
from std_msgs.msg import String
import cv2
import torch
from torchvision import transforms

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_and_publish():
    rospy.init_node('object_detector', anonymous=True)
    pub = rospy.Publisher('/detection_status', String, queue_size=10)
    cap = cv2.VideoCapture(0)  

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = model(frame)
        detected_labels = results.pandas().xyxy[0]['name'].tolist()

        status = "fresh" if "fresh_fruit" in detected_labels else "not_fresh"
        pub.publish(status)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_publish()
```

---

### **2. Conveyor Belt Motor Control Node**  
Controls the conveyor belt based on the detection result.  

```python
import rospy
from std_msgs.msg import String
import RPi.GPIO as GPIO

MOTOR_PIN = 18  

def control_motor(status):
    if status == "fresh":
        GPIO.output(MOTOR_PIN, GPIO.HIGH)  
    else:
        GPIO.output(MOTOR_PIN, GPIO.LOW)  

def callback(data):
    rospy.loginfo("Received status: %s", data.data)
    control_motor(data.data)

def motor_controller():
    rospy.init_node('motor_controller', anonymous=True)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_PIN, GPIO.OUT)

    rospy.Subscriber("/detection_status", String, callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        motor_controller()
    except rospy.ROSInterruptException:
        GPIO.cleanup()
```

---

## **Future Enhancements**  
✅ **Multi-Product Sorting** – Move items to different paths based on category.  
✅ **AI-Based Decision Making** – Improve classification using deep learning.  
✅ **Cloud Integration** – Log results for remote monitoring.  
✅ **Edge Processing** – Run on Jetson Nano / Raspberry Pi.  

---

## **Datasets for Training & Testing**  
You can use these datasets to train your model:  

1. **Fruits Fresh and Rotten for Classification**  
   [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)  

2. **Fresh and Stale Images of Fruits and Vegetables**  
   [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)  

3. **Food Freshness Dataset**  
   [🔗 Kaggle Dataset](https://www.kaggle.com/datasets/alinesellwia/food-freshness)  

---

## **Author**  
👤 **Rahul Rajak**  
🔗 [GitHub Profile](https://github.com/rahulrajak6942)
