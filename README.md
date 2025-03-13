# **Automated Fruit Freshness Detection & Sorting System for Conveyor Belts**  

## **Project Overview**  
This project is an **AI-powered fruit freshness detection and sorting system** designed for **conveyor belt automation**. The system integrates **YOLO (You Only Look Once) for object detection**, **ROS (Robot Operating System) for real-time processing**, and **sensor-based expiry date detection** to classify and sort fruits based on their freshness. The goal is to automate quality control in **food processing industries, warehouses, and agricultural setups**.  

## **Key Features**  
 **YOLO-based Object Detection**: Accurately detects and classifies fruits on the conveyor.  
 **Freshness Classification**: Uses a deep learning model to determine if a fruit is fresh or rotten.  
 **Expiry Date Detection**: Integrates sensors (such as **gas sensors** for ethylene detection or **RFID scanners**) to determine shelf life.  
 **ROS Integration**: Enables real-time communication between the camera, detection system, and motor controllers.  
 **Motorized Sorting Mechanism**: Automatically rotates the motor to divert fruits based on classification.  
 **IoT Connectivity (Future Scope)**: Allows remote monitoring and data logging.  

## **Technology Stack**  
🔹 **Deep Learning & Computer Vision**: YOLOv8, OpenCV, PyTorch  
🔹 **Embedded Systems & Sensors**: Raspberry Pi / Arduino, RFID sensors, Gas Sensors (for spoilage detection)  
🔹 **ROS for Real-Time Processing**: Communication between camera, sensors, and motors  
🔹 **Motor Control**: Servo/Motor driver for conveyor belt sorting  
🔹 **Cloud & IoT (Future Enhancement)**: Remote monitoring and data analytics  

## **How It Works**  
1. **YOLO-Based Object Detection**: The camera captures images of fruits on the conveyor belt and detects their type.  
2. **Freshness Classification**:  
   - A deep learning model predicts if the fruit is **fresh or rotten** based on visual characteristics.  
   - Sensors detect spoilage gases or check expiry tags (for packed products).  
3. **ROS-Based Processing**:  
   - The detected information is sent via ROS nodes to control the sorting mechanism.  
   - ROS ensures seamless communication between sensors, motors, and the detection system.  
4. **Motorized Sorting System**:  
   - If **fresh**, the motor moves the fruit to the **"Fresh"** section.  
   - If **rotten or expired**, the system diverts it to the **"Reject"** section.  
5. **Real-Time Monitoring & Data Logging** (Future Scope):  
   - IoT integration can enable cloud storage of detection results.  
   - Alerts and reports can be generated for quality control.  

## **Applications**  
🔸 **Food Processing & Packaging Industries**  
🔸 **Supermarkets & Warehouses** (Automated sorting for quality control)  
🔸 **Agriculture & Post-Harvest Processing**  
🔸 **Smart Supply Chains with IoT Monitoring**  

## **Future Enhancements**  
🚀 Multi-fruit classification (apples, bananas, mangoes, etc.)  
🚀 AI-based **ripeness prediction**  
🚀 IoT-enabled **cloud monitoring dashboard**  
🚀 Autonomous robotic arm for more precise sorting  

​You can find fruit freshness datasets on Kaggle that are suitable for  project datasets:

1. **Fruits Fresh and Rotten for Classification**  
   This dataset contains images of fresh and rotten fruits, ideal for classification tasks.  
   [Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

2. **Fresh and Stale Images of Fruits and Vegetables**  
   This dataset includes images of six fruits and vegetables, each labeled as fresh or stale.  
   [Fresh and Stale Images of Fruits and Vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)

3. **Food Freshness**  
   This dataset contains images of carrots, tomatoes, and oranges, categorized into three freshness classes.  
   [Food Freshness](https://www.kaggle.com/datasets/alinesellwia/food-freshness)

These datasets should provide a solid foundation for developing and testing your fruit freshness detection system. 
