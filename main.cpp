#include <Servo.h>
#include <ros.h>
#include <std_msgs/Int32.h>

ros::NodeHandle nh;
Servo myServo;

void servo_angle_cb(const std_msgs::Int32& msg) {
  int angle = msg.data;
  myServo.write(angle);  // Set the servo angle
}

ros::Subscriber<std_msgs::Int32> sub("servo_angle_cmd", &servo_angle_cb);

void setup() {
  myServo.attach(9); // Connect servo to pin 9
  nh.initNode();
  nh.subscribe(sub);
}

void loop() {
  nh.spinOnce();
  delay(10);
}
