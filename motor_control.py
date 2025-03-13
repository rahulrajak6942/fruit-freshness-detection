import rospy
from std_msgs.msg import String
import RPi.GPIO as GPIO

MOTOR_PIN = 18  # Change according to your setup

def control_motor(status):
    if status == "fresh":
        GPIO.output(MOTOR_PIN, GPIO.HIGH)  # Rotate motor
    else:
        GPIO.output(MOTOR_PIN, GPIO.LOW)  # Stop motor

def callback(data):
    rospy.loginfo("Received freshness status: %s", data.data)
    control_motor(data.data)

def motor_controller():
    rospy.init_node('motor_controller', anonymous=True)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_PIN, GPIO.OUT)

    rospy.Subscriber("/freshness_status", String, callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        motor_controller()
    except rospy.ROSInterruptException:
        GPIO.cleanup()
