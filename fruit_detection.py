import rospy
from std_msgs.msg import String

def publish_freshness(result):
    rospy.init_node('fruit_freshness_detector', anonymous=True)
    pub = rospy.Publisher('/freshness_status', String, queue_size=10)
    rate = rospy.Rate(1)  
    while not rospy.is_shutdown():
        pub.publish(result)
        rate.sleep()

if __name__ == "__main__":
    detected_result = "fresh"  # Replace this with actual detection logic
    publish_freshness(detected_result)
