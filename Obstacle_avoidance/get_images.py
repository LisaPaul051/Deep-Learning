import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class SaveImages:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/oak/rgb/image_raw', Image, self.image_callback)
        self.obstacle_sub = rospy.Subscriber('/obstacle_detector/obstacle_detected', Bool, self.obstacle_callback)
        
        # Folder to save images
        self.save_dir = "beds_without_obstacles"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        rospy.loginfo(f"Saving images to: {os.path.abspath(self.save_dir)}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("Received Image", cv_image)
            cv2.waitKey(1)

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(self.save_dir, f"image_{timestamp}.jpg")

            # Save the image
            # cv2.imwrite(filename, cv_image)
            # rospy.loginfo(f"Saved: {filename}")

        except Exception as e:
            rospy.logerr(f"Error converting or saving image: {e}")

if __name__ == '__main__':
    try:
        SaveImages()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
