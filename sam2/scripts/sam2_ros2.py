#!/home/super/anaconda3/envs/sam2/bin/python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
import sys
sys.path.append("/home/super/catkin_ws/src/screw_ros/sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from yolov9_ros.msg import BoundingBoxes
class SAM2FromYOLO:
    def __init__(self):
        rospy.init_node("sam2_from_yolo_node2", anonymous=True)
        self.bridge = CvBridge()

        # SAM2 模型参数
        ckpt = rospy.get_param("~checkpoint_path", "/home/super/catkin_ws/src/screw_ros/sam2/checkpoints/sam2_hiera_tiny.pt")
        cfg = rospy.get_param("~config_path", "sam2_hiera_t.yaml")
        self.device = rospy.get_param("~device", "cuda:0")

        rospy.loginfo("Loading SAM2 model...")
        model = build_sam2(cfg, ckpt, device=self.device)
        self.predictor = SAM2ImagePredictor(model)

        # 订阅 YOLOv9 结果图像（假设图像中已标注检测框）
        self.last_image = None 
        rospy.Subscriber("/camera2/color/image_raw",Image,self.image_callback,queue_size=1,buff_size=2**24)
        rospy.Subscriber("/yolov9/bboxes2",BoundingBoxes,self.bbox_callback,queue_size=1)

        self.mask_pub = rospy.Publisher("/sam2/mask_image2", Image, queue_size=1)
        self.vis_pub = rospy.Publisher("/sam2/vis_image2",Image,queue_size=1)
        rospy.loginfo("SAM2 node ready and waiting for YOLOv9 input...")



    def image_callback(self,msg):
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("图像转化失败")

    def bbox_callback(self, msg):

        if self.last_image is None:
            rospy.logwarn("未收到图像~")
            return

        # try:
        rgb_image = cv2.cvtColor(self.last_image,cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_image)

        input_points = []
        for box in msg.boxes:
            if box.classid == 0 and box.confidence >=0.5:
                cx = int((box.xmin + box.xmax) / 2)
                cy = int((box.ymin + box.ymax) / 2)
                input_points.append([cx,cy])
        if not input_points:
            rospy.loginfo("当前未检测到螺栓与螺母")
            return
            
        input_points = np.array(input_points)  # 假设只有一个目标在中心
        input_labels = np.ones(len(input_points),dtype=np.int32)
 
        all_masks = []
        for pt in input_points:
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([pt]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            best_mask = masks[np.argmax(scores)]
            all_masks.append(best_mask)

        if not all_masks:
            print("当前没有有效Mask")
            return

        combined_mask = np.any(np.stack(all_masks),axis=0).astype(np.uint8) * 255
        mask_rgb = cv2.merge([combined_mask]*3)

        # 将分割图像转换回 ROS 格式发布
        ros_mask = self.bridge.cv2_to_imgmsg(mask_rgb, encoding="bgr8")
        self.mask_pub.publish(ros_mask)

        #现在要构建可视化图像
        vis_img = self.last_image.copy()
        blue_mask = np.zeros_like(vis_img)
        blue_mask[:,:,0] = combined_mask

        alpha = 0.3
        vis_img = cv2.addWeighted(blue_mask,alpha,vis_img,1-alpha,0)

        for pt in input_points:
            cx,cy = int(pt[0]),int(pt[1])
            cv2.circle(vis_img,(cx,cy),6,(0,255,0),-1)
        
        vis_ros = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
        self.vis_pub.publish(vis_ros)
        
        # except Exception as e:
        #     rospy.logerr("SAM2 prediction error: %s", e)

if __name__ == "__main__":
    try:
        SAM2FromYOLO()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass