#!/home/super/anaconda3/envs/yolov9/bin/python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
from pathlib import Path
import os
import sys
import pickle
from yolov9_ros.msg import BoundingBox,BoundingBoxes
# === 加载 YOLOv9 模块 ===
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

sys.path.append("/home/super/catkin_ws/src/screw_ros/yolov9")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, cv2
from utils.plots import Annotator, colors

class YOLOv9ROSNode:
    def __init__(self):
        rospy.init_node('yolov9_node')
        self.bridge = CvBridge()

        # 参数设定
        weights = rospy.get_param('~weights', '/home/super/catkin_ws/src/screw_ros/yolov9/runs/train/exp7/weights/best.pt')  # 修改为实际模型路径
        data_yaml = rospy.get_param('~data', '/home/super/catkin_ws/src/screw_ros/yolov9/datasets/dataset/data.yaml')
        self.imgsz = rospy.get_param('~imgsz', [640, 640])
        device = rospy.get_param('~device', 'cuda:0')

        # 模型加载
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(weights, device=self.device,dnn=False, data=data_yaml,fp16=False)
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        self.names = self.model.names
        self.sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher('/yolov9/image_result', Image, queue_size=1)
        self.bbox_pub = rospy.Publisher('/yolov9/bboxes',BoundingBoxes,queue_size=10)

    def image_callback(self, msg):
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        # 推理输入处理
        im = cv2.resize(cv_image, tuple(self.imgsz))  # 确保尺寸匹配
        img_rgb = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im_tensor = torch.from_numpy(img_rgb).to(self.device)
        im_tensor = im_tensor.permute(2, 0, 1).float() / 255.0
        if im_tensor.ndim == 3:
            im_tensor = im_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(im_tensor)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        # 可视化
        im0 = cv_image.copy()
        annotator = Annotator(im0, line_width=2, example=str(self.names))
        bbox_msg = BoundingBoxes()
        bbox_msg.header.stamp = rospy.Time.now()
        bbox_msg.header.frame_id = "camera"

        orig_h,orig_w = cv_image.shape[:2]
        infer_h,infer_w = self.imgsz

        scale_x = orig_w/float(infer_w)
        scale_y = orig_h/float(infer_h)

        for det in pred:
            annotator = Annotator(im, line_width=2)
            if det is not None and len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

                    x1,y1,x2,y2 = map(float,xyxy)
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                    
                    x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
                    class_id = int(cls)
                    conf_val = float(conf)

                    box = BoundingBox()
                    box.classid = class_id
                    box.confidence = conf_val
                    box.xmin = x1
                    box.ymin = y1
                    box.xmax = x2
                    box.ymax = y2
                    bbox_msg.boxes.append(box)
                    

        result_img = annotator.result()
        # cv2.imshow(result_img)
        # try:
        ros_img = self.bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
        bbox_msg.header.stamp = rospy.Time.now()
        self.pub.publish(ros_img)
        self.bbox_pub.publish(bbox_msg)
        # except Exception as e:
        #     rospy.logerr(f"cv_bridge publish error: {e}")

if __name__ == '__main__':
    try:
        YOLOv9ROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("something wrong with yolov9")
