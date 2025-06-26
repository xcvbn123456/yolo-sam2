#!/home/super/anaconda3/envs/yolov9/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
from pathlib import Path
import os
import sys

# === 加载 YOLOv9 模块 ===
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, cv2
from utils.plots import Annotator, colors

class YOLOv9ROSNode:
    def __init__(self):
        rospy.init_node('yolov9_node')
        self.bridge = CvBridge()

        # 参数设定
        weights = rospy.get_param('~weights', '/path/to/best.pt')  # 修改为实际模型路径
        data_yaml = rospy.get_param('~data', '/path/to/data.yaml')
        self.imgsz = rospy.get_param('~imgsz', [640, 640])
        device = rospy.get_param('~device', 'cuda:0')

        # 模型加载
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(weights, device=self.device, data=data_yaml)
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        self.names = self.model.names
        self.sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher('/yolov9/image_result', Image, queue_size=1)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        # 推理输入处理
        im = cv2.resize(cv_image, tuple(self.imgsz))  # 确保尺寸匹配
        im_tensor = torch.from_numpy(im).to(self.device)
        im_tensor = im_tensor.permute(2, 0, 1).float() / 255.0
        if im_tensor.ndim == 3:
            im_tensor = im_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(im_tensor)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        # 可视化
        im0 = cv_image.copy()
        annotator = Annotator(im0, line_width=2, example=str(self.names))

        for det in pred[0]:
            if len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        result_img = annotator.result()
        try:
            ros_img = self.bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
            self.pub.publish(ros_img)
        except Exception as e:
            rospy.logerr(f"cv_bridge publish error: {e}")

if __name__ == '__main__':
    try:
        YOLOv9ROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("something wrong with yolov9")
