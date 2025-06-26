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
# === 加载 YOLOv9 模块 ===
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

sys.path.append("/home/super/catkin_ws/src/screw_ros/yolov9")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, cv2
from utils.plots import Annotator, colors

imgsz = [640, 640]
bridge = CvBridge()
weights = '/home/super/catkin_ws/src/screw_ros/yolov9/runs/train/exp7/weights/best.pt'
data_yaml='/home/super/catkin_ws/src/screw_ros/yolov9/datasets/dataset/data.yaml'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights, device=device,dnn=False, data=data_yaml,fp16=False)
with open('/home/super/catkin_ws/src/screw_ros/yolov9/onemsg.pkl','rb') as f:
    msg = pickle.load(f)

cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
cv2.imshow("yolo_input:",cv_image)
print(cv_image.shape)
cv2.waitKey(3)

im = cv2.resize(cv_image, tuple(imgsz))  # 确保尺寸匹配
img_rgb = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
im_tensor = torch.from_numpy(img_rgb).to(device)
im_tensor = im_tensor.permute(2, 0, 1).float() / 255.0
if im_tensor.ndim == 3:
    im_tensor = im_tensor.unsqueeze(0)

with torch.no_grad():
    pred = model(im_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

for det in pred:
    annotator = Annotator(im, line_width=2)
    if det is not None and len(det):
        det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im.shape).round()
        for *xyxy, conf, cls in det:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(int(cls), True))
result_img = annotator.result()
cv2.imshow("yolov9",result_img)
cv2.waitKey(1000)





# 可视化
# im0 = cv_image.copy()
# annotator = Annotator(im0, line_width=2, example=str(model.names))

# for det in pred[0]:
#     if len(det):
#         det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
#         for *xyxy, conf, cls in det:
#             label = f'{model.names[int(cls)]} {conf:.2f}'
#             annotator.box_label(xyxy, label, color=colors(int(cls), True))

# result_img = annotator.result()
# try:
#     ros_img = bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
# except Exception as e:
#     rospy.logerr(f"cv_bridge publish error: {e}")


# class YOLOv9ROSNode:
#     def __init__(self):
#         rospy.init_node('yolov9_node')
#         self.bridge = CvBridge()

#         # 参数设定
#         weights = rospy.get_param('~weights', '/home/super/catkin_ws/src/screw_ros/yolov9/runs/train/exp7/weights/best.pt')  # 修改为实际模型路径
#         data_yaml = rospy.get_param('~data', '/home/super/catkin_ws/src/screw_ros/yolov9/datasets/dataset/data.yaml')
#         self.imgsz = rospy.get_param('~imgsz', [640, 640])
#         device = rospy.get_param('~device', 'cuda:0')

#         # 模型加载
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.model = DetectMultiBackend(weights, device=self.device,dnn=False, data=data_yaml,fp16=False)
#         self.model.eval()
#         self.model.warmup(imgsz=(1, 3, *self.imgsz))

#         self.names = self.model.names
#         self.sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
#         self.pub = rospy.Publisher('/yolov9/image_result', Image, queue_size=1)

#     def image_callback(self, msg):
        
#         save_path = "/home/super/catkin_ws/src/screw_ros/yolov9/onemsg.pkl"
#         with open(save_path,'wb') as f:
#             pickle.dump(msg,f)
#             print("保存数据")
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             cv2.imshow("yolo_input:",cv_image)
#             print(cv_image.shape)
#             cv2.waitKey(3)
#         except Exception as e:
#             rospy.logerr(f"cv_bridge error: {e}")
#             return

#         # 推理输入处理
#         im = cv2.resize(cv_image, tuple(self.imgsz))  # 确保尺寸匹配
#         im_tensor = torch.from_numpy(im).to(self.device)
#         im_tensor = im_tensor.permute(2, 0, 1).float() / 255.0
#         if im_tensor.ndim == 3:
#             im_tensor = im_tensor.unsqueeze(0)

#         with torch.no_grad():
#             pred = self.model(im_tensor)
#             pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

#         # 可视化
#         im0 = cv_image.copy()
#         annotator = Annotator(im0, line_width=2, example=str(self.names))

#         for det in pred[0]:
#             if len(det):
#                 det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
#                 for *xyxy, conf, cls in det:
#                     label = f'{self.names[int(cls)]} {conf:.2f}'
#                     annotator.box_label(xyxy, label, color=colors(int(cls), True))

#         result_img = annotator.result()
#         try:
#             ros_img = self.bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
#             self.pub.publish(ros_img)
#         except Exception as e:
#             rospy.logerr(f"cv_bridge publish error: {e}")

# if __name__ == '__main__':
    
#     YOLOv9ROSNode()

