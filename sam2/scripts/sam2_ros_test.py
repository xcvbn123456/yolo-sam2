#!/home/super/anaconda3/envs/sam2/bin/python3
import torch
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import pickle
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 加载消息
with open('/home/super/catkin_ws/src/screw_ros/yolov9/onemsg.pkl', 'rb') as f:
    msg = pickle.load(f)

# 初始化 CvBridge
bridge = CvBridge()

# 将 ROS 图像消息转换为 OpenCV 图像
cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

# 获取图像尺寸
h, w, _ = rgb_image.shape

# 提示点（假设目标在图像中心）
input_point = np.array([[w // 2, h // 2]])  # 假设只有一个目标在中心
input_label = np.array([1])

# 加载 SAM2 模型
cfg = "sam2_hiera_t.yaml"
ckpt = "/home/super/catkin_ws/src/screw_ros/SAM222/checkpoints/sam2_hiera_tiny.pt"
device = "cuda:0"

model = build_sam2(cfg, ckpt, device=device)
predictor = SAM2ImagePredictor(model)
predictor.set_image(rgb_image)

# 进行分割
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# 选取最佳掩码
best_mask = masks[np.argmax(scores)].astype(np.uint8) * 255

# 将掩码转换为三通道图像
mask_rgb = cv2.merge([best_mask, best_mask, best_mask])

# 在原始图像上标记提示点
marked_image = cv_image.copy()
cv2.circle(marked_image, tuple(input_point[0]), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色圆点

# 将掩码叠加到原始图像上
alpha = 0.5  # 叠加透明度
overlayed_image = cv2.addWeighted(marked_image, 1, mask_rgb, alpha, 0)

# 显示结果
cv2.imshow("Marked Image", marked_image)
cv2.imshow("SAM Result", best_mask)
cv2.imshow("Overlayed Image", overlayed_image)
cv2.waitKey(20000)

# 将分割图像转换回 ROS 格式发布
ros_mask = bridge.cv2_to_imgmsg(mask_rgb, encoding="bgr8")
rospy.loginfo("Published SAM2 mask with shape: %s", best_mask.shape)