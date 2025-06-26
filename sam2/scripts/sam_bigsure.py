import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, random_color=False, borders=True):
    # 显示掩码图像
    if random_color:
        # 随机颜色
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # 固定颜色（浅蓝色）
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    # 获取掩码的高度和宽度
    h, w = mask.shape[-2:]
    # 将掩码转换为uint8类型
    mask = mask.astype(np.uint8)
    # 将掩码图像调整为颜色
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        # 如果需要显示边界
        import cv2
        # 找到掩码的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 尝试平滑轮廓
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        # 在掩码图像上绘制轮廓
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    # 在坐标轴上显示掩码图像
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    # 显示正负点
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    # 显示正点（绿色星号）
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # 显示负点（红色星号）
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    # 显示边框
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    # 在坐标轴上添加矩形边框
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    # 显示掩码、分数、点和框
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

        

# 打开图像文件 'images/truck.jpg'
#image = Image.open('images/truck.jpg')
path='/home/super/catkin_ws/src/screw_ros/yolov9/datasets/detect holes 1.v2i.yolov9/test/screw/1120aa163145187344e0822588a9d4e3.jpg'
image = Image.open(path)#156,85
path_weiba=path.split('/')[-1]
import os
path_weiba=os.path.splitext(path_weiba)[0]
new_name = f"{path_weiba}.txt"
# pathlabels=os.path.join("/media/zys/gm7000/SAM2/pack/exp11/labels", new_name)
# label_path='/media/zys/gm7000/SAM2/pack/dataset/test/labels/image_589_png.rf.d087aa5ec7d840f52bb50c05bee94d61.txt'
#image = Image.open('/media/zys/gm7000/SAM2/pack/dataset/test/images/image1_22_png.rf.dec86c625a4bc2e4dd141f25644fccf5.jpg')#199,69
# 将图像转换为 RGB 格式，并将其转换为 numpy 数组
image = np.array(image.convert("RGB"))

# 创建一个新的图形窗口，并设置图形窗口的大小为 10x10 英寸
plt.figure(figsize=(10, 10))
# 在当前图形窗口中显示图像
plt.imshow(image)
# 打开坐标轴
plt.axis('on')
# 显示图形窗口
plt.show()


# 指定 SAM 2 模型的检查点路径和配置文件
sam2_checkpoint = "/home/super/catkin_ws/src/screw_ros/SAM222/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
# 加载 SAM 2 模型，建议在 CUDA 设备上运行以获得最佳效果
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
# 创建 SAM2ImagePredictor 实例以便与模型交互
predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)

input_point = np.array([[1950, 450]])#dexycb
input_label = np.array([1])

# 创建一个大小为 10x10 的图像显示窗口
plt.figure(figsize=(10, 10))
# 显示图像
plt.imshow(image)
# 在图像上标记输入的点，前景点用绿色星号表示
show_points(input_point, input_label, plt.gca())
# 显示坐标轴
plt.axis('on')
# 显示图像
plt.show()

print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

# 使用predict方法进行预测
masks, scores, logits = predictor.predict(
    point_coords=input_point,  # 输入点坐标
    point_labels=input_label,  # 输入点标签
    multimask_output=True,  # 多掩膜输出设置为True
)

# 根据scores排序
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

masks.shape  # (number_of_masks) x H x W

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True) #  # 显示mask在图像上的效果