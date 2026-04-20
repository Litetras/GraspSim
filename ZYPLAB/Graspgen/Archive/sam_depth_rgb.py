from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"
open_stage(usd_path)

world = World()
franka: Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) 

controller = PickPlaceController(
    name="pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
    end_effector_initial_height=0.3,  # 机械臂末端执行器初始安全高度
    events_dt=[0.008, 0.005, 1, 0.01, 0.05, 0.05, 0.0025, 1, 0.008, 0.08], 
)

# 初始化相机
from omni.isaac.sensor import Camera
camera_path = "/World/Camera"
camera_width, camera_height = 1280, 720
camera = Camera(prim_path=camera_path, resolution=(camera_width, camera_height))
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
camera.add_rgb_to_frame()

# 仿真预热，让物体稳定
world.reset()
for i in range(100):
    world.step()
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

# 修复后逻辑：手动选点 → SAM3纯点提示分割物体 → SAM3文字提示分割刀具
import sys
sys.path.append(r'/home/zyp/GraspGen')
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from PIL import Image
from scipy.ndimage import zoom  
import os  # 新增：用于路径处理

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ===================== 核心配置（重点修改） =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False
plt.ion()  
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 导出文件保存目录（自定义）
SAVE_DIR = "/home/zyp/Desktop/grasp_export"  # 统一导出目录
os.makedirs(SAVE_DIR, exist_ok=True)  # 确保目录存在

# 2. 导出文件命名规范
RGB_FILE = "camera_rgb.png"               # 原始RGB图像
DEPTH_FILE_PNG = "camera_depth_16bit.png" # 深度图（16位PNG，单位：毫米）
DEPTH_FILE_NPY = "camera_depth_m.npy"     # 深度图（npy格式，单位：米，原始数据）
OBJECT_MASK_FILE = "object_mask.png"      # 物体分割Mask
TOOL_MASK_FILE = "tool_mask.png"          # 刀具分割Mask（替换原手柄）
TOOL_RGBA_FILE = "tool_transparent.png"   # 透明背景刀具图像
FINAL_MASK_FILE = "final_object_tool_mask.png"  # 物体+刀具合并Mask

# ===================== SAM3 初始化 =====================
print(f"加载SAM3模型 (设备: {device})...")
sam3_checkpoint = "/home/zyp/sam3/zypmodel/sam3/sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint)
sam3_processor = Sam3Processor(sam3_model)

# ===================== 导出原始RGB/Depth（核心新增） =====================
# 获取原始相机数据
rgb_data = camera.get_rgb()  # (H,W,3) uint8
depth_data = camera.get_depth()  # (H,W) float32，单位：米

# 导出RGB（PNG格式，无损）
rgb_save_path = os.path.join(SAVE_DIR, RGB_FILE)
cv2.imwrite(rgb_save_path, cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))  # OpenCV默认BGR
print(f"✅ 原始RGB已导出至: {rgb_save_path}")

# 导出Depth（两种格式：16位PNG（毫米） + NPY（米））
# 1) 16位PNG：米→毫米，转换为uint16（避免精度丢失）
depth_mm = (depth_data * 1000).astype(np.uint16)  # 米转毫米
depth_png_path = os.path.join(SAVE_DIR, DEPTH_FILE_PNG)
cv2.imwrite(depth_png_path, depth_mm)
print(f"✅ 深度图(16位PNG, 毫米)已导出至: {depth_png_path}")

# 2) NPY格式：保留原始米单位，方便后续计算
depth_npy_path = os.path.join(SAVE_DIR, DEPTH_FILE_NPY)
np.save(depth_npy_path, depth_data)
print(f"✅ 深度图(NPY, 米)已导出至: {depth_npy_path}")

print("深度图信息 - 形状:", depth_data.shape, "数值范围(米):", 
      np.min(depth_data), "~", np.max(depth_data))

"""
按ESC: 取消选点重新选择
"""
# 初始化选点存储
input_points = []  # 选点坐标 (x, y)
input_labels = []  # 1=前景(左键), 0=背景(右键)
selected_points = []  # 用于可视化选点

# 创建选点窗口
fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(rgb_data)
ax.set_title("Select target area (Left=foreground, Right=background), Press Enter to confirm")
fig.canvas.draw()

# 鼠标点击事件处理函数
def on_click(event):
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if event.button == MouseButton.LEFT:
        input_points.append([x, y])
        input_labels.append(1)
        selected_points.append(ax.plot(x, y, 'ro', markersize=8)[0])  
    elif event.button == MouseButton.RIGHT:
        input_points.append([x, y])
        input_labels.append(0)
        selected_points.append(ax.plot(x, y, 'bo', markersize=8)[0])  
    fig.canvas.draw()

# 键盘事件处理函数
def on_key(event):
    if event.key in ['enter', ' ']:  
        plt.close(fig)
    elif event.key == 'escape':  # 清空重选
        input_points.clear()
        input_labels.clear()
        for p in selected_points:
            p.remove()
        selected_points.clear()
        fig.canvas.draw()

# 绑定事件
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show(block=True)  # 阻塞直到窗口关闭

# 转换选点格式（适配SAM3输入）
input_points_np = np.array(input_points, dtype=np.float32)
input_labels_np = np.array(input_labels, dtype=np.int32)

# 置信度阈值
CONFIDENCE_THRESHOLD = 0.2

# 转换图像格式
rgb_image = Image.fromarray(rgb_data.astype(np.uint8))
# SAM3初始化图像
inference_state_obj = sam3_processor.set_image(rgb_image)

# 方案B：SAM3文字提示+选点过滤分割物体
print("使用SAM3文字提示+选点过滤分割物体...")
output_obj = sam3_processor.set_text_prompt(
    state=inference_state_obj,
    prompt="object"  # 通用物体提示
)
# 基于手动选点过滤mask（只保留包含选点的mask）
obj_masks = output_obj["masks"].cpu().numpy()
obj_scores_np = output_obj["scores"].cpu().numpy()
filtered_masks = []
filtered_scores = []

for mask, score in zip(obj_masks, obj_scores_np):
    if score < CONFIDENCE_THRESHOLD:
        continue

    if len(mask.shape) == 3:
        mask = mask[0]
    # 缩放mask到原图尺寸
    if mask.shape != rgb_data.shape[:2]:
        scale_y = rgb_data.shape[0] / mask.shape[0]
        scale_x = rgb_data.shape[1] / mask.shape[1]
        mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
    # 检查前景点是否在mask内
    foreground_points = input_points_np[input_labels_np == 1].astype(int)
    has_foreground = False
    for (x, y) in foreground_points:
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0.5:
            has_foreground = True
            break
    if has_foreground:
        filtered_masks.append(mask)
        filtered_scores.append(score)
# 替换为过滤后的mask和分数
obj_masks = np.array(filtered_masks)
obj_scores_np = np.array(filtered_scores)
output_obj["masks"] = torch.from_numpy(obj_masks).to(device)
output_obj["scores"] = torch.from_numpy(obj_scores_np).to(device)

# 处理物体mask
obj_masks = output_obj["masks"].cpu().numpy()
obj_scores_np = output_obj["scores"].cpu().numpy()

# 合并所有有效物体mask
object_mask = np.zeros(rgb_data.shape[:2], dtype=np.uint8)
valid_obj_count = 0
if len(obj_masks) > 0:
    for idx, (mask, score) in enumerate(zip(obj_masks, obj_scores_np)):
        if score < CONFIDENCE_THRESHOLD:
            print(f"跳过低置信度物体mask {idx}，置信度: {score:.3f} < {CONFIDENCE_THRESHOLD}")
            continue
        
        # 处理mask维度和尺寸
        if len(mask.shape) == 3:
            mask = mask[0]  # 去除batch维度
        if mask.shape != rgb_data.shape[:2]:
            scale_y = rgb_data.shape[0] / mask.shape[0]
            scale_x = rgb_data.shape[1] / mask.shape[1]
            mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5  # 缩放后二值化
        
        # 合并到总mask
        single_obj_mask = (mask > 0.5).astype(np.uint8)
        object_mask = np.bitwise_or(object_mask, single_obj_mask)
        valid_obj_count += 1
        print(f"合并物体mask {idx}，置信度: {score:.3f}")
    print(f"SAM3物体分割完成：共检测到 {len(obj_masks)} 个实例，合并 {valid_obj_count} 个有效物体mask")
    
    # 导出物体Mask（核心新增）
    object_mask_path = os.path.join(SAVE_DIR, OBJECT_MASK_FILE)
    cv2.imwrite(object_mask_path, object_mask * 255)  # 0→0, 1→255（可视化）
    print(f"✅ 物体分割Mask已导出至: {object_mask_path}")
else:
    raise ValueError("SAM3未检测到任何目标物体！请调整选点或文字提示")

# 提取物体RGB图像
object_mask_3ch = np.repeat(object_mask[:, :, np.newaxis], 3, axis=2)
object_rgb = np.where(object_mask_3ch == 1, rgb_data, 0)

# 可视化物体分割结果
plt.figure("SAM3 Object Segmentation (Points Filtered)", figsize=(15, 7))
plt.subplot(131)
plt.imshow(rgb_data)
# 叠加选点可视化
for (x, y), label in zip(input_points, input_labels):
    color = 'red' if label == 1 else 'blue'
    plt.scatter(x, y, c=color, s=80, edgecolors='white', linewidth=2)
plt.title("Original RGB + Manual Points")
plt.subplot(132)
plt.imshow(object_mask, cmap='gray')
plt.title(f"Object Mask (SAM3, {valid_obj_count} instances)")
plt.subplot(133)
plt.imshow(object_rgb)
plt.title("Extracted Object RGB")
plt.suptitle("Press Enter to continue to tool segmentation", fontsize=12)
plt.draw()
plt.waitforbuttonpress()
plt.close()

# ===================== SAM3文字分割刀具（核心修改：替换手柄为刀具） =====================
print("开始执行SAM3文字提示分割所有刀具...")
TOOL_PROMPT = "tool"  # 可根据实际场景调整为 "cutter" / "knife" / "刀具"

# 处理刀具分割
object_rgb_image = Image.fromarray(object_rgb.astype(np.uint8))
inference_state_tool = sam3_processor.set_image(object_rgb_image)
output_tool = sam3_processor.set_text_prompt(state=inference_state_tool, prompt=TOOL_PROMPT)

# 处理刀具mask
tool_masks = output_tool["masks"].cpu().numpy()
tool_scores_np = output_tool["scores"].cpu().numpy()

# 合并所有有效刀具mask
tool_mask = np.zeros(object_rgb.shape[:2], dtype=np.uint8)
valid_tool_count = 0
if len(tool_masks) > 0:
    for idx, (mask, score) in enumerate(zip(tool_masks, tool_scores_np)):
        if score < CONFIDENCE_THRESHOLD:
            print(f"跳过低置信度刀具mask {idx}，置信度: {score:.3f} < {CONFIDENCE_THRESHOLD}")
            continue
        
        # 处理mask维度和尺寸
        if len(mask.shape) == 3:
            mask = mask[0]
        if mask.shape != object_rgb.shape[:2]:
            scale_y = object_rgb.shape[0] / mask.shape[0]
            scale_x = object_rgb.shape[1] / mask.shape[1]
            mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
        
        # 合并mask
        single_tool_mask = (mask > 0.5).astype(np.uint8)
        tool_mask = np.bitwise_or(tool_mask, single_tool_mask)
        valid_tool_count += 1
        print(f"合并刀具mask {idx}，置信度: {score:.3f}")
    print(f"SAM3刀具分割完成：共检测到 {len(tool_masks)} 个实例，合并 {valid_tool_count} 个有效刀具mask")
    
    # 导出透明背景的刀具图片（核心新增）
    tool_rgb = np.where(np.repeat(tool_mask[:, :, np.newaxis], 3, axis=2) == 1, object_rgb, 0)
    # 创建RGBA刀具图像（Alpha通道控制透明度）
    tool_rgba = np.dstack((tool_rgb, tool_mask * 255))  # Alpha: 0=透明, 255=不透明
    tool_image_pil = Image.fromarray(tool_rgba.astype(np.uint8))
    
    # 保存透明背景刀具图片
    tool_rgba_path = os.path.join(SAVE_DIR, TOOL_RGBA_FILE)
    tool_image_pil.save(tool_rgba_path, format='PNG')
    print(f"✅ 透明背景刀具图像已导出至: {tool_rgba_path}")
    
    # 导出刀具Mask（核心新增）
    tool_mask_path = os.path.join(SAVE_DIR, TOOL_MASK_FILE)
    cv2.imwrite(tool_mask_path, tool_mask * 255)
    print(f"✅ 刀具分割Mask已导出至: {tool_mask_path}")
else:
    print("警告：SAM3未检测到任何刀具！")

# 可视化刀具分割结果
plt.figure("SAM3 Tool Segmentation Result", figsize=(15, 7))
plt.subplot(131)
plt.imshow(rgb_data)
plt.title("Original RGB")
plt.subplot(132)
plt.imshow(object_rgb)
plt.title("Extracted Object RGB (Input to SAM3)")
plt.subplot(133)
plt.imshow(tool_mask, cmap='gray')
plt.title(f"All Tools Mask (SAM3, {valid_tool_count} instances)")
plt.suptitle("Press Enter to continue to grasp planning", fontsize=12)
plt.draw()
plt.waitforbuttonpress()
plt.close()

# 合并物体+刀具mask
final_mask = np.bitwise_and(object_mask, tool_mask)
print(f"最终mask形状: {final_mask.shape}, 有效像素数: {np.sum(final_mask)}")

# 导出最终合并Mask（核心新增）
final_mask_path = os.path.join(SAVE_DIR, FINAL_MASK_FILE)
cv2.imwrite(final_mask_path, final_mask * 255)
print(f"✅ 物体+刀具合并Mask已导出至: {final_mask_path}")

# 可视化最终mask
plt.figure("Final Mask (Object + All Tools)", figsize=(12, 7))
plt.imshow(final_mask, cmap='gray')
plt.title("Final Mask (Object + All Tools)")
plt.suptitle("Press Enter to start grasping", fontsize=12)
plt.draw()
plt.waitforbuttonpress()
plt.close()
print("open meshcat-server")

# ===================== 相机内参处理 =====================
intrinsic = camera.get_intrinsics_matrix()
fx = float(intrinsic[0, 0])
fy = float(intrinsic[1, 1])
cx = float(intrinsic[0, 2])
cy = float(intrinsic[1, 2])
intrinsic = [fx, fy, cx, cy]  # 适配graspnet输入格式
print("相机内参 fx, fy, cx, cy: ", intrinsic)

###########################################################################
# 原有抓取推理和坐标变换逻辑（保持不变）
###########################################################################
from demogen import demo_variable
grasp = demo_variable(rgb_data, depth_data, final_mask, intrinsic)

# 坐标变换工具函数
def get_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T

# 沿抓取方向前移0.1米的函数
def move_along_grasp_dir(htm: np.ndarray, distance: float = 0.1) -> np.ndarray:
    grasp_dir = htm[:3, 2]
    grasp_dir_unit = grasp_dir / np.linalg.norm(grasp_dir)
    new_t = htm[:3, 3] + grasp_dir_unit * distance
    new_htm = np.eye(4)
    new_htm[:3, :3] = htm[:3, :3]
    new_htm[:3, 3] = new_t
    return new_htm

from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat

# 1. 获取相机在世界坐标系中的位姿    
cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

# 2. 构造相机坐标系下的抓取位姿
T_cam_grasp = grasp.pose

# 3. 坐标变换：相机坐标系→世界坐标系
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# 前移抓取位姿
T_world_grasp = move_along_grasp_dir(T_world_grasp, distance=0.1)
print(f"抓取位姿已沿抓取方向前移0.1米")

grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])

# 确定抓取点和放置点
banana_position, banana_orientation = grasp_pos, grasp_quat
goal_position = banana_position.copy()
goal_position[0] += 0    
goal_position[2] += 0.06  

print("抓取点xyz: ", banana_position)
print("放置点xyz: ", goal_position)

###########################################################################
# 机械臂抓取控制（保持不变）
###########################################################################
for i in range(1000000):
    current_joint_positions = franka.get_joint_positions()
    actions = controller.forward(
        picking_position=banana_position,
        placing_position=goal_position,
        current_joint_positions=current_joint_positions,
        end_effector_orientation=banana_orientation
    )
    franka.apply_action(actions)
    world.step(render=True) 

# 关闭仿真
simulation_app.close()
