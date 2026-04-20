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
# print("相机初始化完成")
# print("获取相机内参: \n", camera.get_intrinsics_matrix())

# 仿真预热，让物体稳定
world.reset()
for i in range(100):
    world.step()
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

# 修复后逻辑：手动选点 → SAM3纯点提示分割物体 → SAM3文字提示分割手柄
import sys
sys.path.append(r'/home/zyp/GraspGen')
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from PIL import Image
from scipy.ndimage import zoom  

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ===================== 配置 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False
plt.ion()  
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== SAM3 初始化 =====================
print(f"加载SAM3模型 (设备: {device})...")
sam3_checkpoint = "/home/zyp/sam3/zypmodel/sam3/sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint)
sam3_processor = Sam3Processor(sam3_model)

rgb_data = camera.get_rgb()
depth_data = camera.get_depth()  # graspgen需要m单位深度图
print("深度图形状:", depth_data.shape, "数值范围:", np.min(depth_data), "~", np.max(depth_data))

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


# 转换选点格式（适配SAM3输入，需将像素坐标归一化/调整）
input_points_np = np.array(input_points, dtype=np.float32)
input_labels_np = np.array(input_labels, dtype=np.int32)

#
CONFIDENCE_THRESHOLD = 0.2######################################################################################################

# 转换图像格式
rgb_image = Image.fromarray(rgb_data.astype(np.uint8))
# SAM3初始化图像
inference_state_obj = sam3_processor.set_image(rgb_image)

# # 方案A：若SAM3有set_point_prompt方法（优先尝试）############################################################################################################
# try:
#     # 尝试调用点提示分割方法（适配SAM3的点提示接口）
#     output_obj = sam3_processor.set_point_prompt(
#         state=inference_state_obj,
#         points=input_points_np,
#         point_labels=input_labels_np
#     )
# except AttributeError:
#     # 方案B：若SAM3无点提示接口，改用文字提示+选点后处理（兼容方案）
print("!SAM3无set_point_prompt方法，改用文字提示+选点后处理...")
# 先用文字提示分割所有可能的物体
output_obj = sam3_processor.set_text_prompt(
    state=inference_state_obj,
    prompt="object"  # 通用物体提示
)
# 基于手动选点过滤mask（只保留包含选点的mask）#########################################################################################
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

# 处理物体mask（通用逻辑）
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
plt.suptitle("Press Enter to continue to handle segmentation", fontsize=12)
plt.draw()
plt.waitforbuttonpress()
plt.close()

# ===================== 4. SAM3文字分割手柄（原有逻辑保留） =====================
print("开始执行SAM3文字提示分割所有手柄...")
HANDLE_PROMPT = "handle"

# 处理手柄分割
object_rgb_image = Image.fromarray(object_rgb.astype(np.uint8))
inference_state_handle = sam3_processor.set_image(object_rgb_image)
output_handle = sam3_processor.set_text_prompt(state=inference_state_handle, prompt=HANDLE_PROMPT)

# 处理手柄mask
handle_masks = output_handle["masks"].cpu().numpy()
handle_scores_np = output_handle["scores"].cpu().numpy()

# 合并所有有效手柄mask
handle_mask = np.zeros(object_rgb.shape[:2], dtype=np.uint8)
valid_handle_count = 0
if len(handle_masks) > 0:
    for idx, (mask, score) in enumerate(zip(handle_masks, handle_scores_np)):
        if score < CONFIDENCE_THRESHOLD:
            print(f"跳过低置信度手柄mask {idx}，置信度: {score:.3f} < {CONFIDENCE_THRESHOLD}")
            continue
        
        # 处理mask维度和尺寸
        if len(mask.shape) == 3:
            mask = mask[0]
        if mask.shape != object_rgb.shape[:2]:
            scale_y = object_rgb.shape[0] / mask.shape[0]
            scale_x = object_rgb.shape[1] / mask.shape[1]
            mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
        
        # 合并mask
        single_handle_mask = (mask > 0.5).astype(np.uint8)
        handle_mask = np.bitwise_or(handle_mask, single_handle_mask)
        valid_handle_count += 1
        print(f"合并手柄mask {idx}，置信度: {score:.3f}")
    print(f"SAM3手柄分割完成：共检测到 {len(handle_masks)} 个实例，合并 {valid_handle_count} 个有效手柄mask")
else:
    print("警告：SAM3未检测到任何手柄！")

# 可视化手柄分割结果
plt.figure("SAM3 Handle Segmentation Result", figsize=(15, 7))
plt.subplot(131)
plt.imshow(rgb_data)
plt.title("Original RGB")
plt.subplot(132)
plt.imshow(object_rgb)
plt.title("Extracted Object RGB (Input to SAM3)")
plt.subplot(133)
plt.imshow(handle_mask, cmap='gray')
plt.title(f"All Handles Mask (SAM3, {valid_handle_count} instances)")
plt.suptitle("Press Enter to continue to grasp planning", fontsize=12)
plt.draw()
plt.waitforbuttonpress()
plt.close()

# 合并物体+手柄mask
final_mask = np.bitwise_and(object_mask, handle_mask)
print(f"最终mask形状: {final_mask.shape}, 有效像素数: {np.sum(final_mask)}")

# 可视化最终mask
plt.figure("Final Mask (Object + All Handles)", figsize=(12, 7))
plt.imshow(final_mask, cmap='gray')
plt.title("Final Mask (Object + All Handles)")
plt.suptitle("Press Enter to start grasping", fontsize=12)
plt.draw()
plt.waitforbuttonpress()
plt.close()
print("open meshcat-server")
print("open meshcat-server")

# ===================== 5. 相机内参处理 =====================
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
