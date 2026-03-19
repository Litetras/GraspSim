from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

# 场景加载路径（请确认绝对路径正确性）
usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"
open_stage(usd_path)

# 初始化仿真世界和机械臂
world = World()
franka: Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) 
controller = PickPlaceController(
    name="pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
    end_effector_initial_height=0.3,#####决定了机械臂末端执行器（EEF）在抓取动作启动前的初始安全高度—— 控制器会先将末端执行器抬升到该高度
    
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
print("相机初始化完成")
print("获取相机内参: \n", camera.get_intrinsics_matrix())

# 仿真预热，让物体稳定
world.reset()
for i in range(100):
    world.step()
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

###########################################################################
# 核心替换：SAM → SAM2 分割模块（修复选点和窗口问题）
###########################################################################
import sys
sys.path.append(r'/home/zyp/GraspGen')

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ===================== SAM2 配置 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 关闭中文字体警告
plt.rcParams['axes.unicode_minus'] = False
plt.ion()  # 开启交互模式，避免阻塞

# SAM2模型路径（请确认路径正确性）
sam_checkpoint = "/home/zyp/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 1. 获取相机数据 =====================
rgb_data = camera.get_rgb()
depth_data = camera.get_depth()  # graspgen需要m单位深度图
print("深度图形状:", depth_data.shape, "数值范围:", np.min(depth_data), "~", np.max(depth_data))

# ===================== 2. 加载SAM2模型 =====================
print(f"加载SAM2模型 (设备: {device})...")
sam2_model = build_sam2(model_cfg, sam_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(rgb_data)  # 设置待分割图像

# ===================== 3. 交互式点选目标区域（修复核心问题） =====================
print("""
图像窗口操作说明:
1. 左键点击: 选择前景点（需要分割的目标物体）
2. 右键点击: 选择背景点（需要排除的区域，可选）
3. 按Enter/空格: 确认选点并执行分割
4. 按ESC: 取消选点重新选择
""")

# 初始化选点存储
input_points = []
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
    # 记录坐标和按键类型
    x, y = event.xdata, event.ydata
    if event.button == MouseButton.LEFT:
        input_points.append([x, y])
        input_labels.append(1)
        selected_points.append(ax.plot(x, y, 'ro', markersize=8)[0])  # 前景红点
    elif event.button == MouseButton.RIGHT:
        input_points.append([x, y])
        input_labels.append(0)
        selected_points.append(ax.plot(x, y, 'bo', markersize=8)[0])  # 背景蓝点
    fig.canvas.draw()

# 键盘事件处理函数
def on_key(event):
    if event.key in ['enter', ' ']:  # 确认选点
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

# 校验选点结果
if len(input_points) == 0:
    raise ValueError("未选择任何点！请重新运行并选择目标区域")

# 转换为SAM2所需格式
input_points = np.array(input_points, dtype=np.float32)
input_labels = np.array(input_labels, dtype=np.int32)
print(f"选点完成：前景点{np.sum(input_labels==1)}个，背景点{np.sum(input_labels==0)}个")

# ===================== 4. SAM2执行分割 =====================
print("执行SAM2分割...")
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,  # 生成多个mask选最优
)

# 选择置信度最高的mask并格式化
best_mask_idx = np.argmax(scores)
mask = masks[best_mask_idx].astype(np.uint8)  # True→1, False→0（适配graspnet）
print(f"最优Mask形状: {mask.shape}, 置信度: {scores[best_mask_idx]:.3f}")

# 可视化分割结果（修复窗口显示问题）
plt.figure("Segmentation Mask", figsize=(12, 7))
plt.subplot(121)
plt.imshow(rgb_data)
plt.title("Original RGB")
plt.subplot(122)
plt.imshow(mask, cmap='gray')
plt.title(f"Segmentation Mask (Confidence: {scores[best_mask_idx]:.3f})")
plt.suptitle("Press Enter to continue", fontsize=12)
plt.draw()  # 强制渲染窗口
plt.waitforbuttonpress()  # 等待按键
plt.close()

# ===================== 5. 相机内参处理 =====================
intrinsic = camera.get_intrinsics_matrix()
fx = float(intrinsic[0, 0])
fy = float(intrinsic[1, 1])
cx = float(intrinsic[0, 2])
cy = float(intrinsic[1, 2])
intrinsic = [fx, fy, cx, cy]  # 适配graspnet输入格式

###########################################################################
# 原有抓取推理和坐标变换逻辑（保持不变）
###########################################################################
# graspnet_baseline 推理
from demogen import demo_variable
grasp = demo_variable(rgb_data, depth_data, mask, intrinsic)

# 坐标变换工具函数
def get_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T

from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat

# 1. 获取相机在世界坐标系中的位姿    
cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

# 2. 构造相机坐标系下的抓取位姿
T_cam_grasp = grasp.pose

# import trimesh.transformations as tra; 
# new_grasp = T_cam_grasp @ tra.translation_matrix([0,0,-0.2])  # 抓取点下移10cm以适配机械臂,but实际效果wu


# 3. 坐标变换：相机坐标系→世界坐标系（适配可视化）
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# # 原代码（正方向绕Z轴转90度）
# get_T([0, 0, 0], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# # 修改后（反方向绕Z轴转90度）
# get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])

# 确定抓取点和放置点（微调z轴保证抓取稳定性）
banana_position, banana_orientation = grasp_pos, grasp_quat
banana_position[2] -= 0.0 # 降低抓取高度，避免碰撞
goal_position = banana_position.copy()
goal_position[0] += 0.3    # 放置点偏移
goal_position[2] += 0.05

print("抓取点xyz: ", banana_position)
print("放置点xyz: ", goal_position)

###########################################################################
# 机械臂抓取控制（保持不变）
###########################################################################
for i in range(1000000):
    # 获取当前机械臂关节位置
    current_joint_positions = franka.get_joint_positions()
    # 计算抓取动作
    actions = controller.forward(
        picking_position=banana_position,
        placing_position=goal_position,
        current_joint_positions=current_joint_positions,
        end_effector_orientation=banana_orientation
    )
    # 执行动作
    franka.apply_action(actions)
    world.step(render=True) 

# 关闭仿真
simulation_app.close()
