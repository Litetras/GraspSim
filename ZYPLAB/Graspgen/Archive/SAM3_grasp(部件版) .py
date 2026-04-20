from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka

# ================= 修复点 =================
# 适配最新 Isaac Sim 的导入路径
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
# ==========================================

import numpy as np

usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"
open_stage(usd_path)

world = World()
franka: Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) 

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

# ================= 修复点 =================
# 使用新的类名实例化 IK 求解器
ik_solver = KinematicsSolver(robot_articulation=franka)
# ==========================================

# ===================== SAM3 初始化及图像处理 =====================
import sys
sys.path.append(r'/home/zyp/GraspGen')
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom  

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['toolbar'] = 'None'  
plt.ion()  
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"加载SAM3模型 (设备: {device})...")
sam3_checkpoint = "/home/zyp/sam3/zypmodel/sam3/sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint)
sam3_processor = Sam3Processor(sam3_model)

rgb_data = camera.get_rgb()
depth_data = camera.get_depth()  
print("深度图形状:", depth_data.shape, "数值范围:", np.min(depth_data), "~", np.max(depth_data))
# ===================== 全新逻辑：双 Mask 文字分割 =====================
PROMPT_OBJ = "hammer"    # 用于提取整个物体的轮廓（防碰撞）
PROMPT_PART = "handle"   # 用于定位你要抓取的目标部件

print(f"开始执行SAM3文字提示分割...")
rgb_image = Image.fromarray(rgb_data.astype(np.uint8))
inference_state_obj = sam3_processor.set_image(rgb_image)

# 1. 提取完整物体 Mask
output_obj = sam3_processor.set_text_prompt(state=inference_state_obj, prompt=PROMPT_OBJ)
obj_masks = output_obj["masks"].cpu().numpy()
obj_scores = output_obj["scores"].cpu().numpy()

if len(obj_masks) == 0:
    raise ValueError(f"❌ SAM3未检测到任何 '{PROMPT_OBJ}'！")
best_obj_mask = obj_masks[np.argmax(obj_scores)]
if len(best_obj_mask.shape) == 3: best_obj_mask = best_obj_mask[0]  
if best_obj_mask.shape != rgb_data.shape[:2]:
    scale_y, scale_x = rgb_data.shape[0]/best_obj_mask.shape[0], rgb_data.shape[1]/best_obj_mask.shape[1]
    best_obj_mask = zoom(best_obj_mask, (scale_y, scale_x), order=0) > 0.5
final_obj_mask = (best_obj_mask > 0.5).astype(np.uint8)

# 2. 提取目标部件 Mask (例如: handle)
output_part = sam3_processor.set_text_prompt(state=inference_state_obj, prompt=PROMPT_PART)
part_masks = output_part["masks"].cpu().numpy()
part_scores = output_part["scores"].cpu().numpy()

if len(part_masks) == 0:
    print(f"⚠️ 警告: SAM3未检测到 '{PROMPT_PART}'，将退化为全物体抓取。")
    final_part_mask = None
else:
    best_part_mask = part_masks[np.argmax(part_scores)]
    if len(best_part_mask.shape) == 3: best_part_mask = best_part_mask[0]
    if best_part_mask.shape != rgb_data.shape[:2]:
        scale_y, scale_x = rgb_data.shape[0]/best_part_mask.shape[0], rgb_data.shape[1]/best_part_mask.shape[1]
        best_part_mask = zoom(best_part_mask, (scale_y, scale_x), order=0) > 0.5
    final_part_mask = (best_part_mask > 0.5).astype(np.uint8)
    print(f"✅ 成功找到目标部件 '{PROMPT_PART}'，置信度为: {np.max(part_scores):.3f}")

print("open meshcat-server")

# ===================== 相机内参处理 =====================
intrinsic = camera.get_intrinsics_matrix()
fx, fy, cx, cy = float(intrinsic[0, 0]), float(intrinsic[1, 1]), float(intrinsic[0, 2]), float(intrinsic[1, 2])
intrinsic = [fx, fy, cx, cy]  

###########################################################################
# 抓取推理和坐标变换逻辑
###########################################################################
from demogen_part import demo_variable

target_instruction = "down"  

grasp = demo_variable(
    rgb_data=rgb_data, 
    depth_data=depth_data, 
    mask=final_obj_mask,         # 传入完整物体 Mask
    intrinsic=intrinsic,
    text=target_instruction,
    part_mask=final_part_mask    # <==== 新增参数：传入部件 Mask
)

def get_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T

def move_along_grasp_dir(htm: np.ndarray, distance: float = 0.1) -> np.ndarray:
    grasp_dir = htm[:3, 2]
    grasp_dir_unit = grasp_dir / np.linalg.norm(grasp_dir)
    new_t = htm[:3, 3] + grasp_dir_unit * distance
    new_htm = np.eye(4)
    new_htm[:3, :3] = htm[:3, :3]
    new_htm[:3, 3] = new_t
    return new_htm

cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

T_cam_grasp = grasp.pose
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# 最终的抓取目标位姿
T_world_grasp = move_along_grasp_dir(T_world_grasp, distance=0.1)
grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])

print("抓取点xyz: ", grasp_pos)

###########################################################################
# 全新机械臂运动控制逻辑 (基于 IK 的平滑插值)
###########################################################################

# 定义平滑移动函数
def move_to_pose(target_pos, target_quat, step_count=150):
    action, success = ik_solver.compute_inverse_kinematics(
        target_position=target_pos,
        target_orientation=target_quat
    )
    
    if success:
        current_joints = franka.get_joint_positions() # 长度 9 (7臂 + 2手)
        target_arm_joints = action.joint_positions    # 长度 7 (仅臂)
        
        # 补齐 9 自由度：手臂用 IK 算出来的目标位，夹爪保持当前不动
        target_joints = np.copy(current_joints)
        target_joints[:7] = target_arm_joints 
        
        # 关节空间线性插值
        for i in range(step_count):
            alpha = i / step_count
            interp_joints = current_joints * (1 - alpha) + target_joints * alpha
            franka.apply_action(ArticulationAction(joint_positions=interp_joints))
            world.step(render=True)
    else:
        print(f"❌ IK 求解失败，目标位姿不可达或处于奇异点！跳过此动作。")

# 1. 计算预抓取点：沿着抓取方向 Z 轴反向退后 10cm
grasp_dir = T_world_grasp[:3, 2]
pre_grasp_distance = -0.1  
pre_grasp_pos = grasp_pos + grasp_dir * pre_grasp_distance

print("\n>>> 步骤 1: 机械臂斜向移动到预抓取姿态...")
move_to_pose(pre_grasp_pos, grasp_quat, step_count=200)

print(">>> 步骤 2: 顺着抓取角度，直线插入抓取点...")
move_to_pose(grasp_pos, grasp_quat, step_count=100)

print(">>> 步骤 3: 闭合夹爪...")
franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
# 等待夹爪完全闭合并稳定
for _ in range(80): 
    world.step(render=True)

print(">>> 步骤 4: 保持抓取姿态，向上提起...")
# 计算垂直向上提起的点
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2  
move_to_pose(lift_pos, grasp_quat, step_count=150)

# 留出一点时间观察最终状态
for _ in range(100):
    world.step(render=True)

# 关闭仿真
simulation_app.close()