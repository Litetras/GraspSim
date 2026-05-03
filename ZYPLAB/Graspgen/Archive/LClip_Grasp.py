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

# ===================== 全新逻辑：直接用文字分割置信度最高的刀具 =====================
PROMPT = "knife"
print(f"开始执行SAM3文字提示分割，寻找 '{PROMPT}'...")

rgb_image = Image.fromarray(rgb_data.astype(np.uint8))
inference_state_obj = sam3_processor.set_image(rgb_image)

output_obj = sam3_processor.set_text_prompt(
    state=inference_state_obj,
    prompt=PROMPT
)

masks = output_obj["masks"].cpu().numpy()
scores = output_obj["scores"].cpu().numpy()

if len(masks) == 0:
    raise ValueError(f"❌ SAM3未检测到任何 '{PROMPT}'！请检查相机视野。")

best_idx = np.argmax(scores)
best_mask = masks[best_idx]
best_score = scores[best_idx]

print(f"✅ 成功找到置信度最高的 '{PROMPT}'，置信度为: {best_score:.3f}")

if len(best_mask.shape) == 3:
    best_mask = best_mask[0]  
if best_mask.shape != rgb_data.shape[:2]:
    scale_y = rgb_data.shape[0] / best_mask.shape[0]
    scale_x = rgb_data.shape[1] / best_mask.shape[1]
    best_mask = zoom(best_mask, (scale_y, scale_x), order=0) > 0.5

final_mask = (best_mask > 0.5).astype(np.uint8)

print("open meshcat-server")

# ===================== 相机内参处理 =====================
intrinsic = camera.get_intrinsics_matrix()
fx = float(intrinsic[0, 0])
fy = float(intrinsic[1, 1])
cx = float(intrinsic[0, 2])
cy = float(intrinsic[1, 2])
intrinsic = [fx, fy, cx, cy]  
print("相机内参 fx, fy, cx, cy: ", intrinsic)

###########################################################################
# 抓取推理和坐标变换逻辑

from demogen_Clip import demo_variable
##########################################################################################
target_instruction = "up handle"
print(f"✅ 使用固定抓取指令: '{target_instruction}'\n")
###############################################################################################################
grasp = demo_variable(
    rgb_data=rgb_data, 
    depth_data=depth_data, 
    mask=final_mask, 
    intrinsic=intrinsic,
    text=target_instruction  
)

# ===================== 运动规划与执行 (已更新为下方更简洁的逻辑) =====================
def get_T(t, r):
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T

cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

# 从上文的 demo_variable 获取抓取位姿
T_cam_grasp = grasp.pose

# 坐标系修正：转换到 Isaac Franka 坐标系
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ \
                T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# 抓取控制逻辑
def move_to_pose(target_pos, target_quat, steps=150):
    action, success = ik_solver.compute_inverse_kinematics(target_pos, target_quat)
    if success:
        curr = franka.get_joint_positions()
        targ = np.copy(curr)
        targ[:7] = action.joint_positions
        for i in range(steps):
            alpha = i / steps
            franka.apply_action(ArticulationAction(joint_positions=curr*(1-alpha) + targ*alpha))
            world.step(render=True)
    else: 
        print("IK Failed")

grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])
grasp_dir = T_world_grasp[:3, 2]

print("抓取点xyz: ", grasp_pos)

print(">>> 步骤: 移动到预抓取点...")
move_to_pose(grasp_pos - grasp_dir * 0.1, grasp_quat, steps=180)

print(">>> 步骤: 插入并抓取...")
move_to_pose(grasp_pos + grasp_dir * 0.11, grasp_quat, steps=80) # 增加插入深度到 12cm
franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
for _ in range(60): 
    world.step(render=True)

print(">>> 步骤: 提起物体...")
move_to_pose(grasp_pos + np.array([0, 0, 0.2]), grasp_quat, steps=120)

for _ in range(120): 
    world.step(render=True)

# 关闭仿真
simulation_app.close()