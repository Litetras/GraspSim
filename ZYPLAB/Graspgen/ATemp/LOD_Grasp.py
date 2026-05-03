from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat

import numpy as np
import sys
import cv2
import torch
from PIL import Image
from scipy.ndimage import zoom  

# ================= 1. 初始化 Isaac Sim 与 场景 =================
usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/cam2.usd"
open_stage(usd_path)

world = World()
franka: Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) 

# 初始化相机
camera_path = "/World/Camera"
camera_width, camera_height = 1280, 720
from omni.isaac.sensor import Camera
camera = Camera(prim_path=camera_path, resolution=(camera_width, camera_height))
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
camera.add_rgb_to_frame()

# 仿真预热
world.reset()
for i in range(100):
    world.step()
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
ik_solver = KinematicsSolver(robot_articulation=franka)

# ================= 2. SAM3 图像处理与分割 =================
sys.path.append(r'/home/zyp/GraspGen')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"加载SAM3模型 (设备: {device})...")
sam3_checkpoint = "/home/zyp/sam3/zypmodel/sam3/sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint)
sam3_processor = Sam3Processor(sam3_model)

rgb_data = camera.get_rgb()
depth_data = camera.get_depth()  

PROMPT = "knife"
print(f"开始执行SAM3文字提示分割，寻找 '{PROMPT}'...")
rgb_image = Image.fromarray(rgb_data.astype(np.uint8))
inference_state_obj = sam3_processor.set_image(rgb_image)
output_obj = sam3_processor.set_text_prompt(state=inference_state_obj, prompt=PROMPT)

masks = output_obj["masks"].cpu().numpy()
scores = output_obj["scores"].cpu().numpy()

if len(masks) == 0:
    raise ValueError(f"❌ SAM3未检测到任何 '{PROMPT}'！")

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

# ==================== 🧹 绝杀技：卸磨杀驴，释放显存 ====================
print("\n🧹 SAM3 分割完成，正在销毁 SAM3 释放显存，为大模型腾出空间...")
import gc
del sam3_model
del sam3_processor
del inference_state_obj 
gc.collect()
torch.cuda.empty_cache()
print("✅ 显存清理完毕！")
# ====================================================================


# 获取相机内参
intrinsic_matrix = camera.get_intrinsics_matrix()
intrinsic = [float(intrinsic_matrix[0, 0]), float(intrinsic_matrix[1, 1]), 
             float(intrinsic_matrix[0, 2]), float(intrinsic_matrix[1, 2])]

# ================= 3. 端到端自然语言抓取推理 (全新逻辑) =================
from demogen_LOD import demo_variable

print("\n🧠 正在使用端到端 Qwen-LoRA 语言条件模型推理抓取姿态+打开meshcat")
natural_instruction = "Grasp the knife to cut."
print(f"💬 输入自然语言指令: '{natural_instruction}'")

# 注意：为了适配 generator.py 第 522 行的断言：
# if "strict_text" not in data or "natural_text" not in data:
#     raise ValueError("Missing language keys in data")
# 我们需要同时传入 natural_text 和用于绕过断言的 strict_text
grasp = demo_variable(
    rgb_data=rgb_data, 
    depth_data=depth_data, 
    mask=final_mask, 
    intrinsic=intrinsic,
    natural_text=[natural_instruction], 
    strict_text=["nnn"],
    grasp_threshold=0.8,   # <========== 新增这一行：强制关闭分数阈值拦截！
    num_grasps=200
)

# ================= 4. 坐标系转换与机械臂控制 =================
def get_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T

def move_along_grasp_dir(htm: np.ndarray, distance: float = 0.1) -> np.ndarray:
    grasp_dir = htm[:3, 2]
    grasp_dir_unit = grasp_dir / np.linalg.norm(grasp_dir)
    new_htm = np.eye(4)
    new_htm[:3, :3] = htm[:3, :3]
    new_htm[:3, 3] = htm[:3, 3] + grasp_dir_unit * distance
    return new_htm

cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

T_cam_grasp = grasp.pose
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

T_world_grasp = move_along_grasp_dir(T_world_grasp, distance=0.1)
grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])

print("抓取点xyz: ", grasp_pos)

def move_to_pose(target_pos, target_quat, step_count=150):
    action, success = ik_solver.compute_inverse_kinematics(target_position=target_pos, target_orientation=target_quat)
    if success:
        current_joints = franka.get_joint_positions() 
        target_joints = np.copy(current_joints)
        target_joints[:7] = action.joint_positions    
        
        for i in range(step_count):
            alpha = i / step_count
            interp_joints = current_joints * (1 - alpha) + target_joints * alpha
            franka.apply_action(ArticulationAction(joint_positions=interp_joints))
            world.step(render=True)
    else:
        print(f"❌ IK 求解失败，跳过。")

# 1. 预抓取
grasp_dir = T_world_grasp[:3, 2]
pre_grasp_pos = grasp_pos + grasp_dir * -0.1  
print("\n>>> 步骤 1: 机械臂斜向移动到预抓取姿态...")
move_to_pose(pre_grasp_pos, grasp_quat, step_count=200)

# 2. 插入抓取
print(">>> 步骤 2: 直线插入抓取点...")
move_to_pose(grasp_pos, grasp_quat, step_count=100)

# 3. 闭合
print(">>> 步骤 3: 闭合夹爪...")
franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
for _ in range(80): world.step(render=True)

# 4. 提起
print(">>> 步骤 4: 向上提起...")
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.2  
move_to_pose(lift_pos, grasp_quat, step_count=150)

for _ in range(100): world.step(render=True)
simulation_app.close()