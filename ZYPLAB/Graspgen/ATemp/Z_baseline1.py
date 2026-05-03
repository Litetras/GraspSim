import os
import sys
import numpy as np
import time
import subprocess
import cv2

# ===================== Isaac Sim 核心初始化 =====================
from isaacsim import SimulationApp
# 注意：SimulationApp 必须在所有其他 UI 相关的库之前启动
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.sensor import Camera



# ===================== 路径配置 =====================
CONTACT_PYTHON = "/home/zyp/anaconda3/envs/contact/bin/python"

# 👇 把下面这行改成新写的 PyTorch Worker 的路径
WORKER_SCRIPT = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Z_cgn_worker_baseline1.py" 

TEMP_IN = "/tmp/cgn_in.npz"
TEMP_OUT = "/tmp/cgn_out.npz"

# 场景加载
usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/cam5.usd"
open_stage(usd_path)

world = World()
franka: Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) 
camera = Camera(prim_path="/World/Camera", resolution=(1280, 720))
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
camera.add_rgb_to_frame()

world.reset()
for _ in range(60): world.step()
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
ik_solver = KinematicsSolver(robot_articulation=franka)

# ===================== SAM3 视觉处理 =====================
import torch
from PIL import Image
from scipy.ndimage import zoom
sys.path.append(r'/home/zyp/GraspGen')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print(">>> 正在加载 SAM3 分割模型...")
sam3_model = build_sam3_image_model(checkpoint_path="/home/zyp/sam3/zypmodel/sam3/sam3.pt")
sam3_processor = Sam3Processor(sam3_model)

rgb_data = camera.get_rgb()[:, :, :3]  # <--- 关键修复：去掉 Alpha 通道，只保留 RGB
depth_data = camera.get_depth()
rgb_image = Image.fromarray(rgb_data.astype(np.uint8))

print(">>> 正在进行语义分割 (knife)...")
inference_state = sam3_processor.set_image(rgb_image)
output_obj = sam3_processor.set_text_prompt(state=inference_state, prompt="knife")

masks = output_obj["masks"].cpu().numpy()
scores = output_obj["scores"].cpu().numpy()
if len(masks) == 0: raise ValueError("SAM3 未找到目标！")

best_mask = masks[np.argmax(scores)]
if len(best_mask.shape) == 3: best_mask = best_mask[0]
if best_mask.shape != rgb_data.shape[:2]:
    best_mask = zoom(best_mask, (rgb_data.shape[0]/best_mask.shape[0], rgb_data.shape[1]/best_mask.shape[1]), order=0) > 0.5
final_mask = (best_mask > 0.5).astype(np.uint8)

# ===================== 跨环境调用 CGN =====================
intrinsic = camera.get_intrinsics_matrix()
cam_K = intrinsic[:3, :3].astype(np.float32)

# ----------------- 新增：清洗深度图 -----------------
# 剔除 Isaac Sim 中的无限远 (Inf) 和 NaN，将超出 5 米的深度截断
depth_data = np.nan_to_num(depth_data, posinf=0.0, neginf=0.0)
depth_data = np.clip(depth_data, 0.0, 5.0) 

# ----------------------------------------------------

# ===================== Z_baseline1-ContactGrasp.py =====================

print(">>> 保存数据并调用 Contact-GraspNet 后端...")
np.savez(TEMP_IN, depth=depth_data, K=cam_K, segmap=final_mask, rgb=rgb_data)

# 🌟 终极修复 1：过河拆桥，强制释放 SAM3 占用的所有显存！
print(">>> 正在释放 SAM3 占用的 PyTorch 显存，为抓取模型腾出空间...")
try:
    del sam3_model
    del sam3_processor
    if 'inference_state' in locals(): del inference_state
    if 'output_obj' in locals(): del output_obj
    import gc
    gc.collect()
    torch.cuda.empty_cache()  # 核心：清空 PyTorch 的显存缓存池
except Exception as e:
    pass
# -------------------------------------------------------------------------

import os
my_env = os.environ.copy()
for key in ["PYTHONPATH", "LD_LIBRARY_PATH"]:
    if key in my_env:
        del my_env[key]
my_env["PYTHONUNBUFFERED"] = "1"

if os.path.exists(TEMP_OUT):
    os.remove(TEMP_OUT)

print(">>> [主程序] 已跨进程启动 Contact-GraspNet...")
# 移除会导致冲突的 PYTHONPATH，让子进程去它自己的 conda 环境里找库
if "PYTHONPATH" in my_env:
    del my_env["PYTHONPATH"]
# --------------------------------------------------------

# 🌟 新增：跨进程调用前，先删掉上一次可能残留的输出文件，防止读到“幽灵数据”！
if os.path.exists(TEMP_OUT):
    os.remove(TEMP_OUT)

# 调用 subprocess 时传入 env=my_env
result = subprocess.run(
    [CONTACT_PYTHON, WORKER_SCRIPT, "--in_data", TEMP_IN, "--out_data", TEMP_OUT],
    env=my_env  # <--- 关键：使用清理后的环境
)



if result.returncode != 0 or not os.path.exists(TEMP_OUT):
    print("❌ CGN 后端运行失败")
    simulation_app.close()
    sys.exit()

res_data = np.load(TEMP_OUT)
if not res_data['success']:
    print("❌ CGN 未能生成有效抓取")
    simulation_app.close()
    sys.exit()

T_cam_grasp = res_data['best_grasp']
print(f"✅ 获取到抓取位姿，得分: {res_data['score']:.4f}")




# ===================== 运动规划与执行 =====================
def get_T(t, r):
    T = np.eye(4); T[:3, :3] = r; T[:3, 3] = t; return T

cam_trans, cam_quat = SingleXFormPrim("/World/Camera").get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

# 坐标系修正：CGN -> Isaac Franka
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ \
                T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# 抓取控制逻辑
def move_to_pose(target_pos, target_quat, steps=150):
    action, success = ik_solver.compute_inverse_kinematics(target_pos, target_quat)
    if success:
        curr = franka.get_joint_positions()
        targ = np.copy(curr); targ[:7] = action.joint_positions
        for i in range(steps):
            alpha = i / steps
            franka.apply_action(ArticulationAction(joint_positions=curr*(1-alpha) + targ*alpha))
            world.step(render=True)
    else: print("IK Failed")

grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])
grasp_dir = T_world_grasp[:3, 2]

# 🚨 [新增] 终极调试大法：生成幽灵坐标 Marker
# 红色长方体代表夹爪的正方向 (X轴)。如果它没有对准你想抓的位置，说明矩阵算错了，不用怪 IK！
from omni.isaac.core.objects import VisualCuboid
visual_cube = VisualCuboid(
    prim_path="/World/GraspTargetMarker", 
    name="grasp_marker",
    position=grasp_pos,
    orientation=grasp_quat,
    scale=np.array([0.05, 0.01, 0.01]), 
    color=np.array([1.0, 0.0, 0.0])     
)
world.scene.add(visual_cube)
world.step(render=True)
print("🔍 调试 Marker 已生成，请在画面中检查红色的抓取位姿是否正确对准目标！")


print(">>> 步骤: 移动到预抓取点...")
move_to_pose(grasp_pos - grasp_dir * 0.1, grasp_quat, steps=180)

print(">>> 步骤: 插入并抓取...")
move_to_pose(grasp_pos + grasp_dir * 0.12, grasp_quat, steps=80)# 增加插入深度到 12cm
franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
for _ in range(60): world.step(render=True)

print(">>> 步骤: 提起物体...")
move_to_pose(grasp_pos + np.array([0, 0, 0.2]), grasp_quat, steps=120)

for _ in range(120): world.step(render=True)
simulation_app.close()