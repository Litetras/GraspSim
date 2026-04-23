import os
import sys
import warnings
import logging
import argparse
import numpy as np
import subprocess
import cv2
from scipy.spatial.transform import Rotation as R

# ===================== 终极黑魔法：强行屏蔽底层 C++ 输出 =====================
class SuppressOutput:
    def __enter__(self):
        self.old_stdout = os.dup(sys.stdout.fileno())
        self.old_stderr = os.dup(sys.stderr.fileno())
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, sys.stdout.fileno())
        os.dup2(self.devnull, sys.stderr.fileno())

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.old_stdout, sys.stdout.fileno())
        os.dup2(self.old_stderr, sys.stderr.fileno())
        os.close(self.devnull)
        os.close(self.old_stdout)
        os.close(self.old_stderr)

# ===================== 屏蔽多余输出 =====================
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["OMNI_LOG_LEVEL"] = "error"
os.environ["CARB_LOG_LEVEL"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OV_KIT_ALLOW_ROOT"] = "1"

# ===================== 接收外部参数 =====================
parser = argparse.ArgumentParser()
parser.add_argument("--trial",  type=int, default=0)
parser.add_argument("--cam_id", type=int, default=1)
args = parser.parse_args()
trial_id = args.trial
cam_id   = args.cam_id

IMG_DIR = "/home/zyp/Desktop/eval_results"
os.makedirs(IMG_DIR, exist_ok=True)

# ===================== Isaac Sim 核心初始化 =====================
from isaacsim import SimulationApp

print(">>> 正在静默启动 Isaac Sim，请稍候...")
with SuppressOutput():
    simulation_app = SimulationApp({"headless": True})
    import carb
    carb.settings.get_settings().set_string("/log/level", "error")
    carb.settings.get_settings().set_bool("/log/outputStream", False)

for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import VisualCuboid

# ===================== 路径配置 =====================
CONTACT_PYTHON = "/home/zyp/anaconda3/envs/contact/bin/python"
WORKER_SCRIPT  = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Baseline1/cgn_worker_baseline1.py"
TEMP_IN  = "/tmp/cgn_in.npz"
TEMP_OUT = "/tmp/cgn_out.npz"

# ===================== 场景加载 =====================
usd_path = f"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/cam{cam_id}.usd"
print(f">>> [Trial {trial_id}] 加载场景: {usd_path}")
open_stage(usd_path)

world  = World()
franka: Franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
camera = Camera(prim_path="/World/Camera", resolution=(1280, 720))
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
camera.add_rgb_to_frame()

world.reset()
for _ in range(60):
    world.step(render=True)

pos, quat = SingleXFormPrim("/World/Camera").get_world_pose()
print(f"\n>>> [Trial {trial_id}] 📷 相机位置: {pos}")
print(f">>> 相机四元数 [w,x,y,z]: {quat}")

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
sam3_model     = build_sam3_image_model(checkpoint_path="/home/zyp/sam3/zypmodel/sam3/sam3.pt")
sam3_processor = Sam3Processor(sam3_model)

rgb_data   = camera.get_rgb()[:, :, :3]
depth_data = camera.get_depth()

rgb_image       = Image.fromarray(rgb_data.astype(np.uint8))
inference_state = sam3_processor.set_image(rgb_image)
output_obj      = sam3_processor.set_text_prompt(state=inference_state, prompt="knife")
masks  = output_obj["masks"].cpu().numpy()
scores = output_obj["scores"].cpu().numpy()

if len(masks) == 0:
    print("❌ SAM3 未找到目标！")
    simulation_app.close()
    os._exit(1)

best_mask = masks[np.argmax(scores)]
if len(best_mask.shape) == 3:
    best_mask = best_mask[0]
if best_mask.shape != rgb_data.shape[:2]:
    best_mask = zoom(best_mask,
                     (rgb_data.shape[0] / best_mask.shape[0],
                      rgb_data.shape[1] / best_mask.shape[1]), order=0) > 0.5
final_mask = (best_mask > 0.5).astype(np.uint8)

intrinsic = camera.get_intrinsics_matrix()
cam_K     = intrinsic[:3, :3].astype(np.float32)

depth_data = np.nan_to_num(depth_data, posinf=0.0, neginf=0.0)
depth_data = np.clip(depth_data, 0.0, 5.0)

np.savez(TEMP_IN, depth=depth_data, K=cam_K, segmap=final_mask, rgb=rgb_data)

# ===================== 释放 SAM3 显存 =====================
print(">>> 开始强制释放 SAM3 显存...")
for key in ['sam3_model', 'sam3_processor', 'inference_state', 'output_obj',
            'rgb_image', 'masks', 'scores', 'best_mask']:
    if key in locals():
        del locals()[key]

import gc; gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()

# ===================== 调用 CGN 后端 =====================
my_env = os.environ.copy()
for key in ["PYTHONPATH", "LD_LIBRARY_PATH"]:
    if key in my_env:
        del my_env[key]
my_env["PYTHONUNBUFFERED"] = "1"

if os.path.exists(TEMP_OUT):
    os.remove(TEMP_OUT)

vis_file_path = os.path.join(IMG_DIR, f"trial_{trial_id:03d}_cam{cam_id}_cgn_vis.npz")

result = subprocess.run(
    [CONTACT_PYTHON, WORKER_SCRIPT,
     "--in_data",  TEMP_IN,
     "--out_data", TEMP_OUT,
     "--vis_data", vis_file_path],
    env=my_env
)

if result.returncode != 0 or not os.path.exists(TEMP_OUT):
    print("❌ CGN 后端运行失败")
    simulation_app.close(); sys.exit(1)

res_data = np.load(TEMP_OUT)
if not res_data['success']:
    print("❌ CGN 未能生成有效抓取")
    simulation_app.close(); sys.exit(1)

T_cam_grasp = res_data['best_grasp']
print(f"✅ 获取到抓取位姿，得分: {res_data['score']:.4f}")

# ===================== 坐标变换（最终正确版）=====================
# ┌─────────────────────────────────────────────────────────────┐
# │  🔧 可调参数区：只需要改这里                                 │
# │                                                             │
# │  GRASP_Z_DEG    : 绕进近轴修正夹爪朝向                      │
# │                   先试 90，不对试 -90 或 180                 │
# │                                                             │
# │  EE_FINGER_OFFSET: 手腕 panda_hand 到指尖的距离(m)           │
# │                   Franka 默认约 0.105，可微调               │
# └─────────────────────────────────────────────────────────────┘
GRASP_Z_DEG      = 0       # 👈 调夹爪朝向：90 / -90 / 180 / 0
#EE_FINGER_OFFSET = 0.105#0.105    # 👈 调指尖补偿距离，单位 m

# ── helper ────────────────────────────────────────────────────
def to_T44(R3=None, t3=None):
    T = np.eye(4)
    if R3 is not None: T[:3, :3] = R3
    if t3 is not None: T[:3,  3] = t3
    return T

# ── 相机世界位姿（重新读，保证最新）─────────────────────────────
cam_trans, cam_quat_curr = SingleXFormPrim("/World/Camera").get_world_pose()
T_world_cam = to_T44(quat_to_rot_matrix(cam_quat_curr), cam_trans)

# ── 三步变换 ─────────────────────────────────────────────────
# 1. T_world_cam: 相机在世界系的位姿
# 2. FLIP_CAM: Isaac 相机 -> OpenCV 相机
# 3. T_cam_grasp: CGN 原始输出
FLIP_CAM = np.diag([1., -1., -1.])

# 4. 核心修正：把 CGN 的夹爪约定（X轴开合）转为 Franka 约定（Y轴开合）
# 必须先绕 Z 轴转 90 度（或 -90 度，具体取决于你的 Franka 模型导入版本）
R_cgn_to_franka = R.from_euler('Z', 90, degrees=True).as_matrix()

# 5. 你的手动微调：基于修正后的姿态，再去转你想要的 GRASP_Z_DEG
R_user_tune = R.from_euler('Z', GRASP_Z_DEG, degrees=True).as_matrix()

# 将两个旋转组合 (先转 R_cgn_to_franka，再转 R_user_tune)
fix_rot = R_user_tune @ R_cgn_to_franka 

T_world_grasp = (T_world_cam
                 @ to_T44(FLIP_CAM)     
                 @ T_cam_grasp          
                 @ to_T44(fix_rot))     

grasp_pos  = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])
grasp_dir  = T_world_grasp[:3, 2]      # 进近方向依然是 Z

# ── 指尖补偿：让指尖（而非手腕）到达 CGN 目标点 ───────────────
#grasp_pos_cmd = grasp_pos - grasp_dir * EE_FINGER_OFFSET

# ── 诊断打印 ─────────────────────────────────────────────────
# print(f"\n{'─'*55}")
# print(f"[诊断] GRASP_Z_DEG      = {GRASP_Z_DEG}°")
# #print(f"[诊断] EE_FINGER_OFFSET = {EE_FINGER_OFFSET} m")
# print(f"[诊断] grasp_quat       = {np.round(grasp_quat, 4)}")
# print(f"[诊断] grasp_pos  (CGN目标)  = {np.round(grasp_pos, 3)}")
# #print(f"[诊断] grasp_pos_cmd (→IK)   = {np.round(grasp_pos_cmd, 3)}")
# print(f"[诊断] grasp_dir  (进近方向) = {np.round(grasp_dir, 3)}")
# print(f"{'─'*55}\n")

# ===================== 可视化：RGB 坐标轴 Marker =====================
axis_len   = 0.15
axis_thick = 0.005
R_mat = T_world_grasp[:3, :3]

axes = [
    ("GraspMarker_X", "marker_x", R_mat[:, 0], np.array([axis_len, axis_thick, axis_thick]), np.array([1., 0., 0.])),
    ("GraspMarker_Y", "marker_y", R_mat[:, 1], np.array([axis_thick, axis_len, axis_thick]), np.array([0., 1., 0.])),
    ("GraspMarker_Z", "marker_z", R_mat[:, 2], np.array([axis_thick, axis_thick, axis_len]), np.array([0., 0., 1.])),
]
for prim_path, name, direction, scale, color in axes:
    center = grasp_pos + direction * (axis_len / 2.0)
    world.scene.add(VisualCuboid(
        prim_path=f"/World/{prim_path}", name=name,
        position=center, orientation=grasp_quat,
        scale=scale, color=color,
    ))

world.step(render=True)
print("🔍 坐标轴 Marker 已生成（蓝色Z轴应指向物体内部）")

# ===================== 运动控制 =====================
from scipy.spatial.transform import Slerp

# ===================== 简单稳健的运动控制 (关节空间插值) =====================
def move_to_pose(target_pos, target_quat, steps=150, label=""):
    """只计算终点 IK，在关节空间进行线性插值，绝对稳健"""
    action, success = ik_solver.compute_inverse_kinematics(target_pos, target_quat)
    
    if success:
        curr_joints = franka.get_joint_positions()
        # 复制当前关节，防止夹爪状态被意外覆盖
        target_joints = np.copy(curr_joints) 
        # 只更新前 7 个机械臂关节的指令
        target_joints[:7] = action.joint_positions[:7] 
        
        for i in range(1, steps + 1):
            alpha = i / steps
            interp_joints = curr_joints * (1 - alpha) + target_joints * alpha
            franka.apply_action(ArticulationAction(joint_positions=interp_joints))
            world.step(render=True)
            
        print(f"  [IK诊断]{label} 成功到达 ✅")
        return True
    else: 
        print(f"  [IK诊断]{label} 🛑 IK 求解失败 (目标超出工作空间或碰撞)！")
        return False

def save_cam_img(filename):
    img = camera.get_rgb()[:, :, :3]
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ===================== 运动规划与执行 =====================

# 步骤 0：移动到悬停点
hover_pos = grasp_pos - grasp_dir * 0.10
print(f">>> 步骤 0: 移动到悬停点 {np.round(hover_pos, 3)}...")
move_to_pose(hover_pos, grasp_quat, steps=150, label=" [悬停]")

# 步骤 1：进近并抓取
insert_pos = grasp_pos + grasp_dir * 0.115 # 先到达一个稍微靠近物体的点，增加成功率
print(f"\n>>> 步骤 1: 进近抓取 → {np.round(insert_pos, 3)}")
save_cam_img(os.path.join(IMG_DIR, f"trial_{trial_id:03d}_cam{cam_id}_step0_before_grasp.png"))
move_to_pose(insert_pos, grasp_quat, steps=100, label=" [进近]")

# 夹爪闭合
franka.gripper.apply_action(ArticulationAction(
    joint_positions=franka.gripper.joint_closed_positions))
for _ in range(80):
    world.step(render=True)

save_cam_img(os.path.join(IMG_DIR, f"trial_{trial_id:03d}_cam{cam_id}_step1_grasped.png"))

# 步骤 2：提起物体
lift_pos = grasp_pos + np.array([0., 0., 0.15])
print(f"\n>>> 步骤 2: 提起物体 → {np.round(lift_pos, 3)}")
move_to_pose(lift_pos, grasp_quat, steps=150, label=" [提起]")
for _ in range(120):
    world.step(render=True)

save_cam_img(os.path.join(IMG_DIR, f"trial_{trial_id:03d}_cam{cam_id}_step2_final.png"))

# ===================== 最终汇总诊断 =====================
ee_final, _ = franka.end_effector.get_world_pose()
print(f"\n{'═'*55}")
print(f"✅ Trial {trial_id} / 视角 {cam_id} 完成")
print(f"  CGN 抓取目标:       {np.round(grasp_pos, 3)}")
#print(f"  IK 发送目标 (指尖): {np.round(grasp_pos_cmd, 3)}")
print(f"  末端最终位置:       {np.round(ee_final, 3)}")
print(f"{'═'*55}")
print()
#print("  如果夹爪还偏 90°：修改 GRASP_Z_DEG（当前值 =", GRASP_Z_DEG, "）")
#print("  如果抓取深度不对：修改 EE_FINGER_OFFSET（当前值 =", EE_FINGER_OFFSET, "m）")

simulation_app.close()