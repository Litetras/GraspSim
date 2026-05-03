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
    """在操作系统底层拦截并丢弃所有 C++ 打印信息"""
    def __enter__(self):
        # 记录终端原本的输出流位置
        self.old_stdout = os.dup(sys.stdout.fileno())
        self.old_stderr = os.dup(sys.stderr.fileno())
        # 打开操作系统的“黑洞”
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        # 将标准输出和报错强行接到黑洞里
        os.dup2(self.devnull, sys.stdout.fileno())
        os.dup2(self.devnull, sys.stderr.fileno())
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 启动完成后，把输出流接回正常终端
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
parser.add_argument("--trial", type=int, default=0)
parser.add_argument("--cam_id", type=int, default=1)
args = parser.parse_args()
trial_id = args.trial
cam_id = args.cam_id

IMG_DIR = "/home/zyp/ZYPLAB/Graspgen/eval_results"
os.makedirs(IMG_DIR, exist_ok=True)

# ===================== Isaac Sim 核心初始化 =====================
from isaacsim import SimulationApp

print(">>> 正在静默启动 Isaac Sim，请稍候...")

# 将最吵的启动过程关进“小黑屋”
with SuppressOutput():
    simulation_app = SimulationApp({"headless": False})
    
    # 启动完后，顺便把引擎后续运行中的日志也掐死
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

# (......下面接你的“路径配置”和后续代码......)
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
franka: Franka = world.scene.add(Franka(prim_path="/Franka", name="franka"))
camera = Camera(prim_path="/World/Camera", resolution=(1280, 720))
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
camera.add_rgb_to_frame()

world.reset()

# 暖机渲染
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
for key in ['sam3_model', 'sam3_processor', 'inference_state', 'output_obj', 'rgb_image', 'masks', 'scores', 'best_mask']:
    if key in locals():
        locals()[key] = None
        del locals()[key]

import gc; gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()

# ===================== 调用 CGN 后端 =====================
my_env = os.environ.copy()
for key in ["PYTHONPATH", "LD_LIBRARY_PATH"]:
    if key in my_env: del my_env[key]
my_env["PYTHONUNBUFFERED"] = "1"

if os.path.exists(TEMP_OUT):
    os.remove(TEMP_OUT)

# 🌟 在这里加上路径拼接！因为在这个文件里 IMG_DIR, trial_id, cam_id 都是现成的
vis_file_path = os.path.join(IMG_DIR, f"trial_{trial_id:03d}_cam{cam_id}_cgn_vis.npz")

result = subprocess.run(
    [CONTACT_PYTHON, WORKER_SCRIPT,  # 注意：这里的 WORKER_SCRIPT 必须指向你的 Z_cgn_worker_pt.py
     "--in_data", TEMP_IN, 
     "--out_data", TEMP_OUT,
     "--vis_data", vis_file_path],   # 传给最底层的可视化路径
    env=my_env
)

if result.returncode != 0 or not os.path.exists(TEMP_OUT):
    print("❌ CGN 后端运行失败")
    simulation_app.close(); sys.exit(1)



# 👇🌟 就是这里！把你之前不小心删掉的这几行读取数据的代码补回来
res_data = np.load(TEMP_OUT)
if not res_data['success']:
    print("❌ CGN 未能生成有效抓取")
    simulation_app.close(); sys.exit(1)

T_cam_grasp = res_data['best_grasp']
print(f"✅ 获取到抓取位姿，得分: {res_data['score']:.4f}")
# 👆🌟 =========================================================
# ===================== 坐标变换 =====================
def get_T(t, r):
    T = np.eye(4); T[:3, :3] = r; T[:3, 3] = t; return T

cam_trans, cam_quat_curr = SingleXFormPrim("/World/Camera").get_world_pose()
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat_curr))

T_world_grasp = (T_world_cam
                 @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                 @ T_cam_grasp
                 @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))

grasp_pos  = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])
grasp_dir  = T_world_grasp[:3, 2]

# ===================== 抓取控制逻辑 =====================
def move_to_pose(target_pos, target_quat, steps=150):
    action, success = ik_solver.compute_inverse_kinematics(target_pos, target_quat)
    if success:
        curr = franka.get_joint_positions()
        targ = np.copy(curr); targ[:7] = action.joint_positions
        for i in range(steps):
            alpha = i / steps
            franka.apply_action(ArticulationAction(joint_positions=curr*(1-alpha) + targ*alpha))
            world.step(render=True)
        return True
    else: 
        print("🛑 IK Failed (无法求解运动学逆解)")
        return False

def save_cam_img(filename):
    img = camera.get_rgb()[:, :, :3]
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ===================== 运动规划与执行 =====================
print(">>> 步骤: 移动到预抓取点...")
move_to_pose(grasp_pos - grasp_dir * 0.10, grasp_quat, steps=180)

print(">>> 步骤: 插入并抓取...")
move_to_pose(grasp_pos + grasp_dir * 0.125, grasp_quat, steps=80) # 保持 12cm 的插入深度

franka.gripper.apply_action(ArticulationAction(
    joint_positions=franka.gripper.joint_closed_positions))
for _ in range(60): world.step(render=True)

# 抓取后保存第一张截图
save_cam_img(os.path.join(IMG_DIR, f"trial_{trial_id:03d}_cam{cam_id}_step1_grasped.png"))

print(">>> 步骤: 提起物体...")
move_to_pose(grasp_pos + np.array([0, 0, 0.2]), grasp_quat, steps=120)
for _ in range(120): world.step(render=True)

# 提起后保存第二张截图
save_cam_img(os.path.join(IMG_DIR, f"trial_{trial_id:03d}_cam{cam_id}_step2_final.png"))

print(f"✅ Trial {trial_id} 视角 {cam_id} 测试完成！")
simulation_app.close()