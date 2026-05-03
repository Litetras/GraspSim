# Z_eval_worker_baseline2.py
import os
import sys
import numpy as np
import time
import subprocess
import cv2
import argparse

# ===================== 解析命令行参数 =====================
parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=int, default=0, help="当前的实验轮次")
parser.add_argument('--cam_id', type=int, default=1, help="当前的相机视角ID")
parser.add_argument('--instruction', type=str, default="grasp the knife to cut", help="自然语言指令")
args = parser.parse_args()

# 任务解析
task_ins_txt = args.instruction.lower()
task_name, obj_class = "unknown", "unknown"
if "cut" in task_ins_txt and "knife" in task_ins_txt:
    task_name, obj_class = "cut", "knife"
elif "hammer" in task_ins_txt or "pound" in task_ins_txt:
    task_name, obj_class = "hammer", "hammer"
else:
    print("⚠️ 无法匹配任务，默认使用 hammer")
    task_name, obj_class = "hammer", "hammer"

print(f"\n==================================================")
print(f"🤖 [Trial {args.trial}] 解析结果 -> 任务: {task_name}, 物体: {obj_class}, 视角: cam{args.cam_id}")
print(f"==================================================\n")

# ===================== Isaac Sim 核心初始化 =====================
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True}) # 自动化如果不想看窗口，可改为 True

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import VisualCuboid
# ===================== 路径配置 (带 Trial 隔离的临时文件) =====================
CONTACT_PYTHON = "/home/zyp/anaconda3/envs/contact/bin/python"
WORKER_SCRIPT = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Baseline2/cgn_worker_baseline2.py" 

GRASPGPT_PYTHON = "/home/zyp/anaconda3/envs/graspgpt/bin/python"
GPT_WORKER_SCRIPT = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Baseline2/graspgpt_worker_baseline2.py"

# 🌟 为临时文件加上 trial 后缀，防止多轮缓存冲突
TEMP_IN = f"/tmp/cgn_in_trial{args.trial}.npz"       
TEMP_CGN_OUT = f"/tmp/cgn_out_trial{args.trial}.npz" 
TEMP_GPT_OUT = f"/tmp/gpt_out_trial{args.trial}.npz" 

# 🌟 初始化图片保存目录
IMG_DIR = "eval_results"
os.makedirs(IMG_DIR, exist_ok=True)

# ===================== 动态场景加载 =====================
# 🌟 根据传入的 cam_id 动态加载不同的 USD
usd_path = f"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/cam{args.cam_id}.usd"

open_stage(usd_path)

world = World()
franka: Franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka")) 
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

rgb_data = camera.get_rgb()[:, :, :3]
depth_data = camera.get_depth()
rgb_image = Image.fromarray(rgb_data.astype(np.uint8))

print(f">>> 正在进行语义分割 ({obj_class})...")
inference_state = sam3_processor.set_image(rgb_image)
# 使用解析出的物体名称进行分割
output_obj = sam3_processor.set_text_prompt(state=inference_state, prompt=obj_class.replace('_', ' '))

masks = output_obj["masks"].cpu().numpy()
scores = output_obj["scores"].cpu().numpy()
if len(masks) == 0: 
    print("❌ SAM3 未找到目标！")
    simulation_app.close()
    sys.exit(1)

best_mask = masks[np.argmax(scores)]
if len(best_mask.shape) == 3: best_mask = best_mask[0]
if best_mask.shape != rgb_data.shape[:2]:
    best_mask = zoom(best_mask, (rgb_data.shape[0]/best_mask.shape[0], rgb_data.shape[1]/best_mask.shape[1]), order=0) > 0.5
final_mask = (best_mask > 0.5).astype(np.uint8)

# ===================== 数据准备与显存清理 =====================
intrinsic = camera.get_intrinsics_matrix()
cam_K = intrinsic[:3, :3].astype(np.float32)

depth_data = np.nan_to_num(depth_data, posinf=0.0, neginf=0.0)
depth_data = np.clip(depth_data, 0.0, 5.0) 

print(">>> 保存数据并调用 Contact-GraspNet 后端...")
np.savez(TEMP_IN, depth=depth_data, K=cam_K, segmap=final_mask, rgb=rgb_data)

print(">>> 正在释放 SAM3 占用的 PyTorch 显存...")
try:
    del sam3_model
    del sam3_processor
    if 'inference_state' in locals(): del inference_state
    if 'output_obj' in locals(): del output_obj
    import gc
    gc.collect()
    torch.cuda.empty_cache() 
except Exception as e:
    pass

import os
my_env = os.environ.copy()
for key in ["PYTHONPATH", "LD_LIBRARY_PATH"]:
    if key in my_env:
        del my_env[key]
my_env["PYTHONUNBUFFERED"] = "1"




for tmp_file in [TEMP_CGN_OUT, TEMP_GPT_OUT]:
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

# ===================== 跨环境调用 CGN =====================
print(">>> [主程序] 已跨进程启动 Contact-GraspNet...")
result = subprocess.run(
    [CONTACT_PYTHON, WORKER_SCRIPT, "--in_data", TEMP_IN, "--out_data", TEMP_CGN_OUT],
    env=my_env 
)

if result.returncode != 0 or not os.path.exists(TEMP_CGN_OUT):
    print("❌ CGN 后端运行失败")
    simulation_app.close()
    sys.exit(1)

res_data = np.load(TEMP_CGN_OUT, allow_pickle=True)
if not res_data['success']:
    print("❌ CGN 未能生成有效抓取")
    simulation_app.close()
    sys.exit(1)

# ===================== 跨环境调用 GraspGPT =====================
print(f">>> [主程序] 跨进程启动 GraspGPT (任务: '{task_name}', 物体: '{obj_class}')...")
result_gpt = subprocess.run(
    [GRASPGPT_PYTHON, GPT_WORKER_SCRIPT, 
     "--in_data", TEMP_CGN_OUT, 
     "--out_data", TEMP_GPT_OUT,
     "--task", task_name,
     "--obj_class", obj_class],
    env=my_env 
)

if result_gpt.returncode != 0 or not os.path.exists(TEMP_GPT_OUT):
    print("❌ GraspGPT 后端运行失败")
    simulation_app.close()
    sys.exit(1)

gpt_res = np.load(TEMP_GPT_OUT, allow_pickle=True)
if not gpt_res['success']:
    print("❌ GraspGPT 未能筛选出有效抓取")
    simulation_app.close()
    sys.exit(1)

T_cam_grasp = gpt_res['best_grasp']
print(f"✅ GraspGPT 筛选完毕！最优得分: {gpt_res['score']:.4f}")

# ===================== 运动规划与执行 =====================
def get_T(t, r):
    T = np.eye(4); T[:3, :3] = r; T[:3, 3] = t; return T

cam_trans, cam_quat = SingleXFormPrim("/World/Camera").get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ \
                T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

def move_to_pose(target_pos, target_quat, steps=150):
    action, success = ik_solver.compute_inverse_kinematics(target_pos, target_quat)
    if success:
        curr = franka.get_joint_positions()
        targ = np.copy(curr); targ[:7] = action.joint_positions
        for i in range(steps):
            alpha = i / steps
            franka.apply_action(ArticulationAction(joint_positions=curr*(1-alpha) + targ*alpha))
            world.step(render=True)
    else: print("⚠️ IK Failed")

# 🌟 新增保存图片的辅助函数
def save_cam_img(save_path):
    img_rgb = camera.get_rgb()[:, :, :3]
    img_bgr = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    print(f"📸 已保存截图: {save_path}")

grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])
grasp_dir = T_world_grasp[:3, 2]
# ===================== 🌟 新增：可视化 RGB 坐标轴 Marker =====================
axis_len   = 0.15   # 坐标轴长度 (15cm)
axis_thick = 0.005  # 坐标轴粗细
R_mat = T_world_grasp[:3, :3]

axes = [
    ("GraspMarker_X", "marker_x", R_mat[:, 0], np.array([axis_len, axis_thick, axis_thick]), np.array([1., 0., 0.])),
    ("GraspMarker_Y", "marker_y", R_mat[:, 1], np.array([axis_thick, axis_len, axis_thick]), np.array([0., 1., 0.])),
    ("GraspMarker_Z", "marker_z", R_mat[:, 2], np.array([axis_thick, axis_thick, axis_len]), np.array([0., 0., 1.])),
]
for prim_path, name, direction, scale, color in axes:
    # 让坐标轴从抓取中心点向外延伸
    center = grasp_pos + direction * (axis_len / 2.0)
    world.scene.add(VisualCuboid(
        prim_path=f"/World/{prim_path}", name=name,
        position=center, orientation=grasp_quat,
        scale=scale, color=color,
    ))

world.step(render=True)
print("🔍 坐标轴 Marker 已生成（蓝色的Z轴应指向夹爪前进/插入的方向）")
# =========================================================================
# --- 拍摄第一张：接近抓取点前 ---
print(">>> 步骤 0: 移动到预抓取点...")
move_to_pose(grasp_pos - grasp_dir * 0.1, grasp_quat, steps=180)
save_cam_img(os.path.join(IMG_DIR, f"trial_{args.trial:03d}_cam{args.cam_id}_{task_name}_step0_before_grasp.png"))

# --- 拍摄第二张：闭合并抓稳后 ---
print(">>> 步骤 1: 插入并抓取...")
move_to_pose(grasp_pos + grasp_dir * 0.115, grasp_quat, steps=80)
franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
for _ in range(80): world.step(render=True) # 等待夹爪闭合稳定
save_cam_img(os.path.join(IMG_DIR, f"trial_{args.trial:03d}_cam{args.cam_id}_{task_name}_step1_grasped.png"))

# --- 拍摄第三张：提起物体后 ---
print(">>> 步骤 2: 提起物体...")
move_to_pose(grasp_pos + np.array([0, 0, 0.15]), grasp_quat, steps=120)#0.15m 提升高度
for _ in range(120): world.step(render=True)
save_cam_img(os.path.join(IMG_DIR, f"trial_{args.trial:03d}_cam{args.cam_id}_{task_name}_step2_final.png"))

print("✅ 本轮仿真执行完毕，关闭应用...")
simulation_app.close()
sys.exit(0) # 正常退出，让主控脚本知道