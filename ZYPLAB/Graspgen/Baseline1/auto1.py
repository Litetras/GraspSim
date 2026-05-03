import os
import sys
# 🌟 优化核心 0：防止 PyTorch 显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import warnings
import logging
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
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

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["OMNI_LOG_LEVEL"] = "error"
os.environ["CARB_LOG_LEVEL"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OV_KIT_ALLOW_ROOT"] = "1"

# ===================== 实验设置参数 =====================
TOTAL_TRIALS = 98

# 隔离的 Conda 环境和 Baseline1 后处理脚本路径
CONTACT_PYTHON = "/home/zyp/anaconda3/envs/contact/bin/python"
WORKER_SCRIPT  = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Baseline1/cgn_worker_baseline1.py"

IMG_DIR = "/home/zyp/Desktop/eval_results"
os.makedirs(IMG_DIR, exist_ok=True)

# ===================== 1. 全局初始化 Isaac Sim =====================
print("🚀 [1/3] 正在静默启动 Isaac Sim (只需启动一次)...")
with SuppressOutput():
    from isaacsim import SimulationApp
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

# ===================== 2. 全局加载 SAM3 分割模型 =====================
print("🚀 [2/3] 正在加载 SAM3 模型 (只需加载一次)...")
sys.path.append(r'/home/zyp/GraspGen')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam3_model = build_sam3_image_model(checkpoint_path="/home/zyp/sam3/zypmodel/sam3/sam3.pt")
sam3_processor = Sam3Processor(sam3_model)
# 🌟 初始状态先放到 CPU 里，节约显存
sam3_model.to('cpu') 

my_env = os.environ.copy()
for key in ["PYTHONPATH", "LD_LIBRARY_PATH"]:
    if key in my_env:
        del my_env[key]
my_env["PYTHONUNBUFFERED"] = "1"

# ===================== 辅助控制函数 =====================
def to_T44(R3=None, t3=None):
    T = np.eye(4)
    if R3 is not None: T[:3, :3] = R3
    if t3 is not None: T[:3,  3] = t3
    return T

def save_cam_img(camera_obj, save_path):
    img = camera_obj.get_rgb()[:, :, :3]
    cv2.imwrite(save_path, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"📸 已保存截图: {save_path}")

# Baseline 1 专属：简单稳健的运动控制 (关节空间插值)
def move_to_pose(ik_solver, franka_bot, world_ctx, target_pos, target_quat, steps=150, label=""):
    action, success = ik_solver.compute_inverse_kinematics(target_pos, target_quat)
    if success:
        curr_joints = franka_bot.get_joint_positions()
        target_joints = np.copy(curr_joints) 
        target_joints[:7] = action.joint_positions[:7] 
        for i in range(1, steps + 1):
            alpha = i / steps
            interp_joints = curr_joints * (1 - alpha) + target_joints * alpha
            franka_bot.apply_action(ArticulationAction(joint_positions=interp_joints))
            world_ctx.step(render=True)
        print(f"  [IK诊断]{label} 成功到达 ✅")
        return True
    else: 
        print(f"  [IK诊断]{label} 🛑 IK 求解失败 (目标超出工作空间或碰撞)！")
        return False

# ===================== 3. 核心实验大循环 =====================
print(f"🚀 [3/3] 开始执行 Baseline1 自动化抓取评测，总计 {TOTAL_TRIALS} 轮...")

for trial in range(TOTAL_TRIALS):
    cam_id = (trial % 7) + 1 
    start_time = time.time()
    print(f"\n" + "="*60)
    print(f"🔄 正在启动第 {trial}/{TOTAL_TRIALS-1} 轮... (cam{cam_id}.usd)")
    print(f"="*60)
    
    # ---------------- 3.1 动态加载/重置场景 ----------------
    if World.instance() is not None:
        World.instance().clear_instance()

    usd_path = f"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/cam{cam_id}.usd"
    with SuppressOutput():
        open_stage(usd_path)

    world = World()
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    camera = Camera(prim_path="/World/Camera", resolution=(1280, 720))
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    camera.add_rgb_to_frame()

    world.reset()
    for _ in range(60): world.step(render=True)

    pos, quat = SingleXFormPrim("/World/Camera").get_world_pose()
    print(f">>> [Trial {trial}] 📷 相机位置: {np.round(pos, 3)}")

    franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
    ik_solver = KinematicsSolver(robot_articulation=franka)

    # ---------------- 3.2 图像采集与 SAM3 ----------------
    rgb_data = camera.get_rgb()[:, :, :3]
    depth_data = camera.get_depth()
    rgb_image = Image.fromarray(rgb_data.astype(np.uint8))

    print(">>> 正在进行语义分割 (knife)...")
    
    # 🌟 优化核心 1：搬回显卡推理
    sam3_model.to('cuda')
    
    inference_state = sam3_processor.set_image(rgb_image)
    output_obj = sam3_processor.set_text_prompt(state=inference_state, prompt="knife")
    masks = output_obj["masks"].cpu().numpy()
    scores = output_obj["scores"].cpu().numpy()
    
    # 🌟 优化核心 2：推理结束立刻回内存，并清空显存
    del inference_state
    del output_obj
    sam3_model.to('cpu')
    torch.cuda.empty_cache() 
    
    if len(masks) == 0: 
        print("❌ SAM3 未找到目标，跳过本轮！")
        continue

    best_mask = masks[np.argmax(scores)]
    if len(best_mask.shape) == 3: best_mask = best_mask[0]
    if best_mask.shape != rgb_data.shape[:2]:
        best_mask = zoom(best_mask, (rgb_data.shape[0]/best_mask.shape[0], rgb_data.shape[1]/best_mask.shape[1]), order=0) > 0.5
    final_mask = (best_mask > 0.5).astype(np.uint8)

    intrinsic = camera.get_intrinsics_matrix()
    cam_K = intrinsic[:3, :3].astype(np.float32)
    depth_data = np.nan_to_num(depth_data, posinf=0.0, neginf=0.0)
    depth_data = np.clip(depth_data, 0.0, 5.0) 

    # ---------------- 3.3 交互临时文件存储 ----------------
    TEMP_IN  = f"/tmp/cgn_in_trial{trial}.npz"
    TEMP_OUT = f"/tmp/cgn_out_trial{trial}.npz"
    vis_file_path = os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_cgn_vis.npz")

    if os.path.exists(TEMP_OUT): os.remove(TEMP_OUT)
    np.savez(TEMP_IN, depth=depth_data, K=cam_K, segmap=final_mask, rgb=rgb_data)

    # ---------------- 3.4 跨环境调用 CGN 子进程 ----------------
    print(">>> 启动 Contact-GraspNet 后端...")
    res_cgn = subprocess.run([CONTACT_PYTHON, WORKER_SCRIPT, "--in_data", TEMP_IN, "--out_data", TEMP_OUT, "--vis_data", vis_file_path], env=my_env)
    
    if res_cgn.returncode != 0 or not os.path.exists(TEMP_OUT):
        print("❌ CGN 运行失败，跳过本轮！")
        continue

    res_data = np.load(TEMP_OUT, allow_pickle=True)
    if not res_data['success']:
        print("❌ CGN 未能生成有效抓取，跳过本轮！")
        continue

    T_cam_grasp = res_data['best_grasp']
    print(f"✅ 获取到抓取位姿，得分: {res_data['score']:.4f}")

    # ---------------- 3.5 坐标变换与运动规划 ----------------
    GRASP_Z_DEG = 0
    cam_trans, cam_quat_curr = SingleXFormPrim("/World/Camera").get_world_pose()
    T_world_cam = to_T44(quat_to_rot_matrix(cam_quat_curr), cam_trans)

    FLIP_CAM = np.diag([1., -1., -1.])
    R_cgn_to_franka = R.from_euler('Z', 90, degrees=True).as_matrix()
    R_user_tune = R.from_euler('Z', GRASP_Z_DEG, degrees=True).as_matrix()
    fix_rot = R_user_tune @ R_cgn_to_franka 

    T_world_grasp = T_world_cam @ to_T44(FLIP_CAM) @ T_cam_grasp @ to_T44(fix_rot)     

    grasp_pos  = T_world_grasp[:3, 3]
    grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])
    grasp_dir  = T_world_grasp[:3, 2]

    # 可视化 Marker
    axis_len, axis_thick = 0.15, 0.005
    R_mat = T_world_grasp[:3, :3]
    for prim_path, name, direction, scale, color in [
        ("GraspMarker_X", "marker_x", R_mat[:, 0], np.array([axis_len, axis_thick, axis_thick]), np.array([1., 0., 0.])),
        ("GraspMarker_Y", "marker_y", R_mat[:, 1], np.array([axis_thick, axis_len, axis_thick]), np.array([0., 1., 0.])),
        ("GraspMarker_Z", "marker_z", R_mat[:, 2], np.array([axis_thick, axis_thick, axis_len]), np.array([0., 0., 1.])),
    ]:
        center = grasp_pos + direction * (axis_len / 2.0)
        world.scene.add(VisualCuboid(prim_path=f"/World/{prim_path}", name=name, position=center, orientation=grasp_quat, scale=scale, color=color))
    world.step(render=True)
    print("🔍 坐标轴 Marker 已生成（蓝色Z轴应指向物体内部）")

    # 执行动作
    hover_pos = grasp_pos - grasp_dir * 0.10
    print(f">>> 步骤 0: 移动到悬停点 {np.round(hover_pos, 3)}...")
    move_to_pose(ik_solver, franka, world, hover_pos, grasp_quat, steps=150, label=" [悬停]")
    
    insert_pos = grasp_pos + grasp_dir * 0.115
    print(f">>> 步骤 1: 进近抓取 → {np.round(insert_pos, 3)}")
    save_cam_img(camera, os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_step0_before_grasp.png"))
    move_to_pose(ik_solver, franka, world, insert_pos, grasp_quat, steps=100, label=" [进近]")

    franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
    for _ in range(80): world.step(render=True)
    save_cam_img(camera, os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_step1_grasped.png"))

    lift_pos = grasp_pos + np.array([0., 0., 0.15])
    print(f">>> 步骤 2: 提起物体 → {np.round(lift_pos, 3)}")
    move_to_pose(ik_solver, franka, world, lift_pos, grasp_quat, steps=150, label=" [提起]")
    for _ in range(120): world.step(render=True)
    save_cam_img(camera, os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_step2_final.png"))

    ee_final, _ = franka.end_effector.get_world_pose()
    print(f"\n{'═'*55}")
    print(f"✅ Trial {trial} / 视角 {cam_id} 完成 | 耗时: {time.time() - start_time:.1f}s")
    print(f"  CGN 抓取目标:       {np.round(grasp_pos, 3)}")
    print(f"  末端最终位置:       {np.round(ee_final, 3)}")
    print(f"{'═'*55}\n")

# 循环结束，安全关闭应用
print("\n🎉 全部评测执行完毕！请检查 eval_results 文件夹查看最终截图。")
simulation_app.close()
sys.exit(0)