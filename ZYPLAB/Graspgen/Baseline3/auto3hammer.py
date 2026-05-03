import os
import sys
import time
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===================== 实验设置参数 =====================
TOTAL_TRIALS = 98
INSTRUCTION = "grasp the hammer to pound"

# 隔离的 Conda 环境和后处理脚本路径
CONTACT_PYTHON = "/home/zyp/anaconda3/envs/contact/bin/python"
WORKER_SCRIPT = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Baseline3/Z_cgn_worker_baseline3.py" 

GRASPGPT_PYTHON = "/home/zyp/anaconda3/envs/graspgpt/bin/python"
GPT_WORKER_SCRIPT = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Baseline3/Z_graspgpt_worker_baseline3.py"

IMG_DIR = "eval_results_baseline3"
os.makedirs(IMG_DIR, exist_ok=True)

# 解析自然语言指令
task_ins_txt = INSTRUCTION.lower()
if "cut" in task_ins_txt and "knife" in task_ins_txt:
    task_name, obj_class = "cut", "knife"
elif "hammer" in task_ins_txt or "pound" in task_ins_txt:
    task_name, obj_class = "hammer", "hammer"
else:
    task_name, obj_class = "hammer", "hammer"

# ===================== 1. 全局初始化 Isaac Sim =====================
print("🚀 [1/3] 正在启动 Isaac Sim (只需启动一次)...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # 自动化评测时可改为 True 加速渲染

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

# 准备跨进程环境变量
my_env = os.environ.copy()
for key in ["PYTHONPATH", "LD_LIBRARY_PATH"]:
    if key in my_env:
        del my_env[key]
my_env["PYTHONUNBUFFERED"] = "1"

# ===================== 辅助控制函数 =====================
def get_T(t, r):
    T = np.eye(4); T[:3, :3] = r; T[:3, 3] = t; return T

def move_to_pose(ik_solver, franka_bot, world_ctx, target_pos, target_quat, steps=150):
    action, success = ik_solver.compute_inverse_kinematics(target_pos, target_quat)
    if success:
        curr = franka_bot.get_joint_positions()
        targ = np.copy(curr); targ[:7] = action.joint_positions
        for i in range(steps):
            alpha = i / steps
            franka_bot.apply_action(ArticulationAction(joint_positions=curr*(1-alpha) + targ*alpha))
            world_ctx.step(render=True)
    else: 
        print("⚠️ IK 求解失败！")

def save_cam_img(camera_obj, save_path):
    img_rgb = camera_obj.get_rgb()[:, :, :3]
    img_bgr = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    print(f"📸 已保存截图: {save_path}")


# ===================== 3. 核心实验大循环 =====================
print(f"🚀 [3/3] 开始执行自动化抓取评测，总计 {TOTAL_TRIALS} 轮...")

for trial in range(TOTAL_TRIALS):
    cam_id = (trial % 7) + 1 
    start_time = time.time()
    print(f"\n" + "="*60)
    print(f"🔄 正在启动第 {trial}/{TOTAL_TRIALS-1} 轮... (hammer_cam{cam_id}.usd | 任务: {task_name} | 物体: {obj_class})")
    print(f"="*60)
    
    # ---------------- 3.1 动态加载/重置场景 ----------------
    # 必须清除旧的实例，防止在多次循环中累积导致奔溃
    if World.instance() is not None:
        World.instance().clear_instance()

    usd_path = f"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/hammer_cam{cam_id}.usd"
    open_stage(usd_path)

    # 重新绑定对象
    world = World()
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    camera = Camera(prim_path="/World/Camera", resolution=(1280, 720))
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    camera.add_rgb_to_frame()

    world.reset()
    for _ in range(60): world.step()
    franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
    ik_solver = KinematicsSolver(robot_articulation=franka)

# ---------------- 3.2 图像采集与 SAM3 ----------------
    rgb_data = camera.get_rgb()[:, :, :3]
    depth_data = camera.get_depth()
    rgb_image = Image.fromarray(rgb_data.astype(np.uint8))

    print(f">>> 正在进行语义分割 ({obj_class})...")
    
    # 🌟 优化核心 1：将 SAM3 快速搬回显卡
    sam3_model.to('cuda')
    
    inference_state = sam3_processor.set_image(rgb_image)
    output_obj = sam3_processor.set_text_prompt(state=inference_state, prompt=obj_class.replace('_', ' '))

    masks = output_obj["masks"].cpu().numpy()
    scores = output_obj["scores"].cpu().numpy()
    
    # 🌟 优化核心 2：推理结束后，立即将 SAM3 搬到 CPU 内存，并强制清空 GPU 显存
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
    TEMP_IN = f"/tmp/cgn_in_trial{trial}.npz"       
    TEMP_CGN_OUT = f"/tmp/cgn_out_trial{trial}.npz" 
    TEMP_GPT_OUT = f"/tmp/gpt_out_trial{trial}.npz" 

    for tmp_file in [TEMP_CGN_OUT, TEMP_GPT_OUT]:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    np.savez(TEMP_IN, depth=depth_data, K=cam_K, segmap=final_mask, rgb=rgb_data)

    # ---------------- 3.4 跨环境调用子进程 ----------------
    print(">>> 启动 Contact-GraspNet 后端...")
    res_cgn = subprocess.run([CONTACT_PYTHON, WORKER_SCRIPT, "--in_data", TEMP_IN, "--out_data", TEMP_CGN_OUT], env=my_env)
    
    if res_cgn.returncode != 0 or not os.path.exists(TEMP_CGN_OUT):
        print("❌ CGN 运行失败，跳过本轮！")
        continue

    res_data = np.load(TEMP_CGN_OUT, allow_pickle=True)
    if not res_data['success']:
        print("❌ CGN 未能生成有效抓取，跳过本轮！")
        continue

    print(">>> 启动 GraspGPT 后端...")
    res_gpt = subprocess.run([GRASPGPT_PYTHON, GPT_WORKER_SCRIPT, "--in_data", TEMP_CGN_OUT, "--out_data", TEMP_GPT_OUT, "--task", task_name, "--obj_class", obj_class], env=my_env)
    
    if res_gpt.returncode != 0 or not os.path.exists(TEMP_GPT_OUT):
        print("❌ GraspGPT 运行失败，跳过本轮！")
        continue

    gpt_res = np.load(TEMP_GPT_OUT, allow_pickle=True)
    if not gpt_res['success']:
        print("❌ GraspGPT 未能筛选出有效抓取，跳过本轮！")
        continue

    # ---------------- 3.5 运动规划与抓取执行 ----------------
    T_cam_grasp = gpt_res['best_grasp']
    print(f"✅ GraspGPT 筛选完毕！最优得分: {gpt_res['score']:.4f}")

    cam_trans, cam_quat = SingleXFormPrim("/World/Camera").get_world_pose() 
    T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))
    T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ \
                    T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    grasp_pos = T_world_grasp[:3, 3]
    grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])
    grasp_dir = T_world_grasp[:3, 2]

    # 生成 Marker 可视化
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

    print(">>> 步骤 0: 移动到预抓取点...")
    move_to_pose(ik_solver, franka, world, grasp_pos - grasp_dir * 0.1, grasp_quat, steps=180)
    save_cam_img(camera, os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_{task_name}_step0.png"))

    print(">>> 步骤 1: 插入并闭合夹爪...")
    move_to_pose(ik_solver, franka, world, grasp_pos + grasp_dir * 0.125, grasp_quat, steps=80)
    franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
    for _ in range(80): world.step(render=True)
    save_cam_img(camera, os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_{task_name}_step1.png"))

    print(">>> 步骤 2: 提起物体...")
    move_to_pose(ik_solver, franka, world, grasp_pos + np.array([0, 0, 0.2]), grasp_quat, steps=120)
    for _ in range(120): world.step(render=True)
    save_cam_img(camera, os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_{task_name}_step2.png"))

    print(f"🎉 第 {trial} 轮顺利完成！耗时: {time.time() - start_time:.1f}s")

# 循环结束，安全关闭应用
print("\n🎉 全部评测执行完毕！正在退出 Isaac Sim...")
simulation_app.close()
sys.exit(0)