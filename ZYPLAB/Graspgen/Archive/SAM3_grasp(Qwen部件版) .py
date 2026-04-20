from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka

# ================= 修复点 =================
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
# ==========================================

import numpy as np
import os
import cv2
import torch
import gc
import re
import base64
import requests
from io import BytesIO
from PIL import Image

# 初始化路径和场景
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

# 仿真预热
world.reset()
for i in range(100):
    world.step()
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

ik_solver = KinematicsSolver(robot_articulation=franka)

# ===================== 1. SAM3 处理及显存释放 =====================
import sys
sys.path.append(r'/home/zyp/GraspGen')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scipy.ndimage import zoom  

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"加载SAM3模型 (设备: {device})...")
sam3_checkpoint = "/home/zyp/sam3/zypmodel/sam3/sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint)
sam3_processor = Sam3Processor(sam3_model)

rgb_data = camera.get_rgb()
depth_data = camera.get_depth()  

PROMPT_OBJ = "knife"  # 文字提示：寻找
print(f"开始执行SAM3文字提示分割，寻找 '{PROMPT_OBJ}'...")

rgb_image = Image.fromarray(rgb_data.astype(np.uint8))
inference_state_obj = sam3_processor.set_image(rgb_image)
output_obj = sam3_processor.set_text_prompt(state=inference_state_obj, prompt=PROMPT_OBJ)

masks = output_obj["masks"].cpu().numpy()
scores = output_obj["scores"].cpu().numpy()

if len(masks) == 0:
    raise ValueError(f"❌ SAM3未检测到任何 '{PROMPT_OBJ}'！")

best_mask = masks[np.argmax(scores)]
if len(best_mask.shape) == 3: best_mask = best_mask[0]  
if best_mask.shape != rgb_data.shape[:2]:
    scale_y, scale_x = rgb_data.shape[0] / best_mask.shape[0], rgb_data.shape[1] / best_mask.shape[1]
    best_mask = zoom(best_mask, (scale_y, scale_x), order=0) > 0.5
final_mask = (best_mask > 0.5).astype(np.uint8)

# --- 🚨 核心优化：深度释放 SAM3 显存 🚨 ---
print("✅ SAM3 处理完成，正在回收显存...")
del sam3_model
del sam3_processor
del inference_state_obj
gc.collect()
torch.cuda.empty_cache() # 强制排空显存缓冲区

# ===================== 2. 大模型极速调用逻辑 (稳定分离版) =====================
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# ----------------- A. 纯文本推理抓取方向 (瞬间完成) -----------------
print("\n🧠 正在呼叫 Qwen3.5:4b 推理抓取方向...")
payload_text = {
    "model": "qwen3.5:4b",
    "messages": [
        {"role": "system", "content": "You are a robotic parser. Answer ONLY one word: up or down."},
        {"role": "user", "content": "I need to grasp the knife to cut bread. Direction of palm?"}
    ],
    "stream": False,
    "options": {"temperature": 0.0}
}

try:
    res_text = requests.post(OLLAMA_API_URL, json=payload_text, timeout=30).json()
    dir_res = res_text['message']['content'].lower()
    target_instruction = "up" if "up" in dir_res else "down"
    print(f"✅ 决策抓取方向: '{target_instruction}' (模型输出: {dir_res.strip()})")
except Exception as e:
    print(f"⚠️ 方向推理失败 ({e})，默认使用 'down'")
    target_instruction = "down"

# ----------------- B. 视觉推理提取 BBox (降采样极速版) -----------------
print("\n👁️ 正在呼叫 Qwen3.5:4b 提取把手边界框 (BBox)...")

# 🚀 核心提速：将 1280x720 缩小为 640x360，让 VLM 计算量骤减四分之三！
resized_img = rgb_image.resize((640, 360), Image.Resampling.LANCZOS)
buffered = BytesIO()
resized_img.save(buffered, format="JPEG", quality=70)
img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

payload_vis = {
    "model": "qwen3.5:4b",
    "messages": [
        {
            "role": "user", 
            "content": "Identify the center of the  knife‘s blade. Return the bounding box in [ymin, xmin, ymax, xmax] format using <box></box> tags.",
            "images": [img_base64]
        }
    ],
    "stream": False,
    "options": {"temperature": 0.0} # 去掉 num_predict，防止模型突然截断
}

final_part_mask = None
try:
    # 视觉推理由于图片已经缩小，速度会快很多
    res_vis = requests.post(OLLAMA_API_URL, json=payload_vis, timeout=60).json()
    bbox_res = res_vis['message']['content'].lower()
    print(f"大模型原始回答:\n{bbox_res}")
    
    # 解析 BBox
    numbers = re.findall(r'\d+', bbox_res)
    H, W = rgb_data.shape[:2]

    if len(numbers) >= 4:
        # 提取前四个数字：通常顺序是 ymin, xmin, ymax, xmax
        x1, y1, x2, y2 = map(int, numbers[:4])
        
        # 打印一下解析出的原始数值，方便调试
        print(f"解析数值: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Qwen 坐标转换 (0-1000 映射到像素)
        px1, py1 = int((x1 / 1000.0) * W), int((y1 / 1000.0) * H)
        px2, py2 = int((x2 / 1000.0) * W), int((y2 / 1000.0) * H)
        
        # 简单的边界检查，防止越界
        px1, px2 = min(px1, px2), max(px1, px2)
        py1, py2 = min(py1, py2), max(py1, py2)

        final_part_mask = np.zeros((H, W), dtype=np.uint8)
        final_part_mask[max(0,py1):min(H,py2), max(0,px1):min(W,px2)] = 1
        print(f"✅ 部件 BBox 解析成功: [{px1}, {py1}, {px2}, {py2}]")
        
        # 🌟 保存可视化图片供你检查
        img_bgr = cv2.cvtColor(rgb_data.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_bgr, (px1, py1), (px2, py2), (0, 255, 0), 3)
        cv2.putText(img_bgr, f'Qwen Target', (px1, max(py1-10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite("qwen_combined_result.png", img_bgr)
    else:
        print("⚠️ 未能提取有效 BBox，将执行全物体抓取")
        
except Exception as e:
    print(f"⚠️ 大模型视觉调用失败 ({e})，将执行全物体抓取")


# ===================== 3. 抓取生成与还原 =====================
print("\nopen meshcat-server")
intrinsic_mat = camera.get_intrinsics_matrix()
intrinsic = [float(intrinsic_mat[0, 0]), float(intrinsic_mat[1, 1]), float(intrinsic_mat[0, 2]), float(intrinsic_mat[1, 2])]

from demogen_part import demo_variable

grasp = demo_variable(
    rgb_data=rgb_data, 
    depth_data=depth_data, 
    mask=final_mask,            # 完整的物体(防碰撞)
    intrinsic=intrinsic,
    text=target_instruction,    # 大模型决定的方向
    part_mask=final_part_mask   # 大模型定位的部件(爪尖过滤)
)

# 坐标变换逻辑
def get_T(translation, rotation_matrix):
    T = np.eye(4); T[:3, :3] = rotation_matrix; T[:3, 3] = translation
    return T

def move_along_grasp_dir(htm, distance=0.1):
    grasp_dir = htm[:3, 2]
    new_t = htm[:3, 3] + (grasp_dir / np.linalg.norm(grasp_dir)) * distance
    new_htm = np.eye(4); new_htm[:3, :3] = htm[:3, :3]; new_htm[:3, 3] = new_t
    return new_htm

cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))
T_cam_grasp = grasp.pose
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

T_world_grasp = move_along_grasp_dir(T_world_grasp, distance=0.1)
grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])

# ===================== 4. 机械臂控制序列 =====================
def move_to_pose(target_pos, target_quat, step_count=150):
    action, success = ik_solver.compute_inverse_kinematics(target_position=target_pos, target_orientation=target_quat)
    if success:
        current_joints = franka.get_joint_positions()
        target_joints = np.copy(current_joints); target_joints[:7] = action.joint_positions 
        for i in range(step_count):
            alpha = i / step_count
            franka.apply_action(ArticulationAction(joint_positions=current_joints*(1-alpha) + target_joints*alpha))
            world.step(render=True)
    else:
        print(f"❌ IK 失败！")

# 执行动作
grasp_dir = T_world_grasp[:3, 2]
pre_grasp_pos = grasp_pos + grasp_dir * -0.1

print("\n>>> 正在移动到预抓取位...")
move_to_pose(pre_grasp_pos, grasp_quat, step_count=200)
print(">>> 正在插入抓取点...")
move_to_pose(grasp_pos, grasp_quat, step_count=100)
print(">>> 闭合夹爪...")
franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
for _ in range(80): world.step(render=True)
print(">>> 提起物体...")
lift_pos = grasp_pos.copy(); lift_pos[2] += 0.2  
move_to_pose(lift_pos, grasp_quat, step_count=150)

for _ in range(100): world.step(render=True)
simulation_app.close()