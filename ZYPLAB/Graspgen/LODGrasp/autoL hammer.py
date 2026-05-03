import os
# 🌟 优化 1：防止 PyTorch 长时间运行产生显存碎片导致 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import gc
import cv2
import time
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import zoom  

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import VisualCuboid

# 导入你的自定义库
sys.path.append(r'/home/zyp/GraspGen')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from demogen_LOD import demo_variable

# ================= 全局参数配置 =================
SCENE_DIR = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib"
IMG_DIR = "batch_test_results"
os.makedirs(IMG_DIR, exist_ok=True)

PROMPT = "hammer"
task_name = "hammer_pin"
natural_instruction = "Grasp the hammer to pin."

camera_width, camera_height = 1280, 720
axis_len = 0.15   
axis_thick = 0.005  

# ================= 辅助函数 =================
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

def save_cam_img(camera_obj, save_path):
    img_rgb = camera_obj.get_rgb()[:, :, :3]
    img_bgr = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)
    print(f"📸 已保存截图: {save_path}")

# ================= 全局加载模型 (🌟 优化 2：仅从硬盘读一次) =================
print("🚀 正在全局加载 SAM3 模型...")
sam3_model = build_sam3_image_model(checkpoint_path="/home/zyp/sam3/zypmodel/sam3/sam3.pt")
sam3_processor = Sam3Processor(sam3_model)
# 初始化后立刻放入 CPU 内存，不占用宝贵的 GPU 显存
sam3_model.to('cpu') 


# ================= 批量测试主循环 =================
for cam_id in range(1, 8):#########################3567
    usd_path = os.path.join(SCENE_DIR, f"hammer_cam{cam_id}.usd")
    print("\n" + "=" * 60)
    print(f"🌍 [阶段 1] 正在加载全新场景: {usd_path}")
    print("=" * 60)

    # 清理并重建 World，防止多个场景堆叠
    if World.instance() is not None:
        World.instance().clear_instance()
        
    open_stage(usd_path)
    world = World()

    # 初始化机器人与相机
    franka: Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) 
    camera_path = "/World/Camera"
    camera = Camera(prim_path=camera_path, resolution=(camera_width, camera_height))
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    camera.add_rgb_to_frame()

    # 预创建坐标轴Marker (设为不可见，复用这些Prim防止卡顿)
    marker_x = world.scene.add(VisualCuboid(prim_path="/World/marker_x", name="marker_x", scale=np.array([axis_len, axis_thick, axis_thick]), color=np.array([1., 0., 0.])))
    marker_y = world.scene.add(VisualCuboid(prim_path="/World/marker_y", name="marker_y", scale=np.array([axis_thick, axis_len, axis_thick]), color=np.array([0., 1., 0.])))
    marker_z = world.scene.add(VisualCuboid(prim_path="/World/marker_z", name="marker_z", scale=np.array([axis_thick, axis_thick, axis_len]), color=np.array([0., 0., 1.])))
    marker_x.set_visibility(False)
    marker_y.set_visibility(False)
    marker_z.set_visibility(False)

    # 物理引擎预热
    world.reset()
    for _ in range(100): world.step()
    
    # 🌟 优化 3：求解器针对每个机器人在外层实例化一次即可
    ik_solver = KinematicsSolver(robot_articulation=franka)

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
            print("❌ IK 求解失败，跳过本次移动。")

    # 进入当前场景的14次抓取测试
    for trial in range(1, 15):
        print("\n" + "-" * 50)
        print(f"🎥 场景: hammer_cam{cam_id} | 🔄 测试轮次: {trial} / 14")
        print("-" * 50)

        # 每次 trial 前重置环境和机器人状态
        world.reset()
        franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
        for _ in range(100): world.step() # 等待稳定

        # ================== SAM3 图像处理 ==================
        print(f"[{time.strftime('%H:%M:%S')}] 正在进行 SAM3 图像分割...")
        rgb_data = camera.get_rgb()
        depth_data = camera.get_depth()  
        rgb_image = Image.fromarray(rgb_data.astype(np.uint8))

        # 🌟 优化 4：需要推理时，秒切回 GPU
        sam3_model.to('cuda')
        inference_state_obj = sam3_processor.set_image(rgb_image)
        output_obj = sam3_processor.set_text_prompt(state=inference_state_obj, prompt=PROMPT)
        
        masks = output_obj["masks"].cpu().numpy()
        scores = output_obj["scores"].cpu().numpy()

        # 🌟 优化 5：推理完毕，立刻释放显存，把模型搬回内存！
        del inference_state_obj
        del output_obj
        sam3_model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

        if len(masks) == 0:
            print(f"⚠️ SAM3未检测到'{PROMPT}'，跳过该 trial。")
            continue

        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        if len(best_mask.shape) == 3: best_mask = best_mask[0]  
        if best_mask.shape != rgb_data.shape[:2]:
            scale_y, scale_x = rgb_data.shape[0] / best_mask.shape[0], rgb_data.shape[1] / best_mask.shape[1]
            best_mask = zoom(best_mask, (scale_y, scale_x), order=0) > 0.5

        final_mask = (best_mask > 0.5).astype(np.uint8)

        # ================== Qwen 抓取推理 ==================
        intrinsic_matrix = camera.get_intrinsics_matrix()
        intrinsic = [float(intrinsic_matrix[0, 0]), float(intrinsic_matrix[1, 1]), 
                     float(intrinsic_matrix[0, 2]), float(intrinsic_matrix[1, 2])]

        print(f"[{time.strftime('%H:%M:%S')}] 🧠 运行大模型推理...")
        try:
            # 此时 GPU 显存已被 SAM3 完全腾出，供后续大模型尽情使用
            grasp = demo_variable(
                rgb_data=rgb_data, 
                depth_data=depth_data, 
                mask=final_mask, 
                intrinsic=intrinsic,
                natural_text=[natural_instruction], 
                strict_text=["nnn"],
                grasp_threshold=0.65, 
                num_grasps=200
            )
        except Exception as e:
            print(f"⚠️ 推理失败 ({e})，跳过该 trial。")
            continue

        # ================== 坐标转换与抓取执行 ==================
        cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
        T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

        T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ grasp.pose @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        T_world_grasp = move_along_grasp_dir(T_world_grasp, distance=0.1)
        
        grasp_pos = T_world_grasp[:3, 3]
        grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])
        grasp_dir = T_world_grasp[:3, 2]

        # --- 激活并对齐坐标轴 Marker ---
        R_mat = T_world_grasp[:3, :3]
        axes_data = [
            (marker_x, R_mat[:, 0]),
            (marker_y, R_mat[:, 1]),
            (marker_z, R_mat[:, 2]),
        ]
        for marker, direction in axes_data:
            center = grasp_pos + direction * (axis_len / 2.0)
            marker.set_world_pose(position=center, orientation=grasp_quat)
            marker.set_visibility(True)
        world.step(render=True)
        print("🔍 坐标轴 Marker 已更新显示 (蓝色Z轴指向插入方向)")

        # --- 动作步骤与截图 ---
        # 步骤 0
        print(">>> 步骤 0: 移动到预抓取点...")
        move_to_pose(grasp_pos-grasp_dir * 0.10, grasp_quat, step_count=180)
        save_cam_img(camera, os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_{task_name}_step0_before.png"))

        # 步骤 1
        print(">>> 步骤 1: 插入并闭合夹爪...")
        move_to_pose(grasp_pos + grasp_dir * 0.03, grasp_quat, step_count=80)
        franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
        for _ in range(80): world.step(render=True)
        save_cam_img(camera, os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_{task_name}_step1_grasped.png"))

        # 步骤 2
        print(">>> 步骤 2: 提起物体...")
        move_to_pose(grasp_pos -grasp_dir * 0.08, grasp_quat, step_count=120)
        for _ in range(120): world.step(render=True)
        save_cam_img(camera, os.path.join(IMG_DIR, f"trial_{trial:03d}_cam{cam_id}_{task_name}_step2_final.png"))

        # 本轮结束，隐藏 Marker
        marker_x.set_visibility(False)
        marker_y.set_visibility(False)
        marker_z.set_visibility(False)

simulation_app.close()
print("🎉 所有 98 次测试执行完毕！")