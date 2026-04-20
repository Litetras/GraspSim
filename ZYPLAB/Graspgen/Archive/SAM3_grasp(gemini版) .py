from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController 

usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"
open_stage(usd_path)

world = World()
franka: Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) 

controller = RMPFlowController(
    name="rmp_controller",
    robot_articulation=franka
)

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

# ===================== SAM3 初始化及图像处理 =====================
import sys
sys.path.append(r'/home/zyp/GraspGen')
import numpy as np
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

plt.figure("SAM3 Best Mask Result", figsize=(12, 6))
plt.subplot(121)
plt.imshow(rgb_data)
plt.title("Original RGB")
plt.subplot(122)
plt.imshow(final_mask, cmap='gray')
plt.title(f"Best Mask ('{PROMPT}', Score: {best_score:.3f})")
plt.suptitle("Press Enter to start grasp planning", fontsize=14)
plt.draw()
plt.waitforbuttonpress()
plt.close()

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
# 原有抓取推理和坐标变换逻辑（一字未改）
###########################################################################
from demogen import demo_variable

target_instruction = "down"  

grasp = demo_variable(
    rgb_data=rgb_data, 
    depth_data=depth_data, 
    mask=final_mask, 
    intrinsic=intrinsic,
    text=target_instruction  
)

def get_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T

def move_along_grasp_dir(htm: np.ndarray, distance: float = 0.1) -> np.ndarray:
    grasp_dir = htm[:3, 2]
    grasp_dir_unit = grasp_dir / np.linalg.norm(grasp_dir)
    new_t = htm[:3, 3] + grasp_dir_unit * distance
    new_htm = np.eye(4)
    new_htm[:3, :3] = htm[:3, :3]
    new_htm[:3, 3] = new_t
    return new_htm

from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat

cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))
T_cam_grasp = grasp.pose
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

T_world_grasp = move_along_grasp_dir(T_world_grasp, distance=0.1)
print(f"抓取位姿已沿抓取方向前移0.1米")

grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])

banana_position, banana_orientation = grasp_pos, grasp_quat
goal_position = banana_position.copy()
goal_position[0] += 0    
goal_position[2] += 0.06  

print("抓取点xyz: ", banana_position)
print("放置点xyz: ", goal_position)

###########################################################################
# === 底层控制状态机（增加反覆盖修复） ===
###########################################################################

rot_mat = quat_to_rot_matrix(banana_orientation)
local_z_axis = rot_mat[:, 2] 
approach_position = banana_position - 0.1 * local_z_axis

state = "APPROACH"
step_counter = 0
controller.reset()

print("开始执行自定义抓取流程...")

for i in range(100000):
    current_pos, _ = franka.end_effector.get_world_pose()
    
    # 默认保持张开
    gripper_action = franka.gripper.forward(action="open")
    actions = None
    
    if state == "APPROACH":
        actions = controller.forward(
            target_end_effector_position=approach_position,
            target_end_effector_orientation=banana_orientation
        )
        dist = np.linalg.norm(current_pos - approach_position)
        if dist < 0.02 or step_counter > 400:  
            state = "GRASP"
            step_counter = 0
            print(f"✅ 到达预备点 (误差:{dist:.3f}m)，准备推进...")
            
    elif state == "GRASP":
        actions = controller.forward(
            target_end_effector_position=banana_position,
            target_end_effector_orientation=banana_orientation
        )
        dist = np.linalg.norm(current_pos - banana_position)
        if dist < 0.015 or step_counter > 300:
            state = "CLOSE"
            step_counter = 0
            print(f"✅ 到达抓取点 (误差:{dist:.3f}m)，开始闭合夹爪...")
            
    elif state == "CLOSE":
        actions = controller.forward(
            target_end_effector_position=banana_position,
            target_end_effector_orientation=banana_orientation
        )
        # 生成闭合动作指令
        gripper_action = franka.gripper.forward(action="close")
        
        # 加长等待时间，给物理引擎 100 步的时间让夹爪合拢
        if step_counter > 100:
            state = "LIFT"
            step_counter = 0
            print("✅ 夹爪已闭合，准备抬起并放置...")
            
    elif state == "LIFT":
        actions = controller.forward(
            target_end_effector_position=goal_position,
            target_end_effector_orientation=banana_orientation
        )
        # 抬起和移动过程中持续保持闭合指令
        gripper_action = franka.gripper.forward(action="close")

    # =========================================================
    # 🌟 核心修复代码：防止指令冲突
    # =========================================================
    if actions is not None:
        # 如果 RMPFlow 输出了全套 9 个关节的值，我们就把夹爪真实想要执行的值强行插进去
        if actions.joint_positions is not None and len(actions.joint_positions) == franka.num_dof:
            for idx, pos in zip(franka.gripper.joint_dof_indices, gripper_action.joint_positions):
                actions.joint_positions[idx] = pos
        franka.apply_action(actions)
        
    # 双重保险：再次单独下发一次夹爪动作
    franka.gripper.apply_action(gripper_action)
    
    world.step(render=True) 
    step_counter += 1

simulation_app.close()