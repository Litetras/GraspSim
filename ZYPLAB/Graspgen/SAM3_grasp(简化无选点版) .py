from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"
open_stage(usd_path)

world = World()
franka: Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) 

controller = PickPlaceController(
    name="pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
    end_effector_initial_height=0.3,  # 机械臂末端执行器初始安全高度
    events_dt=[0.008, 0.005, 1, 0.01, 0.05, 0.05, 0.0025, 1, 0.008, 0.08], 
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
plt.ion()  
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"加载SAM3模型 (设备: {device})...")
sam3_checkpoint = "/home/zyp/sam3/zypmodel/sam3/sam3.pt"
sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint)
sam3_processor = Sam3Processor(sam3_model)

rgb_data = camera.get_rgb()
depth_data = camera.get_depth()  # graspgen需要m单位深度图
print("深度图形状:", depth_data.shape, "数值范围:", np.min(depth_data), "~", np.max(depth_data))


# ===================== 全新逻辑：直接用文字分割置信度最高的刀具 =====================
PROMPT = "knife"
print(f"开始执行SAM3文字提示分割，寻找 '{PROMPT}'...")

# 转换图像格式并送入SAM3
rgb_image = Image.fromarray(rgb_data.astype(np.uint8))
inference_state_obj = sam3_processor.set_image(rgb_image)

# 用文字提示进行分割
output_obj = sam3_processor.set_text_prompt(
    state=inference_state_obj,
    prompt=PROMPT
)

masks = output_obj["masks"].cpu().numpy()
scores = output_obj["scores"].cpu().numpy()

if len(masks) == 0:
    raise ValueError(f"❌ SAM3未检测到任何 '{PROMPT}'！请检查相机视野。")

# 提取置信度最高的一个 Mask
best_idx = np.argmax(scores)
best_mask = masks[best_idx]
best_score = scores[best_idx]

print(f"✅ 成功找到置信度最高的 '{PROMPT}'，置信度为: {best_score:.3f}")

# 处理mask维度和尺寸（适配原图尺寸）
if len(best_mask.shape) == 3:
    best_mask = best_mask[0]  # 去除batch/channel维度
if best_mask.shape != rgb_data.shape[:2]:
    scale_y = rgb_data.shape[0] / best_mask.shape[0]
    scale_x = rgb_data.shape[1] / best_mask.shape[1]
    best_mask = zoom(best_mask, (scale_y, scale_x), order=0) > 0.5

# 转换成 uint8 格式，GraspGen 要求目标掩码值为 1
final_mask = (best_mask > 0.5).astype(np.uint8) # type: ignore

# 可视化提取结果
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
intrinsic = [fx, fy, cx, cy]  # 适配graspnet输入格式
print("相机内参 fx, fy, cx, cy: ", intrinsic)



###########################################################################
# 原有抓取推理和坐标变换逻辑（保持不变）
###########################################################################
from demogen import demo_variable

# 测试你想抓取的方向
target_instruction = "down"  # 或者是 "down"

# 送入网络进行抓取生成
grasp = demo_variable(
    rgb_data=rgb_data, 
    depth_data=depth_data, 
    mask=final_mask, 
    intrinsic=intrinsic,
    text=target_instruction  # 传入语言指令
)

##############################################################
# 坐标变换工具函数
def get_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T

# 沿抓取方向前移0.1米的函数
def move_along_grasp_dir(htm: np.ndarray, distance: float = 0.1) -> np.ndarray:
    grasp_dir = htm[:3, 2]
    grasp_dir_unit = grasp_dir / np.linalg.norm(grasp_dir)
    new_t = htm[:3, 3] + grasp_dir_unit * distance
    new_htm = np.eye(4)
    new_htm[:3, :3] = htm[:3, :3]
    new_htm[:3, 3] = new_t
    return new_htm

from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat

# 1. 获取相机在世界坐标系中的位姿    
cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

# 2. 构造相机坐标系下的抓取位姿
T_cam_grasp = grasp.pose

# 3. 坐标变换：相机坐标系→世界坐标系
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# 前移抓取位姿
T_world_grasp = move_along_grasp_dir(T_world_grasp, distance=0.1)
print(f"抓取位姿已沿抓取方向前移0.1米")

grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])

# 确定抓取点和放置点
banana_position, banana_orientation = grasp_pos, grasp_quat
goal_position = banana_position.copy()
goal_position[0] += 0    
goal_position[2] += 0.06  

print("抓取点xyz: ", banana_position)
print("放置点xyz: ", goal_position)

###########################################################################
# 机械臂抓取控制（保持不变）
###########################################################################
for i in range(100000):
    current_joint_positions = franka.get_joint_positions()
    actions = controller.forward(
        picking_position=banana_position,
        placing_position=goal_position,
        current_joint_positions=current_joint_positions,
        end_effector_orientation=banana_orientation
    )
    franka.apply_action(actions)
    world.step(render=True) 

# 关闭仿真
simulation_app.close()