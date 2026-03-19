from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

# 场景加载路径（请确认绝对路径正确性）
usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"
open_stage(usd_path)

# 初始化仿真世界和机械臂
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
print("相机初始化完成")
print("获取相机内参: \n", camera.get_intrinsics_matrix())

# 仿真预热，让物体稳定
world.reset()
for i in range(100):
    world.step()
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

###########################################################################
# 核心替换：SAM → SAM2 分割模块（保留完整SAM2逻辑） + SAM3文字分割所有手柄
###########################################################################
import sys
sys.path.append(r'/home/zyp/GraspGen')

# ===================== 基础依赖导入 =====================
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from PIL import Image
from scipy.ndimage import zoom  # SAM3需要的mask尺寸调整依赖
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ===================== SAM3 相关导入（新增） =====================
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ===================== SAM2 配置（保留完整） =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False
plt.ion()  

# SAM2模型路径（请确认路径正确性）
sam_checkpoint = "/home/zyp/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== SAM3 初始化（新增） =====================
print(f"加载SAM3模型 (设备: {device})...")
# SAM3模型路径（请确认路径正确性）
sam3_checkpoint = "/home/zyp/sam3/zypmodel/sam3/sam3.pt"
# 初始化SAM3模型和处理器
sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint)
sam3_processor = Sam3Processor(sam3_model)

# ===================== 1. 获取相机数据 =====================
rgb_data = camera.get_rgb()
depth_data = camera.get_depth()  # graspgen需要m单位深度图
print("深度图形状:", depth_data.shape, "数值范围:", np.min(depth_data), "~", np.max(depth_data))

# ===================== 2. 加载SAM2模型（保留完整） =====================
print(f"加载SAM2模型 (设备: {device})...")
sam2_model = build_sam2(model_cfg, sam_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(rgb_data)  # 设置待分割图像

# ===================== 3. 交互式点选目标区域（保留完整） =====================
print("""
图像窗口操作说明:
1. 左键点击: 选择前景点（需要分割的目标物体）
2. 右键点击: 选择背景点（需要排除的区域，可选）
3. 按Enter/空格: 确认选点并执行分割
4. 按ESC: 取消选点重新选择
""")

# 初始化选点存储
input_points = []
input_labels = []  # 1=前景(左键), 0=背景(右键)
selected_points = []  # 用于可视化选点

# 创建选点窗口
fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(rgb_data)
ax.set_title("Select target area (Left=foreground, Right=background), Press Enter to confirm")
fig.canvas.draw()

# 鼠标点击事件处理函数
def on_click(event):
    if event.inaxes != ax:
        return
    # 记录坐标和按键类型
    x, y = event.xdata, event.ydata
    if event.button == MouseButton.LEFT:
        input_points.append([x, y])
        input_labels.append(1)
        selected_points.append(ax.plot(x, y, 'ro', markersize=8)[0])  # 前景红点
    elif event.button == MouseButton.RIGHT:
        input_points.append([x, y])
        input_labels.append(0)
        selected_points.append(ax.plot(x, y, 'bo', markersize=8)[0])  # 背景蓝点
    fig.canvas.draw()

# 键盘事件处理函数
def on_key(event):
    if event.key in ['enter', ' ']:  # 确认选点
        plt.close(fig)
    elif event.key == 'escape':  # 清空重选
        input_points.clear()
        input_labels.clear()
        for p in selected_points:
            p.remove()
        selected_points.clear()
        fig.canvas.draw()

# 绑定事件
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show(block=True)  # 阻塞直到窗口关闭

# 校验选点结果
if len(input_points) == 0:
    raise ValueError("未选择任何点！请重新运行并选择目标区域")

# 转换为SAM2所需格式
input_points = np.array(input_points, dtype=np.float32)
input_labels = np.array(input_labels, dtype=np.int32)
print(f"选点完成：前景点{np.sum(input_labels==1)}个，背景点{np.sum(input_labels==0)}个")

# ===================== 4. SAM2执行分割（保留完整） =====================
print("执行SAM2分割（仅输出一个mask）...")
# multimask_output=False 确保只输出一个mask，无需选择最优
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,  # 关键参数：只输出一个mask
)

# 直接获取唯一的mask（无需选最优）
object_mask = masks[0].astype(np.uint8)  # True→1, False→0
print(f"物体Mask形状: {object_mask.shape}, 置信度: {scores[0]:.3f}")

# -------------------------- 关键修改：用mask提取物体的RGB图像 --------------------------
print("从原始RGB中提取目标物体的RGB图像...")
# 扩展mask维度以匹配RGB图像（H,W）→（H,W,3）
object_mask_3ch = np.repeat(object_mask[:, :, np.newaxis], 3, axis=2)
# 提取物体RGB图像（mask=1的区域保留原始像素，mask=0的区域设为黑色）
object_rgb = np.where(object_mask_3ch == 1, rgb_data, 0)

# 可视化SAM2分割结果和提取的物体RGB图像
plt.figure("SAM2 Segmentation & Object RGB", figsize=(15, 7))
plt.subplot(131)
plt.imshow(rgb_data)
plt.title("Original RGB")
plt.subplot(132)
plt.imshow(object_mask, cmap='gray')
plt.title(f"Object Mask (Confidence: {scores[0]:.3f})")
plt.subplot(133)
plt.imshow(object_rgb)
plt.title("Extracted Object RGB")
plt.suptitle("Press Enter to continue to handle segmentation", fontsize=12)
plt.draw()
plt.waitforbuttonpress()
plt.close()

# ===================== 5. SAM3文字提示分割所有手柄（核心修改） =====================
print("开始执行SAM3文字提示分割所有手柄（输入：物体RGB图像 + 'handle'提示）...")
# 设置置信度阈值（可根据实际情况调整，建议0.2-0.5）
CONFIDENCE_THRESHOLD = 0.6

# 将numpy数组转换为PIL Image（SAM3要求的输入格式）
object_rgb_image = Image.fromarray(object_rgb.astype(np.uint8))
# 设置图像并进行文字提示分割
inference_state = sam3_processor.set_image(object_rgb_image)
output = sam3_processor.set_text_prompt(state=inference_state, prompt="handle")

# 获取SAM3输出的mask、边界框和分数
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# 处理mask格式，适配原有代码逻辑
masks = masks.cpu().numpy()
scores_np = scores.cpu().numpy()

# 初始化最终的手柄mask（全0）
handle_mask = np.zeros(object_rgb.shape[:2], dtype=np.uint8)
valid_mask_count = 0

# 遍历所有检测到的mask，合并所有符合置信度的手柄
if len(masks) > 0:
    for idx, (mask, score) in enumerate(zip(masks, scores_np)):
        # 过滤低置信度mask
        if score < CONFIDENCE_THRESHOLD:
            print(f"跳过低置信度mask {idx}，置信度: {score:.3f} < {CONFIDENCE_THRESHOLD}")
            continue
        
        # 确保mask是2D的（SAM3可能输出3D tensor）
        if len(mask.shape) == 3:
            mask = mask[0]
        
        # 调整mask大小以匹配原始物体RGB图像尺寸
        if mask.shape != object_rgb.shape[:2]:
            scale_y = object_rgb.shape[0] / mask.shape[0]
            scale_x = object_rgb.shape[1] / mask.shape[1]
            mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
        
        # 转换为uint8格式（0/1），并合并到总mask中
        single_handle_mask = (mask > 0.5).astype(np.uint8)
        handle_mask = np.bitwise_or(handle_mask, single_handle_mask)
        valid_mask_count += 1
        print(f"合并手柄mask {idx}，置信度: {score:.3f}")
    
    print(f"SAM3分割成功：共检测到 {len(masks)} 个实例，合并 {valid_mask_count} 个有效手柄mask")
else:
    # 如果没有检测到手柄，创建空mask
    print("警告：SAM3未检测到任何手柄！")

#可视化SAM3手柄分割结果（显示所有合并后的手柄）
plt.figure("Handle Segmentation Result (SAM3 - All Handles)", figsize=(15, 7))
plt.subplot(131)
plt.imshow(rgb_data)
plt.title("Original RGB")
plt.subplot(132)
plt.imshow(object_rgb)
plt.title("Extracted Object RGB (Input to SAM3)")
plt.subplot(133)
plt.imshow(handle_mask, cmap='gray')
plt.title(f"All Handles Mask (SAM3, {valid_mask_count} instances)")
plt.suptitle("Press Enter to continue to grasp planning", fontsize=12)
plt.draw()
plt.waitforbuttonpress()
plt.close()

# 可选：合并物体mask和手柄mask（只保留物体中的手柄区域，增强鲁棒性）
final_mask = np.bitwise_and(object_mask, handle_mask)
print(f"最终mask形状: {final_mask.shape}, 有效像素数: {np.sum(final_mask)}")

# 可视化最终mask
plt.figure("Final Mask (Object + All Handles)", figsize=(12, 7))
plt.imshow(final_mask, cmap='gray')
plt.title("Final Mask (Object + All Handles)")
plt.suptitle("Press Enter to start grasping", fontsize=12)
plt.draw()
plt.waitforbuttonpress()
plt.close()
print("open meshcat-server")

# ===================== 6. 相机内参处理 =====================
intrinsic = camera.get_intrinsics_matrix()
fx = float(intrinsic[0, 0])
fy = float(intrinsic[1, 1])
cx = float(intrinsic[0, 2])
cy = float(intrinsic[1, 2])
intrinsic = [fx, fy, cx, cy]  # 适配graspnet输入格式

###########################################################################
# 原有抓取推理和坐标变换逻辑（保持不变，使用final_mask）
###########################################################################
# graspnet_baseline 推理（使用最终的手柄mask进行抓取规划）
from demogen import demo_variable
grasp = demo_variable(rgb_data, depth_data, final_mask, intrinsic)

# 坐标变换工具函数
def get_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T

# -------------------------- 沿抓取方向前移0.1米的函数 --------------------------
def move_along_grasp_dir(htm: np.ndarray, distance: float = 0.1) -> np.ndarray:
    """
    沿抓取位姿的Z轴方向（抓取方向）前移指定距离
    :param htm: 4x4抓取位姿齐次矩阵
    :param distance: 前移距离（米）
    :return: 偏移后的齐次矩阵
    """
    # 提取抓取方向（Z轴）的单位向量（旋转矩阵第3列）
    grasp_dir = htm[:3, 2]
    grasp_dir_unit = grasp_dir / np.linalg.norm(grasp_dir)  # 归一化确保距离精确
    
    # 计算平移增量并叠加
    new_t = htm[:3, 3] + grasp_dir_unit * distance
    
    # 构造新矩阵（旋转不变，仅更新平移）
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

# 3. 坐标变换：相机坐标系→世界坐标系（适配可视化）
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ T_cam_grasp @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# -------------------------- 调用偏移函数，前移0.1米 --------------------------
T_world_grasp = move_along_grasp_dir(T_world_grasp, distance=0.1)
print(f"抓取位姿已沿抓取方向前移0.1米")

grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])

# 确定抓取点和放置点（微调z轴保证抓取稳定性）
banana_position, banana_orientation = grasp_pos, grasp_quat

goal_position = banana_position.copy()
goal_position[0] += 0    # 放置点偏移
goal_position[2] += 0.06  # 提高放置点高度，防止碰撞

print("抓取点xyz: ", banana_position)
print("放置点xyz: ", goal_position)

###########################################################################
# 机械臂抓取控制（保持不变）
###########################################################################
for i in range(1000000):
    # 获取当前机械臂关节位置
    current_joint_positions = franka.get_joint_positions()
    # 计算抓取动作
    actions = controller.forward(
        picking_position=banana_position,
        placing_position=goal_position,
        current_joint_positions=current_joint_positions,
        end_effector_orientation=banana_orientation
    )
    # 执行动作
    franka.apply_action(actions)
    world.step(render=True) 

# 关闭仿真
simulation_app.close()
