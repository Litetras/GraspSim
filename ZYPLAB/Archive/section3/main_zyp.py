from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
# #####################################################
# import torch
# torch.backends.cudnn.enabled = False
# ################################################

usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"  # 必须绝对路径
open_stage(usd_path)
world = World()

franka:Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) 
controller = PickPlaceController(
            name="pick_place_controller",
            gripper=franka.gripper,
            robot_articulation=franka,
            end_effector_initial_height=0.3,#######################################
            events_dt=[0.008, 0.005, 1, 0.01, 0.05, 0.05, 0.0025, 1, 0.008, 0.08], 
        )

##########################################################################################################################  新增相机
# 这里更新了usd中相机的位姿，垂直正对着物体，graspnet_base输出的结果会好一些
from omni.isaac.sensor import Camera
camera_path = "/World/Camera"            # 和usd创建的相机路径一致
camera_width, camera_height = 1280, 720  # 设置相机分辨率 和 Graspnet baseline算法一致
camera = Camera(prim_path=camera_path, resolution=(camera_width, camera_height))
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
camera.add_rgb_to_frame()
print("相机初始化完成")
print("获取相机内参: \n", camera.get_intrinsics_matrix())
##########################################################################################################################  新增相机


world.reset()
for i in range(100): # 先执行100步仿真，得到物体掉落后稳定
    world.step()
franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)


########################################################################################################################## 新增获取抓取点
import sys
sys.path.append(r'/home/zyp/IsaacLab/ZYPLAB/section2/graspnet-baseline')
from demo import demo_variable
from graspnetAPI import  Grasp
import numpy as np
import cv2
import matplotlib.pyplot as plt  # 新增：导入matplotlib

##### 前处理: 数据准备
rgb_data = camera.get_rgb()
depth_data = camera.get_depth()*1000 # 转换为mm
print(depth_data)
from ultralytics import SAM # 导入SAM模型
sam_model = SAM(r"sam2.1_b.pt")  # 加载SAM模型 会自动下载权重
points = []

# ========== 替换：用Matplotlib实现鼠标点击选点 ==========
def onclick(event):
    global points
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        print(f"点击位置: ({x}, {y})")
        points.append([x, y, 1])
        # 绘制点击标记
        plt.scatter(x, y, c='red', s=50, marker='x')
        plt.draw()

# 创建Matplotlib窗口并绑定鼠标事件
plt.figure("RGB Image (Click to select points)", figsize=(12, 7))
plt.imshow(rgb_data)
plt.title("Enter")
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.waitforbuttonpress()  # 按Enter键退出窗口
plt.close()

kwargs = {
    'points': np.array([(x, y) for x, y, _ in points], dtype=np.float32),
    'labels': np.array([label for _, _, label in points], dtype=np.int32),
}
results = sam_model(rgb_data,** kwargs)

# ========== 替换：用Matplotlib显示Mask ==========
if results[0].masks is not None and len(results[0].masks.data) > 0:
    mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
    plt.figure("Segmentation Mask")
    plt.imshow(mask, cmap='gray')
    plt.title("Enter")
    plt.waitforbuttonpress()
    plt.close()
else:
    print("No mask detected.")

intrinsic = camera.get_intrinsics_matrix()

##### graspnet_baseline 推理
detected_grasp:Grasp = demo_variable(rgb_data, depth_data, mask, intrinsic)#########################################

##### 后处理：坐标变换
def get_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat

# 1. 获取相机在世界坐标系中的位姿    
cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose() 
T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

# 2. 构造相机坐标系下的抓取位姿
T_cam_grasp = get_T(detected_grasp.translation, detected_grasp.rotation_matrix)          

# 3. 将相机坐标系下的抓取位姿转换到世界坐标系并应用到可视化坐标系
T_world_grasp = T_world_cam @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ T_cam_grasp @ get_T([0, 0, 0], [[0, 0, 1], [0, -1, 0], [1, 0, 0]]) 
grasp_pos = T_world_grasp[:3, 3]
grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])

# 可视化抓取位姿
vis_prim_path = "/World/GraspVisualization"
grasp_vis = SingleXFormPrim(prim_path=vis_prim_path, name="grasp_visualization")
grasp_vis.set_world_pose(position=grasp_pos, orientation=grasp_quat) 

########################################################################################################################## 新增获取抓取点

# 确定抓取点和放置点
banana_position, banana_orientation = grasp_pos, grasp_quat
banana_position[2] -= 0.025#################################
goal_position = banana_position.copy()
goal_position[0] += 0.2
goal_position[2] += 0.05
print("抓取点xyz: ", banana_position)
print("放置点xyz: ", goal_position)

for i in range(1000000):
    # 获取当前机械臂关节位置
    current_joint_positions = franka.get_joint_positions()
    # 计算需要执行的动作
    actions = controller.forward(
            picking_position=banana_position,
            placing_position=goal_position,
            current_joint_positions=current_joint_positions,
            end_effector_orientation=banana_orientation # 抓取点的夹爪姿态
    )
    # 执行动作
    franka.apply_action(actions)
    
    world.step(render=True) 
simulation_app.close()     
