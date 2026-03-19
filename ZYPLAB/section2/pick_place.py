'''pick_place.py'''

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"# 场景文件路径
open_stage(usd_path)# 打开场景文件
world = World()# 创建仿真世界
banana = SingleXFormPrim(name="banana", prim_path="/World/banana")# 获取香蕉物体

# 创建机器人Articulation和PickPlaceController
franka:Franka = world.scene.add(Franka(prim_path="/Franka", name="franka")) # 创建Franka机械臂
controller = PickPlaceController(
            name="pick_place_controller",
            gripper=franka.gripper,
            robot_articulation=franka,
            end_effector_initial_height=0.41633,# 末端执行器初始高度
            events_dt=[0.008, 0.005, 1, 0.01, 0.05, 0.05, 0.0025, 1, 0.008, 0.08], # 各个阶段的时间间隔
        )

world.reset()
for i in range(50): # 先执行50步仿真，得到物体掉落后稳定
    world.step()

franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

# 确定抓取点和放置点
banana_position, banana_orientation = banana.get_world_pose()# 获取香蕉物体的世界坐标和朝向
# 抓取点在香蕉物体上方0.01米处
banana_position[2] += 0.01
goal_position = banana_position.copy()
goal_position[0] += 0.2# 放置点在抓取点前方0.2米处
goal_position[2] += 0.05# 放置点在地面上方0.05米处
print(banana_position)
print(goal_position)

for i in range(1000000):
    # 获取当前机械臂关节位置
    current_joint_positions = franka.get_joint_positions()
    # 计算需要执行的动作
    actions = controller.forward(#
            picking_position=banana_position,# 抓取位置
            placing_position=goal_position,# 放置位置
            current_joint_positions=current_joint_positions,# 当前关节位置
    )
    # 执行动作
    franka.apply_action(actions)
    
    world.step(render=True) 
simulation_app.close()     






