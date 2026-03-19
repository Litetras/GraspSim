from isaacsim import SimulationApp
import cv2
import numpy as np
import os
from datetime import datetime

# 初始化仿真应用（非无头模式，可看到界面）
simulation_app = SimulationApp({"headless": False})

# 导入ISAAC Sim核心模块
from omni.isaac.core.utils.stage import open_stage
from isaacsim.core.api import World
from omni.isaac.sensor import Camera

# ===================== 核心配置 =====================
# USD场景路径（请根据实际路径修改）
usd_path = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"
# 图片保存根目录（自动创建，避免路径不存在）
SAVE_DIR = "./camera_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# 相机参数配置
CAMERA_PATH = "/World/Camera"  # 场景中相机的Prim路径
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720  # 相机分辨率

# ===================== 加载场景 & 初始化世界 =====================
# 加载USD场景文件
open_stage(usd_path)

# 初始化仿真世界
world = World()

# ===================== 初始化相机传感器 =====================
print("正在初始化相机...")
# 创建相机对象
camera = Camera(
    prim_path=CAMERA_PATH,
    resolution=(CAMERA_WIDTH, CAMERA_HEIGHT)
)
# 初始化相机
camera.initialize()
# 启用RGB和深度数据采集（深度数据为「到图像平面的距离」，单位：米）
camera.add_rgb_to_frame()
camera.add_distance_to_image_plane_to_frame()

# ===================== 仿真预热（让场景稳定） =====================
print("仿真预热中（100步）...")
world.reset()
for _ in range(100):
    world.step()  # 执行仿真步，让场景物体稳定

# ===================== 采集并保存相机数据 =====================
print("开始采集相机数据...")
# 获取RGB数据（格式：HWC，RGB通道，uint8类型）
rgb_data = camera.get_rgb()
# 获取深度数据（格式：HW，float32类型，单位：米，原始高精度值）
depth_data = camera.get_depth()

# 生成时间戳（避免文件名重复，格式：年月日_时分秒）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. 保存RGB图片（转换为OpenCV的BGR格式后保存为PNG）
rgb_save_path = os.path.join(SAVE_DIR, f"rgb_20251213_235101.png")
rgb_bgr = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)  # RGB→BGR（OpenCV默认格式）
cv2.imwrite(rgb_save_path, rgb_bgr)
print(f"✅ RGB图片已保存：{rgb_save_path}")

# 2. 保存深度数据（原始float32格式，.npy文件，保留米单位高精度值）
depth_npy_path = os.path.join(SAVE_DIR, f"depth_20251213_235101.npy")
np.save(depth_npy_path, depth_data)
print(f"✅ 深度数据已保存（NPY格式，米为单位）：{depth_npy_path}")

# ===================== 打印数据信息（供调试） =====================
print("\n📊 数据信息：")
print(f"RGB数据 - 形状：{rgb_data.shape} | 数据类型：{rgb_data.dtype}")
print(f"深度数据 - 形状：{depth_data.shape} | 数据类型：{depth_data.dtype}")
print(f"深度值范围：{np.min(depth_data[depth_data>0]):.4f} ~ {np.max(depth_data):.4f} 米")  # 排除无效0值
print(f"相机内参矩阵：\n{camera.get_intrinsics_matrix()}")

# ===================== 关闭仿真 =====================
print("\n所有数据保存完成，即将关闭仿真...")
simulation_app.close()
