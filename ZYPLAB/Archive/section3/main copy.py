from isaacsim import SimulationApp
import omni
import cv2
import numpy as np
from typing import Optional, Tuple

# -------------------------- 仿真初始化（保持原有场景路径）--------------------------
sim_config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "anti_aliasing": False  # 关闭抗锯齿，减少相机数据延迟
}
simulation_app = SimulationApp(sim_config)

from omni.isaac.core.utils.stage import open_stage, get_stage_units
from omni.isaac.core.api import World
from omni.isaac.sensor import Camera
from omni.isaac.core.prims import SingleXFormPrim

# 原有场景文件路径（无需修改）
USD_PATH = r"/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"
CAMERA_PRIM_PATH = "/World/camera"  # 与USD中相机路径一致
SAVE_RGB_PATH = "stable_camera_rgb.png"
SAVE_DEPTH_PATH = "stable_camera_depth.png"

# -------------------------- 核心：稳定相机类（适配5.0版本）--------------------------
class StableCamera:
    def __init__(
        self,
        prim_path: str,
        resolution: Tuple[int, int] = (1280, 720),
        update_rate: float = 30.0,
        max_retries: int = 3,
        min_rgb_mean: float = 20.0  # 最小RGB均值（避免黑屏）
    ):
        self.prim_path = prim_path
        self.resolution = resolution
        self.update_rate = update_rate
        self.max_retries = max_retries
        self.min_rgb_mean = min_rgb_mean
        self.camera: Optional[Camera] = None
        self.stage_units = get_stage_units()  # 获取场景单位（默认米）

    def initialize(self, world: World) -> bool:
        """初始化相机（含版本适配和状态校验）"""
        # 1. 检查相机Prim是否存在且为Camera类型
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(self.prim_path)
        if not prim.IsValid() or not prim.IsA(omni.usd.schema.physx.sensor.Camera):
            print(f"错误：{self.prim_path} 不是有效Camera类型Prim！")
            return False

        # 2. 创建相机并配置（5.0版本必填参数）
        self.camera = Camera(
            prim_path=self.prim_path,
            resolution=self.resolution,
            clipping_range=(0.01, 10.0)  # 调整裁剪范围，避免近距/远距物体无数据
        )
        self.camera.initialize()
        self.camera.set_update_rate(self.update_rate)  # 强制设置更新频率
        self.camera.add_rgb_to_frame()  # 启用RGB缓冲区
        self.camera.add_distance_to_image_plane_to_frame()  # 启用深度缓冲区

        # 3. 5.0版本关键：强制传感器同步（解决数据延迟）
        omni.sensor.get_sensor_interface().sync_sensors()
        world.step()  # 同步后立即步进一步

        print("相机初始化完成，内参矩阵：")
        print(self.camera.get_intrinsics_matrix())
        return True

    def get_valid_data(self, world: World) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取有效RGB和深度数据（含重试机制）"""
        for retry in range(self.max_retries):
            # 强制触发相机更新（5.0版本必须手动调用）
            self.camera.update()
            world.step()  # 等待数据采集完成

            # 获取原始数据
            rgb_data = self.camera.get_rgb()
            depth_data = self.camera.get_depth()  # 单位：米（与场景单位一致）

            # 4. 数据有效性校验（多重保障）
            valid = True
            if rgb_data is None or rgb_data.shape != (*self.resolution[::-1], 3):
                print(f"重试 {retry+1}/{self.max_retries}：RGB数据形状无效")
                valid = False
            elif rgb_data.mean() < self.min_rgb_mean:
                print(f"重试 {retry+1}/{self.max_retries}：RGB过暗（均值={rgb_data.mean():.2f} < {self.min_rgb_mean}）")
                valid = False

            if depth_data is None or depth_data.shape != self.resolution[::-1]:
                print(f"重试 {retry+1}/{self.max_retries}：深度数据形状无效")
                valid = False
            elif np.count_nonzero(depth_data) < self.resolution[0] * self.resolution[1] * 0.1:
                print(f"重试 {retry+1}/{self.max_retries}：有效深度数据不足10%")
                valid = False

            if valid:
                # 深度转换为mm（兼容原有graspnet推理逻辑）
                depth_data_mm = depth_data * 1000.0
                return rgb_data, depth_data_mm

            # 重试间隔：执行100步仿真，等待相机数据流稳定
            for _ in range(100):
                world.step()

        print("错误：多次重试后仍未获取有效相机数据！")
        return None, None

    def save_data(self, rgb_data: np.ndarray, depth_data_mm: np.ndarray) -> None:
        """安全保存图像（含异常处理）"""
        try:
            # 保存RGB（转换为BGR格式，OpenCV兼容）
            rgb_bgr = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            cv2.imwrite(SAVE_RGB_PATH, rgb_bgr)
            print(f"RGB图像已稳定保存至：{SAVE_RGB_PATH}（均值={rgb_data.mean():.2f}）")

            # 保存深度图（归一化到0-255，便于可视化）
            depth_normalized = cv2.normalize(
                depth_data_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            cv2.imwrite(SAVE_DEPTH_PATH, depth_normalized)
            print(f"深度图像已稳定保存至：{SAVE_DEPTH_PATH}（有效像素数={np.count_nonzero(depth_data_mm)}）")
        except Exception as e:
            print(f"保存图像失败：{str(e)}")

# -------------------------- 主流程（保持原有场景和初始化逻辑）--------------------------
def main():
    # 1. 加载场景
    if not open_stage(USD_PATH):
        print(f"错误：无法加载场景文件 {USD_PATH}")
        simulation_app.close()
        return

    # 2. 创建世界并重置
    world = World(stage_units_in_meters=1.0)  # 明确场景单位为米
    world.reset()

    # 3. 初始化稳定相机
    stable_cam = StableCamera(prim_path=CAMERA_PRIM_PATH)
    if not stable_cam.initialize(world):
        simulation_app.close()
        return

    # 4. 延长初始化仿真（1000步，确保相机完全就绪）
    print("相机预热中...")
    for i in range(1000):
        world.step()
        if i % 200 == 0:
            print(f"预热进度：{i}/1000")

    # 5. 稳定获取并保存图像
    rgb_data, depth_data_mm = stable_cam.get_valid_data(world)
    if rgb_data is not None and depth_data_mm is not None:
        stable_cam.save_data(rgb_data, depth_data_mm)

    # -------------------------- 后续：原有抓取逻辑（可直接对接）--------------------------
    print("\n图像获取完成，可继续执行SAM分割和抓取推理...")
    # 此处可直接添加原有抓取代码（使用rgb_data和depth_data_mm作为输入）

    # 保持仿真运行（如需后续操作可保留，仅测试图像获取可注释）
    # while simulation_app.is_running():
    #     world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
