

# 不要from isaacsim.simulation_app import SimulationApp 
# isaacsim.模块 是软件内部的api，python解释器会报错说找不到。
# 如下这两行必须在导入模块之前就导入执行
from isaacsim import SimulationApp 
simulation_app = SimulationApp({"headless": False}) 
from isaacsim.core.api import World

# ====================== 仅新增这部分（打开指定USD） ======================
import omni.usd  # 新增：USD上下文管理（仅添加，不修改原有代码）
# 替换为你的USD文件绝对路径（Linux格式）
USD_FILE_PATH = "/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib/grasp.usd"  # 需手动修改！
# 打开指定USD文件（替换整个场景）
success = omni.usd.get_context().open_stage(USD_FILE_PATH)
if not success:
    print(f"打开USD失败：文件路径错误或文件损坏 → {USD_FILE_PATH}")
    simulation_app.close()
    exit(1)
# =======================================================================

world = World()
world.reset()

for i in range(5000000):
    world.step(render=True) # execute one physics step and one rendering step
simulation_app.close()      # close Isaac Sim
