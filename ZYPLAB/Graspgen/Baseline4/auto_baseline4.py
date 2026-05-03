import subprocess
import time
import os

# ===================== 评测配置 =====================
TOTAL_TRIALS = 98  # 总共跑 98 次 (14 个循环 * 7 个视角)

# 这里指向我们刚刚写好的、融合了 ShapeGrasp 的仿真主代码！
# 假设你把它命名为了 main_sg.py
MAIN_SIM_SCRIPT = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Baseline4/shapegrasp.py" 

# 运行仿真主代码的 Python 解释器 (自带 Isaac Sim 环境的那个)
ISAAC_PYTHON = "/home/zyp/anaconda3/envs/sam3_gen/bin/python"

print(f"🚀 开始执行 ShapeGrasp 自动化抓取评测，总计 {TOTAL_TRIALS} 轮...")

for i in range(TOTAL_TRIALS):
    # 🌟 计算当前应该用哪个相机视角 (1 到 7 循环)
    cam_id = (i % 7) + 1 
    
    print(f"\n" + "="*45)
    print(f"🔄 正在启动第 {i}/{TOTAL_TRIALS-1} 轮... (当前使用 cam{cam_id}.usd)")
    print("="*45)
    
    start_time = time.time()
    
    try:
        # 调用仿真主程序，传入 trial 编号和 cam_id
        result = subprocess.run(
            [ISAAC_PYTHON, MAIN_SIM_SCRIPT, "--trial", str(i), "--cam_id", str(cam_id)]
        )
        
        if result.returncode == 0:
            print(f"✅ 第 {i} 轮成功执行完毕 (耗时: {time.time() - start_time:.1f}s)")
        else:
            print(f"⚠️ 第 {i} 轮执行失败/报错退出 (Code: {result.returncode})")
            
    except Exception as e:
        print(f"❌ 运行报错: {e}")
        
    # 每跑完一轮，彻底清理残留进程，释放宝贵的显存
    print("⏳ 强制清理 Omniverse 残留进程，等待操作系统回收显存 (8秒)...")
    os.system("pkill -9 -f omni.kit.app") 
    time.sleep(8)
    
print("\n🎉 ShapeGrasp 全部评测执行完毕！请检查 eval_results 文件夹查看最终截图和结果。")