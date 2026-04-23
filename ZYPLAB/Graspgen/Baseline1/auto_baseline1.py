import subprocess
import time
import os

TOTAL_TRIALS = 98  # 总共跑 14 次
WORKER_SCRIPT = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Baseline1/Z_eval_worker_baseline1.py" 
ISAAC_PYTHON = "/home/zyp/anaconda3/envs/sam3_gen/bin/python"

print(f"🚀 开始执行自动化抓取评测，总计 {TOTAL_TRIALS} 轮...")

for i in range(TOTAL_TRIALS):
    # 🌟 核心：计算当前应该用哪个相机视角 (1 到 7 循环)
    cam_id = (i % 7) + 1 
    
    print(f"\n=====================================")
    print(f"🔄 正在启动第 {i}/{TOTAL_TRIALS-1} 轮... (当前使用 cam{cam_id}.usd)")
    print(f"=====================================")
    
    start_time = time.time()
    
    try:
        # 🌟 核心修复：这里应该是启动 Isaac Sim 的主控脚本，而不是直接调后端的 CGN
        result = subprocess.run(
            [ISAAC_PYTHON, WORKER_SCRIPT, "--trial", str(i), "--cam_id", str(cam_id)]
        )
        
        if result.returncode == 0:
            print(f"✅ 第 {i} 轮成功执行完毕 (耗时: {time.time() - start_time:.1f}s)")
        else:
            print(f"⚠️ 第 {i} 轮执行失败/报错退出 (Code: {result.returncode})")
            
    except Exception as e:
        print(f"❌ 运行报错: {e}")
        
    print("⏳ 等待操作系统回收显存 (8秒)...")
    time.sleep(8)
    
    os.system("pkill -9 -f omni.kit.app") 

print("\n🎉 全部评测执行完毕！请检查 eval_results 文件夹查看最终截图。")