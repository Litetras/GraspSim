# run_eval_baseline2.py
import subprocess
import time
import os

TOTAL_TRIALS = 28  # 总共跑 28 次 (7个视角循环4次)
# 👇 请确保这里的路径指向你修改后的 Isaac Sim Worker 脚本
WORKER_SCRIPT = "/home/zyp/IsaacLab/ZYPLAB/Graspgen/Baseline2/Z_eval_worker_baseline2.py" 
ISAAC_PYTHON = "/home/zyp/anaconda3/envs/sam3_gen/bin/python"

# 默认的任务指令
INSTRUCTION = "grasp the knife to cut"

print(f"🚀 开始执行 Baseline2 自动化抓取评测，总计 {TOTAL_TRIALS} 轮...")

for i in range(TOTAL_TRIALS):
    # 计算当前应该用哪个相机视角 (1 到 7 循环)
    cam_id = (i % 7) + 1 
    
    print(f"\n" + "="*50)
    print(f"🔄 正在启动第 {i}/{TOTAL_TRIALS-1} 轮... (当前使用 cam{cam_id}.usd)")
    print(f"==================================================")
    
    start_time = time.time()
    
    try:
        # 启动 Isaac Sim 工作脚本，传入当前 trial 轮次、cam_id 和 指令
        result = subprocess.run(
            [ISAAC_PYTHON, WORKER_SCRIPT, 
             "--trial", str(i), 
             "--cam_id", str(cam_id),
             "--instruction", INSTRUCTION]
        )
        
        if result.returncode == 0:
            print(f"✅ 第 {i} 轮成功执行完毕 (耗时: {time.time() - start_time:.1f}s)")
        else:
            print(f"⚠️ 第 {i} 轮执行失败/报错退出 (Code: {result.returncode})")
            
    except Exception as e:
        print(f"❌ 运行报错: {e}")
        
    print("⏳ 等待操作系统回收显存 (8秒)...")
    time.sleep(8)
    
    # 暴力清理可能卡死的 Isaac Sim 残留进程
    os.system("pkill -9 -f omni.kit.app") 

print("\n🎉 全部评测执行完毕！")