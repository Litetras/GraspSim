import argparse
import numpy as np
import cv2
import os
import sys

# ===================== 核心修复在这里 =====================
# 用 insert(0, ...) 把 ShapeGrasp 的路径插到第一顺位！
# 这样就不会和 Python 自带的 code 模块起冲突了
sys.path.insert(0, "/home/zyp/pan1/ShapeGrasp")

from demo import run_pipeline  
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data", type=str, required=True)
    parser.add_argument("--out_data", type=str, required=True)
    args = parser.parse_args()

    # 1. 加载 Isaac Sim 传过来的数据
    data = np.load(args.in_data)
    rgb = data['rgb']
    depth = data['depth']
    mask = data['segmap']
    
    # 2. 强制切换工作目录到 ShapeGrasp，确保它能正确读取和生成内部的 data/ outputs/ 文件
    os.chdir("/home/zyp/pan1/ShapeGrasp")
    
    os.makedirs("data", exist_ok=True)
    obj_name = "knife"###############################
    
    # 保存 RGB (OpenCV 默认 BGR，需要转换一下)
    cv2.imwrite(f"data/{obj_name}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    # 保存 Mask (必须是 0 和 255 的图像)
    cv2.imwrite(f"data/{obj_name}_mask.png", (mask * 255).astype(np.uint8))
    # 保存 Depth
    np.save(f"data/{obj_name}_depth.npy", depth)

    try:
        grasp_pose, _, _ = run_pipeline(
            obj=obj_name, 
            task_string="pick up the knife to cit ",#################################### 
            data_dir="data/", 
            iter="", 
            output_idx=1, 
            mode="2d", 
            threshold=0.2, 
            no_object=False, 
            model="gpt4o", 
            eps=0.02
        )
        
        np.savez(args.out_data, success=True, grasp_2d=grasp_pose)
        print(f"ShapeGrasp 推理成功: {grasp_pose}")
        
    except Exception as e:
        print(f"ShapeGrasp 推理失败: {e}")
        np.savez(args.out_data, success=False)

if __name__ == "__main__":
    main()