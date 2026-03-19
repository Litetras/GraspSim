# main_sam_grasp.py
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import SAM
import cv2
import sys

# 假设 serve.utils 在您的环境中可用
from serve.utils import get_point_cloud_from_rgbd 

# 核心修改：导入接口
from OR_grasp_gen_interface import demo_variable

# 全局变量
points = []

def main():
    # 1. 路径配置 (保持您的路径)
    image_path = "/home/zyp/SoFar/assets/open6dor.png"
    depth_path = "/home/zyp/SoFar/assets/open6dor.npy"

    # 2. 加载数据
    image = Image.open(image_path).convert("RGB")
    depth = np.load(depth_path)
    rgb_np = np.array(image)
    H, W = rgb_np.shape[:2]

    # 3. 相机参数
    vinvs = np.array([
        [0., 1., 0., 0.], 
        [-0.9028605, -0., 0.42993355, -0.], 
        [0.42993355, -0., 0.9028605, -0.], 
        [1., 0., 1.2, 1.]
    ])
    projs = [
        [1.7320507, 0., 0., 0.], 
        [0., 2.5980759, 0., 0.], 
        [0., 0., 0., -1.], 
        [0., 0., 0.05, 0.]
    ]

    # 4. 生成完整场景点云
    print("Generating full point cloud using custom logic...")
    pcd_raw = get_point_cloud_from_rgbd(depth, rgb_np, vinvs, projs)
    if torch.is_tensor(pcd_raw):
        pcd_raw = pcd_raw.cpu().numpy()
    
    full_pc_xyz = pcd_raw[:, :3].astype(np.float32)
    full_pc_rgb = pcd_raw[:, 3:6].astype(np.uint8)

    # 5. 交互式选点
    print("="*50)
    print("Initializing SAM model...")
    sam_model = SAM(r"sam2.1_b.pt")
    
    global points
    points.clear()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(rgb_np)
    ax.set_title("CLICK TARGET -> CLOSE WINDOW", color="red", fontweight="bold")
    ax.axis('off')
    
    def onclick(event):
        global points
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            points.append([x, y, 1])
            ax.scatter(x, y, c='red', marker='x', s=100)
            fig.canvas.draw()
            print(f"Selected: ({x}, {y})")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # 6. 处理SAM分割
    object_pc_xyz = None
    object_pc_rgb = None
    scene_pc_xyz = None
    scene_pc_rgb = None

    if len(points) > 0:
        print(f"Segmenting with {len(points)} points...")
        sam_kwargs = {
            'points': np.array([(x, y) for x, y, _ in points], dtype=np.float32),
            'labels': np.array([l for _, _, l in points], dtype=np.int32),
        }
        results = sam_model(rgb_np, **sam_kwargs)
        
        if results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            mask_flat = mask.reshape(-1)
            object_indices = (mask_flat == 1)
            object_pc_xyz = full_pc_xyz[object_indices]
            object_pc_rgb = full_pc_rgb[object_indices]
            
            scene_indices = (mask_flat == 0)
            scene_pc_xyz = full_pc_xyz[scene_indices]
            scene_pc_rgb = full_pc_rgb[scene_indices]
            
            print(f"Object points: {len(object_pc_xyz)}, Scene points: {len(scene_pc_xyz)}")
        else:
            print("Segmentation failed.")
            return
    else:
        print("No points selected.")
        return

    # ===================== 7. 调用抓取接口 (包含方向向量) =====================
    print("\n" + "="*50)
    print("🚀 Passing segmented point clouds to GraspGen with Orientation Filter...")
    
    # 【核心新增】定义目标向量 (Blade orientation)
    blade_vector = [0.27247822284698486, 0.8750270009040833, 0.40010401606559753]
    
    try:
        grasp_result = demo_variable(
            scene_pc=scene_pc_xyz,
            object_pc=object_pc_xyz,
            scene_colors=scene_pc_rgb,
            object_colors=object_pc_rgb,
            
            # --- 传入方向引导 ---
            target_vector=blade_vector,
            angle_threshold=30.0,  # 允许30度误差，太严可能会把所有抓取滤掉
            
            visualize=True,
            save_results=True,
            output_file="sam_grasp_results.npz"
        )

        print("\n===== SUCCESSS: Optimal Guided Grasp Found =====")
        print(f"Score: {grasp_result.score:.4f}")
        print(f"Pose Matrix:\n{grasp_result.pose}")
        
    except Exception as e:
        print(f"Grasp generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()