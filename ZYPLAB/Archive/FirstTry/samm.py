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

# ===================== 核心修改：导入第一个代码的接口 =====================
# 假设第一个文件名叫 grasp_gen_interface.py
from grasp_gen_interface import demo_variable

# 全局变量
points = []

def main():
    # 1. 路径配置
    image_path = "/home/zyp/SoFar/assets/open6dor.png"
    depth_path = "/home/zyp/SoFar/assets/open6dor.npy"

    # 2. 加载数据
    image = Image.open(image_path).convert("RGB")
    depth = np.load(depth_path)
    rgb_np = np.array(image)
    H, W = rgb_np.shape[:2]

    # 3. 您的自定义相机参数 (保持不变)
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

    # 4. 生成完整场景点云 (使用您的逻辑)
    print("Generating full point cloud using custom logic...")
    pcd_raw = get_point_cloud_from_rgbd(depth, rgb_np, vinvs, projs)
    if torch.is_tensor(pcd_raw):
        pcd_raw = pcd_raw.cpu().numpy()
    
    # pcd_raw shape is usually (N, 6) -> XYZRGB
    full_pc_xyz = pcd_raw[:, :3].astype(np.float32)
    full_pc_rgb = pcd_raw[:, 3:6].astype(np.uint8)

    # 5. 交互式选点 (保持您的SAM逻辑)
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
    plt.show() # 等待窗口关闭

    # 6. 处理SAM分割结果
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
            # 获取Mask (H, W)
            mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # --- 关键步骤：利用Mask拆分物体点云和场景点云 ---
            mask_flat = mask.reshape(-1)
            
            # 提取物体点云
            object_indices = (mask_flat == 1)
            object_pc_xyz = full_pc_xyz[object_indices]
            object_pc_rgb = full_pc_rgb[object_indices]
            
            # 提取场景点云 (排除物体本身，作为避障环境)
            # 注意：GraspGen通常希望场景点云不包含目标物体，以防把自己当成障碍物
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

    # ===================== 7. 调用抓取接口 =====================
    print("\n" + "="*50)
    print("🚀 Passing segmented point clouds to GraspGen...")
    
    try:
        # 调用第一个文件中的函数
        # 注意：这里不再传 depth_data/intrinsic，而是直接传处理好的点云
        grasp_result = demo_variable(
            scene_pc=scene_pc_xyz,
            object_pc=object_pc_xyz,
            scene_colors=scene_pc_rgb,
            object_colors=object_pc_rgb,
            visualize=True,         # 开启MeshCat可视化
            save_results=True,
            output_file="sam_grasp_results.npz"
        )

        print("\n===== SUCCESSS: Optimal Grasp Found =====")
        print(f"Score: {grasp_result.score:.4f}")
        print(f"Pose Matrix:\n{grasp_result.pose}")
        
    except Exception as e:
        print(f"Grasp generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()