# Z_cgn_worker_pt.py
import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d  # 确保环境中有 open3d

# ===================== PyTorch 显存优化 =====================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ===================== 路径配置 =====================
CGN_REPO_ROOT = r"/home/zyp/pan1/contact_graspnet_pytorch"
if CGN_REPO_ROOT not in sys.path:
    sys.path.append(CGN_REPO_ROOT)

# 2. 🌟 关键：把内部源码目录也加进去 (为了让里面的文件能互相 import mesh_utils)
CGN_SRC_DIR = os.path.join(CGN_REPO_ROOT, "contact_graspnet_pytorch")
if CGN_SRC_DIR not in sys.path:
    sys.path.append(CGN_SRC_DIR)



try:
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch.checkpoints import CheckpointIO 
    # 🌟 关键：导入你刚才展示的那个 Open3D 可视化工具
    from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps
    print("✅ CGN 模块及 Open3D 可视化工具导入成功", flush=True)
except ImportError as e:
    print(f"❌ 导入失败！请检查路径或是否安装了 open3d: {e}", flush=True)
    raise e

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def run_inference(data_path, out_path):
    print("[PT-Worker] 1. 读取数据...", flush=True)
    data = np.load(data_path)
    depth = data['depth']
    cam_K = data['K']
    segmap = data['segmap']
    rgb = data['rgb'] if 'rgb' in data else None

    ckpt_dir = os.path.join(CGN_REPO_ROOT, "checkpoints/contact_graspnet")
    global_config = config_utils.load_config(ckpt_dir, batch_size=1)
    
    # 初始化模型
    grasp_estimator = GraspEstimator(global_config)
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(ckpt_dir, 'checkpoints'), model=grasp_estimator.model)
    checkpoint_io.load('model.pt')
    
    print("[PT-Worker] 2. 提取点云...", flush=True)
    # 这里会生成带颜色的点云（如果 rgb 不为空）
    pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
        depth, cam_K, segmap=segmap, rgb=rgb, z_range=[0.01, 2.0]
    )

    print("[PT-Worker] 3. 执行推理...", flush=True)
    with torch.no_grad():
        pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(
            pc_full, 
            pc_segments=pc_segments, 
            local_regions=True, 
            filter_grasps=True, 
            forward_passes=5############################
        )

    # 4. 后处理与过滤
    obj_id = 1
    if obj_id in pred_grasps_cam and len(pred_grasps_cam[obj_id]) > 0:
        raw_grasps = pred_grasps_cam[obj_id]
        raw_scores = scores[obj_id]
        raw_contact_pts = contact_pts[obj_id]
        raw_openings = gripper_openings[obj_id]
        
        # 距离过滤逻辑 (保留靠近物体的抓取)
        knife_points = pc_segments[obj_id]
        valid_idxs = []
        for i, c_pt in enumerate(raw_contact_pts):
            dist = np.min(np.linalg.norm(knife_points - c_pt, axis=1))
            if dist < 0.15: # 稍微放宽到 10cm  # <--- 放宽这个距离限制
                valid_idxs.append(i)
        
        if len(valid_idxs) > 0:
            valid_idxs = np.array(valid_idxs)
            final_grasps = raw_grasps[valid_idxs]
            final_scores = raw_scores[valid_idxs]
            final_openings = raw_openings[valid_idxs]

            best_idx = np.argmax(final_scores)
            
            # 🌟 核心修改：调用 Open3D 进行可视化
            print("\n>>> [Open3D] 正在弹出 3D 可视化窗口...", flush=True)
            print(">>> 提示：在窗口中按 'q' 键关闭窗口并继续执行 Isaac Sim 动作。", flush=True)
            
            # 构造可视化函数需要的格式
            vis_grasps = {obj_id: final_grasps}
            vis_scores = {obj_id: final_scores}
            vis_openings = {obj_id: final_openings}

            visualize_grasps(
                pc_full, 
                vis_grasps, 
                vis_scores, 
                plot_opencv_cam=True, 
                pc_colors=pc_colors,
                gripper_openings=vis_openings
            )

            # 保存结果返回给 Isaac Sim
            np.savez(out_path, best_grasp=final_grasps[best_idx], score=final_scores[best_idx], success=True)
        else:
            print("❌ 过滤后无有效抓取点", flush=True)
            np.savez(out_path, success=False)
    else:
        print("❌ 未生成抓取", flush=True)
        np.savez(out_path, success=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_data', type=str, required=True)
    parser.add_argument('--out_data', type=str, required=True)
    args = parser.parse_args()
    run_inference(args.in_data, args.out_data)