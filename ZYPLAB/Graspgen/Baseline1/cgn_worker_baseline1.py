import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d  

# ===================== PyTorch 显存优化 =====================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ===================== 路径配置 =====================
CGN_REPO_ROOT = r"/home/zyp/pan1/contact_graspnet_pytorch"
if CGN_REPO_ROOT not in sys.path:
    sys.path.append(CGN_REPO_ROOT)

CGN_SRC_DIR = os.path.join(CGN_REPO_ROOT, "contact_graspnet_pytorch")
if CGN_SRC_DIR not in sys.path:
    sys.path.append(CGN_SRC_DIR)

try:
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch.checkpoints import CheckpointIO 
    print("✅ CGN 模块导入成功", flush=True)
except ImportError as e:
    print(f"❌ 导入失败！请检查路径或是否安装了 open3d: {e}", flush=True)
    raise e

torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# ===================== 🌟 核心修复：强制矩阵正交化 =====================
def enforce_orthogonal_grasps(grasps):
    """
    通过施密特正交化，修复神经网络输出的畸变旋转矩阵
    grasps: shape (N, 4, 4)
    """
    fixed_grasps = np.copy(grasps)
    for i in range(len(fixed_grasps)):
        R = fixed_grasps[i, :3, :3]
        
        # 提取网络预测的原始坐标轴
        x_raw = R[:, 0]
        z_raw = R[:, 2] # 进近方向，最重要，必须保留
        
        # 1. 保留 Z 轴并归一化
        z_new = z_raw / np.linalg.norm(z_raw)
        
        # 2. 叉乘获得完美垂直的 Y 轴 (Z x X = Y)
        y_new = np.cross(z_new, x_raw)
        y_norm = np.linalg.norm(y_new)
        
        # 防止共线的极端情况（极少发生）
        if y_norm < 1e-6: 
            fallback_x = np.array([1.0, 0.0, 0.0]) if abs(z_new[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            y_new = np.cross(z_new, fallback_x)
            y_norm = np.linalg.norm(y_new)
            
        y_new = y_new / y_norm
        
        # 3. 叉乘获得完美垂直的 X 轴 (Y x Z = X)
        x_new = np.cross(y_new, z_new)
        
        # 重新填入矩阵
        fixed_grasps[i, :3, 0] = x_new
        fixed_grasps[i, :3, 1] = y_new
        fixed_grasps[i, :3, 2] = z_new
        
    return fixed_grasps
# =======================================================================

def run_inference(data_path, out_path, vis_path=None):
    print("[PT-Worker] 1. 读取数据...", flush=True)
    data = np.load(data_path)
    depth = data['depth']
    cam_K = data['K']
    segmap = data['segmap']
    rgb = data['rgb'] if 'rgb' in data else None

    ckpt_dir = os.path.join(CGN_REPO_ROOT, "checkpoints/contact_graspnet")
    global_config = config_utils.load_config(ckpt_dir, batch_size=1)
    
    grasp_estimator = GraspEstimator(global_config)
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(ckpt_dir, 'checkpoints'), model=grasp_estimator.model)
    checkpoint_io.load('model.pt')
    
    print("[PT-Worker] 2. 提取点云...", flush=True)
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
            forward_passes=3
        )

    obj_id = 1
    if obj_id in pred_grasps_cam and len(pred_grasps_cam[obj_id]) > 0:
        raw_grasps = pred_grasps_cam[obj_id]
        raw_scores = scores[obj_id]
        raw_contact_pts = contact_pts[obj_id]
        raw_openings = gripper_openings[obj_id]
        
        knife_points = pc_segments[obj_id]
        valid_idxs = []
        for i, c_pt in enumerate(raw_contact_pts):
            dist = np.min(np.linalg.norm(knife_points - c_pt, axis=1))
            if dist < 0.15: 
                valid_idxs.append(i)
        
        if len(valid_idxs) > 0:
            valid_idxs = np.array(valid_idxs)
            final_grasps = raw_grasps[valid_idxs]
            final_scores = raw_scores[valid_idxs]
            final_openings = raw_openings[valid_idxs]

            # 🛠️ 就在这里！将过滤后的抓取强制转化为完美的正交矩阵！
            final_grasps = enforce_orthogonal_grasps(final_grasps)

            best_idx = np.argmax(final_scores)

            if vis_path:
                print(f">>> [PT-Worker] 正在打包离线可视化数据至: {vis_path}", flush=True)
                np.savez_compressed(
                    vis_path,
                    pc_full=pc_full,
                    pc_colors=pc_colors,
                    grasps=final_grasps,
                    scores=final_scores,
                    openings=final_openings,
                    best_idx=best_idx 
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
    parser.add_argument('--vis_data', type=str, default=None)
    args = parser.parse_args()
    
    run_inference(args.in_data, args.out_data, args.vis_data)