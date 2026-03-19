# grasp_gen_interface.py
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import time
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra

# GraspGen项目内部依赖 (保持原样)
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import (
    point_cloud_outlier_removal,
    depth_and_segmentation_to_point_clouds,
    filter_colliding_grasps,
)
from grasp_gen.robot import get_gripper_info

# ===================== 1. 定义抓取结果封装类 =====================
class Grasp:
    def __init__(
        self,
        pose: np.ndarray,
        score: float,
        collision_free: bool,
        gripper_name: str,
        all_collision_free_grasps: Optional[np.ndarray] = None,
        all_collision_free_scores: Optional[np.ndarray] = None
    ):
        self.pose = pose
        self.score = score
        self.collision_free = collision_free
        self.gripper_name = gripper_name
        self.all_collision_free_grasps = all_collision_free_grasps
        self.all_collision_free_scores = all_collision_free_scores

    def __repr__(self):
        return f"Grasp(score={self.score:.3f}, collision_free={self.collision_free}, gripper={self.gripper_name})"

# ===================== 2. 核心封装函数（必须包含 scene_colors 参数） =====================
def demo_variable(
    # --- 原始输入模式 (可选) ---
    rgb_data: Optional[np.ndarray] = None,
    depth_data: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    intrinsic: Optional[List[float]] = None,
    
    # --- 【重点】确保这里有这些新参数 ---
    scene_pc: Optional[np.ndarray] = None,      # (N, 3) float32
    object_pc: Optional[np.ndarray] = None,     # (M, 3) float32
    scene_colors: Optional[np.ndarray] = None,  # (N, 3) uint8
    object_colors: Optional[np.ndarray] = None, # (M, 3) uint8

    # --- 配置参数 ---
    gripper_config: str = "/home/zyp/GraspGen/models/checkpoints/graspgen_franka_panda.yml",
    grasp_threshold: float = 0.8,
    num_grasps: int = 200,
    return_topk: bool = False,
    topk_num_grasps: int = -1,
    collision_threshold: float = 0,  # 0.005,#####################################################
    max_scene_points: int = 8192,
    visualize: bool = True,
    save_results: bool = True,
    output_file: str = "collision_free_grasps_results.npz"
) -> Grasp:
    """
    核心接口：支持 'RGBD+Mask' 或 '直接点云' 两种输入模式。
    """
    start_time = time.time()
    print(f"Starting grasp detection at {time.strftime('%H:%M:%S')}")

    # 1. 输入模式检查与数据准备
    use_external_pc = (scene_pc is not None) and (object_pc is not None)
    
    if not use_external_pc:
        # --- 模式A: 使用内部逻辑从深度图生成点云 ---
        if depth_data is None or mask is None or intrinsic is None:
            raise ValueError("若未提供点云，必须提供 depth_data, mask 和 intrinsic")
        if len(intrinsic) != 4:
            raise ValueError("intrinsic 必须为 [fx, fy, cx, cy]")
        
        fx, fy, cx, cy = intrinsic
        pc_start = time.time()
        try:
            scene_pc, object_pc, scene_colors, object_colors = depth_and_segmentation_to_point_clouds(
                depth_image=depth_data,
                segmentation_mask=mask,
                fx=fx, fy=fy, cx=cx, cy=cy,
                rgb_image=rgb_data,
                target_object_id=1,
                remove_object_from_scene=True,
            )
        except Exception as e:
            raise ValueError(f"点云生成失败: {str(e)}") from e
        print(f"Point cloud generated internally in {time.time() - pc_start:.2f}s")
    else:
        # --- 模式B: 使用外部传入的点云 ---
        print(f"Using provided external point clouds.")
        # 简单校验
        if object_pc.shape[1] != 3 or scene_pc.shape[1] != 3:
            raise ValueError("点云数据 shape 必须为 (N, 3)")

    # 2. 加载夹爪配置
    if not os.path.exists(gripper_config):
        raise ValueError(f"夹爪配置文件不存在: {gripper_config}")
    grasp_cfg = load_grasp_cfg(gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    gripper_collision_mesh = get_gripper_info(gripper_name).collision_mesh

    # 3. 初始化可视化
    vis = None
    if visualize:
        vis = create_visualizer()

    # 4. 物体点云去噪
    filter_start = time.time()
    object_pc_torch = torch.from_numpy(object_pc.astype(np.float32))
    pc_filtered, pc_removed = point_cloud_outlier_removal(object_pc_torch)
    pc_filtered = pc_filtered.numpy()
    print(f"Filtering took: {time.time() - filter_start:.2f}s (Removed {len(pc_removed)})")

    # 5. 抓取姿态推理 (GraspGen核心)
    inference_start = time.time()
    grasp_sampler = GraspGenSampler(grasp_cfg)
    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc_filtered,
        grasp_sampler,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps if return_topk else -1,
    )
    
    if len(grasps_inferred) == 0:
        raise ValueError("无有效抓取生成，请降低阈值")
        
    # 转CPU numpy
    grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
    grasps_inferred = grasps_inferred.cpu().numpy()
    grasps_inferred[:, 3, 3] = 1
    print(f"Inferred {len(grasps_inferred)} grasps in {time.time() - inference_start:.2f}s")

    # 6. 坐标系中心化处理
    def process_point_cloud(pc, grasps, grasp_conf, pc_colors=None):
        scores = get_color_from_score(grasp_conf, use_255_scale=True)
        T_center = tra.translation_matrix(-pc.mean(axis=0))
        pc_centered = tra.transform_points(pc, T_center)
        grasps_centered = np.array([T_center @ g for g in grasps.tolist()])
        
        pc_colors_centered = pc_colors
        if pc_colors is not None:
            pc_colors_centered = pc_colors.copy().astype(np.float32)
            pc_colors_centered[:, 0] = np.clip(pc_colors_centered[:, 0] * 1.4, 0, 255)
            pc_colors_centered = pc_colors_centered.astype(np.uint8)
        return pc_centered, grasps_centered, scores, T_center, pc_colors_centered

    pc_centered, grasps_centered, scores, T_center, object_colors_viz = process_point_cloud(
        pc_filtered, grasps_inferred, grasp_conf_inferred, object_colors
    )
    scene_pc_centered = tra.transform_points(scene_pc, T_center)
    
    # 7. 碰撞检测
    # 场景下采样
    if len(scene_pc_centered) > max_scene_points:
        idx = np.random.choice(len(scene_pc_centered), max_scene_points, replace=False)
        scene_down = scene_pc_centered[idx]
    else:
        scene_down = scene_pc_centered

    collision_start = time.time()
    collision_free_mask = filter_colliding_grasps(
        scene_pc=scene_down,
        grasp_poses=grasps_centered,
        gripper_collision_mesh=gripper_collision_mesh,
        collision_threshold=collision_threshold,
    )
    
    collision_free_grasps = grasps_centered[collision_free_mask]
    collision_free_scores = grasp_conf_inferred[collision_free_mask]
    print(f"Collision check: {len(collision_free_grasps)} valid grasps in {time.time() - collision_start:.2f}s")

    if len(collision_free_grasps) == 0:
        raise ValueError("无无碰撞抓取！")

    # 8. 选出最优并还原坐标
    best_idx = np.argmax(collision_free_scores)
    best_grasp_centered = collision_free_grasps[best_idx]
    T_restore = tra.inverse_matrix(T_center)
    best_grasp_original = T_restore @ best_grasp_centered # 还原回原始世界坐标

    # 9. 保存结果
    if save_results:
        np.savez(output_file, 
                 best_grasp=best_grasp_original, 
                 object_pc=pc_centered, 
                 scene_pc=scene_pc_centered,
                 camera_intrinsics={"data": "external_pc_used"} if use_external_pc else {"intrinsic": intrinsic})
        print(f"Results saved to {output_file}")

    # 10. MeshCat 可视化
    if visualize and vis is not None:
        # 可视化点云
        if scene_colors is not None:
            visualize_pointcloud(vis, "scene_pc", scene_pc_centered, scene_colors, size=0.002)
        else:
            visualize_pointcloud(vis, "scene_pc", scene_pc_centered, [128,128,128], size=0.002)
            
        if object_colors_viz is not None:
            visualize_pointcloud(vis, "object_pc", pc_centered, object_colors_viz, size=0.0025)
        
        # 可视化Top抓取
        sorted_idx = np.argsort(collision_free_scores)[::-1]
        for i in range(min(50, len(sorted_idx))):
            idx = sorted_idx[i]
            visualize_grasp(vis, f"grasps/top_{i}", collision_free_grasps[idx], [0,255,0], gripper_name)
        
        # 最优抓取（黄色高亮）
        visualize_grasp(vis, "grasps/best_grasp", best_grasp_centered, [255,255,0], gripper_name, linewidth=4)

    return Grasp(
        pose=best_grasp_original,
        score=collision_free_scores[best_idx],
        collision_free=True,
        gripper_name=gripper_name,
        all_collision_free_grasps=collision_free_grasps,
        all_collision_free_scores=collision_free_scores
    )