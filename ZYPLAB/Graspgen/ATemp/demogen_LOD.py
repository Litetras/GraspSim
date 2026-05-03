# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
封装后接口说明 (LOD 端到端语言条件版本)：
核心函数 demo_variable(rgb_data, depth_data, mask, intrinsic, **kwargs)
"""
import os
import time
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import trimesh.transformations as tra

# 【关键修改】：这里导入了我们即将新建的 grasp_server_LOD
from grasp_gen.grasp_server_LOD import GraspGenSampler, load_grasp_cfg
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

# ===================== 2. 核心封装函数（对外接口） =====================
def demo_variable(
    rgb_data: Optional[np.ndarray],    
    depth_data: np.ndarray,           
    mask: np.ndarray,                 
    intrinsic: List[float],           
    # 【关键修改】：接收从 Isaac Sim 传进来的两个 text
    natural_text: List[str] = None,   
    strict_text: List[str] = None,    
    gripper_config: str = "/home/zyp/pan1/#LODGrasp核心权重/zyp_dataset7teacher/tutorial/models/tutorial_model_config.yaml",
    grasp_threshold: float = 0.8,
    num_grasps: int = 300,##########
    return_topk: bool = True,
    topk_num_grasps: int = 200,
    collision_threshold: float = 0.009,
    max_scene_points: int = 8192,
    visualize: bool = True,
    save_results: bool = False,
    output_file: str = "collision_free_grasps_results.npz"
) -> Grasp:

    start_time = time.time()
    print(f"Starting collision-free grasp detection at {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    if return_topk and topk_num_grasps == -1:
        topk_num_grasps = 100

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
    pc_creation_time = time.time() - pc_start

    grasp_cfg = load_grasp_cfg(gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    gripper_info = get_gripper_info(gripper_name)
    gripper_collision_mesh = gripper_info.collision_mesh

    vis = None
    if visualize:
        vis = create_visualizer()

    filter_start = time.time()
    object_pc_torch = torch.from_numpy(object_pc)
    pc_filtered, pc_removed = point_cloud_outlier_removal(object_pc_torch)
    pc_filtered = pc_filtered.numpy()
    pc_removed = pc_removed.numpy()
    filter_time = time.time() - filter_start

    NUM_TARGET_POINTS = 4096#2048
    if len(pc_filtered) == 0:
        raise ValueError("去离群点后点云为空，无法抓取！")
        
    if len(pc_filtered) > NUM_TARGET_POINTS:
        indices = np.random.choice(len(pc_filtered), NUM_TARGET_POINTS, replace=False)
    else:
        indices = np.random.choice(len(pc_filtered), NUM_TARGET_POINTS, replace=True)
        
    pc_filtered = pc_filtered[indices]
    if object_colors is not None:
        object_colors = object_colors[indices]

    inference_start = time.time()
    grasp_sampler = GraspGenSampler(grasp_cfg)
    
    pc_mean = pc_filtered.mean(axis=0)
    T_center_to_origin = tra.translation_matrix(-pc_mean)
    pc_centered_input = tra.transform_points(pc_filtered, T_center_to_origin)

    # 【关键修改】：把 natural_text 和 strict_text 传给模型
    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc_centered_input,  
        grasp_sampler,
        natural_text=natural_text,  
        strict_text=strict_text,          
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
    )
    inference_time = time.time() - inference_start

    if len(grasps_inferred) == 0:
        raise ValueError("无有效抓取姿态生成！请降低grasp_threshold或检查输入数据")

    grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
    grasps_inferred = grasps_inferred.cpu().numpy()
    grasps_inferred[:, 3, 3] = 1

    R_90 = tra.rotation_matrix(np.pi / 2, [0, 0, 1])
    grasps_inferred = np.array([g @ R_90 for g in grasps_inferred])
    
    T_origin_to_camera = tra.inverse_matrix(T_center_to_origin)
    grasps_inferred = np.array([T_origin_to_camera @ g for g in grasps_inferred])
    
    pc_centered = pc_filtered
    scene_pc_centered = scene_pc
    grasps_centered = grasps_inferred
    object_colors_centered = object_colors
    scene_colors_centered = scene_colors
    T_center = np.eye(4) 

    if len(scene_pc_centered) > max_scene_points:
        indices = np.random.choice(len(scene_pc_centered), max_scene_points, replace=False)
        scene_pc_downsampled = scene_pc_centered[indices]
    else:
        scene_pc_downsampled = scene_pc_centered

    collision_start = time.time()
    collision_free_mask = filter_colliding_grasps(
        scene_pc=scene_pc_downsampled,
        grasp_poses=grasps_centered,
        gripper_collision_mesh=gripper_collision_mesh,
        collision_threshold=collision_threshold,
    )
    collision_time = time.time() - collision_start

    collision_free_grasps = grasps_centered[collision_free_mask]
    collision_free_scores = grasp_conf_inferred[collision_free_mask]

    if len(collision_free_grasps) == 0:
        raise ValueError("无无碰撞抓取姿态！请调整碰撞阈值或抓取参数")

    best_idx = np.argmax(collision_free_scores)
    best_grasp_centered = collision_free_grasps[best_idx]
    best_score = collision_free_scores[best_idx]

    T_restore = tra.inverse_matrix(T_center)
    best_grasp_original = T_restore @ best_grasp_centered  

    if save_results:
        results = {
            "all_grasps": grasps_centered,
            "all_scores": grasp_conf_inferred,
            "collision_free_mask": collision_free_mask,
            "collision_free_grasps": collision_free_grasps,
            "collision_free_scores": collision_free_scores,
            "scene_pc": scene_pc_centered,
            "object_pc": pc_centered,
            "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "best_grasp_original": best_grasp_original,  
            "best_score": best_score,
            "T_center": T_center,
        }
        np.savez(output_file, **results)

    if visualize and vis is not None:
        viz_start = time.time()
        if object_colors_centered is not None:
            visualize_pointcloud(vis, "object_pc", pc_centered, object_colors_centered, size=0.0025)
        else:
            visualize_pointcloud(vis, "object_pc", pc_centered, [0, 255, 0], size=0.0025)

        collision_free_sorted_idx = np.argsort(collision_free_scores)[::-1]
        sorted_collision_free_grasps = collision_free_grasps[collision_free_sorted_idx]
        sorted_collision_free_scores = collision_free_scores[collision_free_sorted_idx]
        
        top50_num = min(100, len(sorted_collision_free_grasps))
        for i in range(top50_num):
            grasp = sorted_collision_free_grasps[i]
            visualize_grasp(vis, f"collision_free_grasps/top50_{i:03d}", grasp, [0, 255, 0], gripper_name, linewidth=4)
        
        if len(sorted_collision_free_grasps) > 0:
            best_grasp_viz = sorted_collision_free_grasps[0]
            visualize_grasp(vis, "collision_free_grasps/best", best_grasp_viz, [0, 255, 0], gripper_name, linewidth=2)

        colliding_grasps = grasps_centered[~collision_free_mask]
        for i, grasp in enumerate(colliding_grasps[:5]):
            visualize_grasp(vis, f"colliding_grasps/{i:03d}", grasp, [255, 0, 0], gripper_name, linewidth=2)

    total_time = time.time() - start_time

    return Grasp(
        pose=best_grasp_original,  
        score=best_score,
        collision_free=True,
        gripper_name=gripper_name,
        all_collision_free_grasps=collision_free_grasps,
        all_collision_free_scores=collision_free_scores
    )