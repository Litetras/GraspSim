# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
封装后接口说明：
核心函数 demo_variable(rgb_data, depth_data, mask, intrinsic, **kwargs)
- 输入：内存中的图像数组+相机内参
- 输出：最优无碰撞抓取结果（Grasp类实例，其pose属性为4x4齐次变换矩阵）
- 保留所有原功能：点云下采样、碰撞检测、可视化、结果保存等
"""

# 基础库导入
import argparse
import os
import time
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm

# GraspGen项目内部依赖
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import (
    point_cloud_outlier_removal,
    knn_points,
    depth_and_segmentation_to_point_clouds,
    filter_colliding_grasps,
)
from grasp_gen.robot import get_gripper_info


# ===================== 1. 定义抓取结果封装类 =====================
class Grasp:
    """
    抓取结果结构化封装类
    属性说明：
    - pose: 4x4 齐次变换矩阵（抓取位姿，世界坐标系）【重点：已默认返回4x4矩阵】
    - score: 抓取质量分数（0-1）
    - collision_free: 是否无碰撞（bool）
    - gripper_name: 夹爪名称
    - all_collision_free_grasps: 所有无碰撞抓取姿态列表（可选，每个元素都是4x4矩阵）
    - all_collision_free_scores: 所有无碰撞抓取分数列表（可选）
    """
    def __init__(
        self,
        pose: np.ndarray,
        score: float,
        collision_free: bool,
        gripper_name: str,
        all_collision_free_grasps: Optional[np.ndarray] = None,
        all_collision_free_scores: Optional[np.ndarray] = None
    ):
        self.pose = pose  # 【改动1：注释强化】确保pose是4x4齐次变换矩阵
        self.score = score
        self.collision_free = collision_free
        self.gripper_name = gripper_name
        self.all_collision_free_grasps = all_collision_free_grasps  # 所有无碰撞抓取（每个元素4x4矩阵）
        self.all_collision_free_scores = all_collision_free_scores  # 所有无碰撞分数

    def __repr__(self):
        return f"Grasp(score={self.score:.3f}, collision_free={self.collision_free}, gripper={self.gripper_name})"


# ===================== 2. 核心封装函数（对外接口） =====================
def demo_variable(
    rgb_data: Optional[np.ndarray],    # RGB图像 (H,W,3) uint8，None则无彩色可视化
    depth_data: np.ndarray,           # 深度图像 (H,W) float32，单位：米（关键！）
    mask: np.ndarray,                 # 分割掩码 (H,W) uint8，目标物体ID=1
    intrinsic: List[float],           # 相机内参 [fx, fy, cx, cy]
    text: str = "down",                 # <====== 新增这一行：接收语言指令##############
    # 以下为可选参数（保持原代码默认值，可按需覆盖）
    gripper_config: str = "/home/zyp/Desktop/zyp_dataset2/tutorial/models/tutorial_model_config.yaml",
    grasp_threshold: float = 0.8,
    num_grasps: int = 200,
    return_topk: bool = False,
    topk_num_grasps: int = -1,
    collision_threshold: float = 0,#0.02
    max_scene_points: int = 8192,
    #max_object_points: int = 60000,
    visualize: bool = True,
    save_results: bool = True,
    output_file: str = "collision_free_grasps_results.npz"
) -> Grasp:
    """
    对外核心接口：输入图像数据+相机内参，输出最优无碰撞抓取结果
    
    Args:
        rgb_data: RGB图像数组，shape=(H,W,3)，dtype=uint8，None则不使用彩色可视化
        depth_data: 深度图像数组，shape=(H,W)，dtype=float32，**必须是米单位**
        mask: 实例分割掩码，shape=(H,W)，dtype=uint8，目标物体的掩码值为1
        intrinsic: 相机内参列表，[fx, fy, cx, cy]
        gripper_config: 夹爪配置文件路径
        grasp_threshold: 抓取分数阈值（-1返回Top100）
        num_grasps: 生成的初始抓取数量
        return_topk: 是否仅返回TopK高分抓取
        topk_num_grasps: TopK数量（return_topk=True时生效）
        collision_threshold: 碰撞检测距离阈值（米）
        max_scene_points: 场景点云最大数量（碰撞检测优化）
        max_object_points: 物体点云最大数量（避免CUDA OOM）
        visualize: 是否开启MeshCat可视化
        save_results: 是否保存结果到npz文件
        output_file: 结果保存路径
    
    Returns:
        Grasp: 最优无碰撞抓取结果（pose属性为4x4齐次变换矩阵，包含平移+旋转）【改动2：返回值注释强化】
    
    Raises:
        ValueError: 关键步骤失败时抛出（如点云生成失败、无有效抓取）
    """
    # 初始化计时
    start_time = time.time()
    print(f"Starting collision-free grasp detection at {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    # 1. 参数预处理
    if return_topk and topk_num_grasps == -1:
        topk_num_grasps = 100

    # 校验输入数据合法性
    if depth_data.ndim != 2:
        raise ValueError(f"depth_data必须是2维数组，当前shape={depth_data.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask必须是2维数组，当前shape={mask.shape}")
    if len(intrinsic) != 4:
        raise ValueError(f"intrinsic必须是4个元素[fx, fy, cx, cy]，当前长度={len(intrinsic)}")
    if rgb_data is not None and rgb_data.shape[:2] != depth_data.shape[:2]:
        raise ValueError(f"RGB图像与深度图像尺寸不匹配！RGB={rgb_data.shape}, Depth={depth_data.shape}")

    # 2. 校验夹爪配置文件
    if not os.path.exists(gripper_config):
        raise ValueError(f"夹爪配置文件不存在: {gripper_config}")

    # 3. 生成场景/物体点云
    fx, fy, cx, cy = intrinsic
    pc_start = time.time()
    try:
        scene_pc, object_pc, scene_colors, object_colors = depth_and_segmentation_to_point_clouds(
            depth_image=depth_data,
            segmentation_mask=mask,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            rgb_image=rgb_data,
            target_object_id=1,  # 目标物体掩码ID固定为1（可按需改为参数）
            remove_object_from_scene=True,
        )
    except Exception as e:
        raise ValueError(f"点云生成失败: {str(e)}") from e
    pc_creation_time = time.time() - pc_start
    print(f"Point cloud creation took: {pc_creation_time:.2f} seconds")

    # 4. 加载夹爪配置
    grasp_cfg = load_grasp_cfg(gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    gripper_info = get_gripper_info(gripper_name)
    gripper_collision_mesh = gripper_info.collision_mesh
    print(f"Using gripper: {gripper_name}")
    print(f"Gripper collision mesh vertices: {len(gripper_collision_mesh.vertices)}")

    # 5. 初始化可视化（如果开启）
    vis = None
    if visualize:
        vis = create_visualizer()

    # 7. 物体点云去离群点
    filter_start = time.time()
    object_pc_torch = torch.from_numpy(object_pc)
    pc_filtered, pc_removed = point_cloud_outlier_removal(object_pc_torch)
    pc_filtered = pc_filtered.numpy()
    pc_removed = pc_removed.numpy()
    filter_time = time.time() - filter_start
    print(f"Point cloud filtering took: {filter_time:.2f}s")
    print(f"Filtered PC: {len(pc_filtered)} points (removed {len(pc_removed)} outliers)")

    # 8. 抓取姿态推理
    inference_start = time.time()
    grasp_sampler = GraspGenSampler(grasp_cfg)
    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc_filtered,
        grasp_sampler,
        text=text,  # <====== 极其关键：新增这一行，把文字打包成 list 传给模型
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
    )
    inference_time = time.time() - inference_start
    print(f"Grasp inference took: {inference_time:.2f}s")

    # 校验推理结果
    if len(grasps_inferred) == 0:
        raise ValueError("无有效抓取姿态生成！请降低grasp_threshold或检查输入数据")

    # 转换为numpy（GPU→CPU）
    grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
    grasps_inferred = grasps_inferred.cpu().numpy()
    grasps_inferred[:, 3, 3] = 1  # 确保齐次矩阵合法【改动3：注释强化】
    # ================= 就在这里新增旋转逻辑 ===################为了graspknife-2026.3.18版本适配，强制将所有生成的抓取姿态绕局部Z轴旋转90度（因为模型输出的姿态与实际夹爪方向不一致）
    print("\n[注意]：生成的点云抓取姿态已绕局部 Z 轴旋转了 90 度！\n")
    import trimesh.transformations as tra
    R_90 = tra.rotation_matrix(np.pi / 2, [0, 0, 1])
    grasps_inferred = np.array([g @ R_90 for g in grasps_inferred])
    # ========================================================
    print(f"Inferred {len(grasps_inferred)} grasps (score range: {grasp_conf_inferred.min():.3f}~{grasp_conf_inferred.max():.3f})")

    # 9. 点云/抓取姿态中心化（统一坐标系）
    def process_point_cloud(pc, grasps, grasp_conf, pc_colors=None):
        """内部辅助函数：点云中心化"""
        scores = get_color_from_score(grasp_conf, use_255_scale=True)
        grasps[:, 3, 3] = 1  # 确保齐次矩阵合法
        T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
        pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
        grasps_centered = np.array([T_subtract_pc_mean @ g for g in grasps.tolist()])
        
        pc_colors_centered = pc_colors
        if pc_colors is not None:
            pc_colors_centered = pc_colors.copy().astype(np.float32)
            pc_colors_centered[:, 0] = np.clip(pc_colors_centered[:, 0] * 1.4, 0, 255)
            pc_colors_centered = pc_colors_centered.astype(np.uint8)
        return pc_centered, grasps_centered, scores, T_subtract_pc_mean, pc_colors_centered

    pc_centered, grasps_centered, scores, T_center, object_colors_centered = process_point_cloud(
        pc_filtered, grasps_inferred, grasp_conf_inferred, object_colors
    )
    scene_pc_centered = tra.transform_points(scene_pc, T_center)

    # 场景点云颜色增强（可选）
    scene_colors_centered = scene_colors
    if scene_colors is not None:
        scene_colors_centered = scene_colors.copy().astype(np.float32)
        scene_colors_centered[:, 0] = np.clip(scene_colors_centered[:, 0] * 1.4, 0, 255)
        scene_colors_centered = scene_colors_centered.astype(np.uint8)

    # 10. 场景点云下采样（加速碰撞检测）
    if len(scene_pc_centered) > max_scene_points:
        indices = np.random.choice(len(scene_pc_centered), max_scene_points, replace=False)
        scene_pc_downsampled = scene_pc_centered[indices]
        print(f"Downsampled scene PC: {len(scene_pc_centered)} → {len(scene_pc_downsampled)} points")
    else:
        scene_pc_downsampled = scene_pc_centered

    # 11. 碰撞检测
    collision_start = time.time()
    collision_free_mask = filter_colliding_grasps(
        scene_pc=scene_pc_downsampled,
        grasp_poses=grasps_centered,
        gripper_collision_mesh=gripper_collision_mesh,
        collision_threshold=collision_threshold,
    )
    collision_time = time.time() - collision_start
    print(f"Collision detection took: {collision_time:.2f}s")

    # 筛选无碰撞抓取
    collision_free_grasps = grasps_centered[collision_free_mask]
    collision_free_scores = grasp_conf_inferred[collision_free_mask]
    print(f"Final: {len(collision_free_grasps)} collision-free grasps / {len(grasps_inferred)} total")

    # 校验无碰撞结果
    if len(collision_free_grasps) == 0:
        raise ValueError("无无碰撞抓取姿态！请调整碰撞阈值或抓取参数")

    # 12. 选择最优无碰撞抓取（分数最高）
    best_idx = np.argmax(collision_free_scores)
    best_grasp_centered = collision_free_grasps[best_idx]
    best_score = collision_free_scores[best_idx]

    # 还原到原始坐标系（去中心化）
    T_restore = tra.inverse_matrix(T_center)
    best_grasp_original = T_restore @ best_grasp_centered  # 【改动4：注释强化】结果为4x4齐次变换矩阵

    # 13. 保存结果（可选）
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
            "best_grasp_original": best_grasp_original,  # 保存完整的4x4矩阵
            "best_score": best_score,
            "T_center": T_center,
        }
        np.savez(output_file, **results)
        print(f"Results saved to {output_file}")

    # 14. 可视化（可选）
    if visualize and vis is not None:
        viz_start = time.time()
        # 可视化场景点云
        if scene_colors_centered is not None:
            visualize_pointcloud(vis, "scene_pc", scene_pc_centered, scene_colors_centered, size=0.002)
        else:
            visualize_pointcloud(vis, "scene_pc", scene_pc_centered, [128, 128, 128], size=0.002)
        # 可视化物体点云
        if object_colors_centered is not None:
            visualize_pointcloud(vis, "object_pc", pc_centered, object_colors_centered, size=0.0025)
        else:
            visualize_pointcloud(vis, "object_pc", pc_centered, [0, 255, 0], size=0.0025)



    # ===================== 关键修改：可视化优化 =====================
    # 对无碰撞抓取按分数排序（降序）
    collision_free_sorted_idx = np.argsort(collision_free_scores)[::-1]
    sorted_collision_free_grasps = collision_free_grasps[collision_free_sorted_idx]
    sorted_collision_free_scores = collision_free_scores[collision_free_sorted_idx]
    
    # 可视化前100个无碰撞抓取（绿色）
    top50_num = min(100, len(sorted_collision_free_grasps))
    for i in range(top50_num):
        grasp = sorted_collision_free_grasps[i]
        # 绿色：[0, 255, 0]
        visualize_grasp(vis, f"collision_free_grasps/top50_{i:03d}", grasp, [0, 255, 0], gripper_name, linewidth=4)
    
    # 可视化最优无碰撞抓取（黄色，覆盖top50中的第一个）
    if len(sorted_collision_free_grasps) > 0:
        best_grasp_viz = sorted_collision_free_grasps[0]
        # 黄色：[255, 255, 0]
        visualize_grasp(vis, "collision_free_grasps/best", best_grasp_viz, [0, 255, 0], gripper_name, linewidth=2)

    # 可视化碰撞抓取（前10个，红色不变）
    colliding_grasps = grasps_centered[~collision_free_mask]
    for i, grasp in enumerate(colliding_grasps[:5]):
        visualize_grasp(vis, f"colliding_grasps/{i:03d}", grasp, [255, 0, 0], gripper_name, linewidth=2)
    viz_time = time.time() - viz_start



        
        # # ===================== 关键修改：可视化最优无碰撞抓取（替换原"第一个"逻辑） =====================
        # # 可视化最优无碰撞抓取（分数最高的那个）
        # collision_free_colors = get_color_from_score(collision_free_scores, use_255_scale=True)
        # # 先判断是否存在无碰撞抓取，避免索引越界
        # if len(collision_free_grasps) > 0:
        #     # 取分数最高的最优无碰撞抓取和对应颜色（使用之前计算的best_idx）
        #     best_grasp_viz = collision_free_grasps[best_idx]
        #     best_color_viz = collision_free_colors[best_idx]
        #     visualize_grasp(vis, "collision_free_grasps/best", best_grasp_viz, best_color_viz, gripper_name, linewidth=5)

        # # 可视化碰撞抓取（前10个）
        # colliding_grasps = grasps_centered[~collision_free_mask]
        # for i, grasp in enumerate(colliding_grasps[:10]):
        #     visualize_grasp(vis, f"colliding_grasps/{i:03d}", grasp, [255, 0, 0], gripper_name, linewidth=0.9)
        # viz_time = time.time() - viz_start

        # quxiao ke shi hua de enter
        # print(f"Visualization setup took: {viz_time:.2f}s")
        # print("Visualization tips:")
        # print("- Green: target object | Gray: scene")
        # print("- Colored grasps: collision-free (score-based color) | Red grasps: colliding")
        # input("Press Enter to exit visualization...")

    # 15. 打印耗时汇总
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TIMING SUMMARY:")
    print(f"Point cloud creation: {pc_creation_time:.2f}s")
    print(f"Point cloud filtering: {filter_time:.2f}s")
    print(f"Grasp inference:     {inference_time:.2f}s")
    print(f"Collision detection: {collision_time:.2f}s")
    print(f"Total time:          {total_time:.2f}s")
    print("=" * 60)

    # 16. 返回结构化抓取结果
    return Grasp(
        pose=best_grasp_original,  # 【核心：返回的pose是4x4齐次变换矩阵】
        score=best_score,
        collision_free=True,
        gripper_name=gripper_name,
        all_collision_free_grasps=collision_free_grasps,
        all_collision_free_scores=collision_free_scores
    )


# ===================== 3. 测试调用示例（模拟外部传入数据） =====================
if __name__ == "__main__":
    """
    测试示例：模拟外部加载图像数据，调用demo_variable接口
    实际使用时，只需将rgb_data/depth_data/mask/intrinsic替换为你的数据即可
    """
    import cv2

    # ---------------------- 步骤1：加载测试数据（模拟外部传入） ----------------------
    # 1. 加载深度图（转换为米单位）
    depth_path = "/home/zyp/GraspGen/scripts/data1/depth.png"
    depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # mm → m

    # 2. 加载分割掩码（目标ID=1）
    mask_path = "/home/zyp/GraspGen/scripts/data1/color_mask.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 3. 加载RGB图（可选）
    rgb_path = "/home/zyp/GraspGen/scripts/data1/color.png"
    rgb_data = cv2.imread(rgb_path)
    rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)  # 转为RGB格式（与模型输入一致）

    # 4. 相机内参
    intrinsic = [605.26, 605.34, 318.31, 234.76]  # [fx, fy, cx, cy]

    # ---------------------- 步骤2：调用封装接口（核心） ----------------------
    try:
        # 目标调用形式：直接传入内存中的数组
        detected_grasp: Grasp = demo_variable(
            rgb_data=rgb_data,
            depth_data=depth_data,
            mask=mask,
            intrinsic=intrinsic,
            text="down",      # <====== 新增：告诉它你要抓上面还是下面
            visualize=True,  # 开启可视化
            save_results=True,  # 保存结果
            grasp_threshold=0.8,
            num_grasps=100
        )

        # 打印结果
        print("\n===== 最优抓取结果 =====")
        print(f"抓取分数: {detected_grasp.score:.3f}")
        print(f"抓取位姿（XYZ）: {detected_grasp.pose[:3, 3]}")
        # 【改动5：新增】打印完整的4x4齐次变换矩阵
        print(f"抓取位姿（完整4x4齐次变换矩阵）:\n {detected_grasp.pose}")
        print(f"夹爪类型: {detected_grasp.gripper_name}")
        print(f"无碰撞抓取总数: {len(detected_grasp.all_collision_free_grasps)}")

    except ValueError as e:
        print(f"Error: {e}")
