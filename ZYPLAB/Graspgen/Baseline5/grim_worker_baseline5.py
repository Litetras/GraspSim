#!/usr/bin/env python3
"""GRIM worker for IsaacSim Baseline5.

The worker runs outside IsaacSim, in the `grim` conda environment.  It consumes
one RGB-D observation plus a target mask, builds a lightweight GRIM scene from
that observation, aligns a mapped GRIM memory object to the target point cloud,
and returns a transferred task grasp pose in OpenCV camera coordinates.

This intentionally does not run GRIM's RTA/predefined-grasp precision stage.
Baseline5 evaluates the transferred GRIM grasp directly in IsaacSim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

GRIM_ROOT_DEFAULT = "/home/zyp/pan1/GRIM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GRIM transfer inference for IsaacSim.")
    parser.add_argument("--in_data", required=True)
    parser.add_argument("--out_data", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--memory_obj", required=True)
    parser.add_argument("--memory_task", required=True)
    parser.add_argument("--grim_root", default=GRIM_ROOT_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_points", type=int, default=22000)
    parser.add_argument("--dino_long_side", type=int, default=700)
    parser.add_argument("--feature_mode", choices=("auto", "dinov2", "geometry"), default="auto")
    parser.add_argument("--dinov2_repo", default="")
    parser.add_argument("--dinov2_model", default="dinov2_vitl14")
    parser.add_argument("--dinov2_allow_download", action="store_true")
    parser.add_argument("--num_euler_steps", type=int, default=8)
    parser.add_argument("--top_k_orientations", type=int, default=10)
    parser.add_argument("--icp_iterations", type=int, default=50)
    parser.add_argument("--grasp_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def import_grim_utilities(grim_root: str):
    if grim_root not in sys.path:
        sys.path.insert(0, grim_root)
    from utilities import downsample_with_features, calculate_combined_score
    return downsample_with_features, calculate_combined_score


def clean_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0)
    return np.clip(depth, 0.0, 5.0)


def masked_point_cloud(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, max_points: int, seed: int):
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0.01) & (depth < 5.0)
    ys, xs = np.nonzero(valid)
    if len(xs) < 30:
        raise RuntimeError(f"目标 mask 有效深度点太少: {len(xs)}")

    if len(xs) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(xs), size=max_points, replace=False)
        xs = xs[keep]
        ys = ys[keep]

    z = depth[ys, xs].astype(np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts, xs.astype(np.int64), ys.astype(np.int64)


def load_dinov2(device: str, repo: str, model_name: str, allow_download: bool):
    print(f"[GRIM-Worker] 加载 DINOv2 特征模型: {model_name}", flush=True)
    if device == "cuda":
        # DINOv2 的位置编码插值可能触发 cuDNN 初始化；IsaacSim/SAM3 已占用 GPU 时
        # 这里容易报 CUDNN_STATUS_NOT_INITIALIZED。禁用 cuDNN 不影响 xFormers 注意力，
        # 但能让 DINOv2 继续走 CUDA kernel，避免退回 CPU。
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    attempts = []
    repo = repo.strip()
    if repo:
        attempts.append((repo, {"source": "local"}))

    hub_dir = Path(torch.hub.get_dir())
    cached_repos = sorted(hub_dir.glob("facebookresearch_dinov2*"))
    attempts.extend((str(path), {"source": "local"}) for path in cached_repos if path.is_dir())

    if allow_download:
        attempts.append(("facebookresearch/dinov2", {"trust_repo": True}))

    if not attempts:
        raise RuntimeError(
            "没有可用的本地 DINOv2 repo/cache。请传 --dinov2_repo 指向本地 dinov2 repo，"
            "或在联网环境下使用 --dinov2_allow_download 预先下载。"
        )

    errors = []
    for repo_or_dir, kwargs in attempts:
        try:
            print(f"[GRIM-Worker] 尝试 DINOv2 来源: {repo_or_dir}", flush=True)
            model = torch.hub.load(repo_or_dir, model_name, **kwargs)
            model = model.to(device).eval()
            if device == "cuda":
                model = model.half()
            return model
        except Exception as exc:
            errors.append(f"{repo_or_dir}: {repr(exc)}")

    raise RuntimeError("DINOv2 加载失败:\n  " + "\n  ".join(errors))
    model = model.to(device).eval()
    return model


def resize_to_patch_multiple(rgb: np.ndarray, long_side: int, patch: int = 14):
    h, w = rgb.shape[:2]
    scale = min(1.0, float(long_side) / float(max(h, w))) if long_side > 0 else 1.0
    new_h = max(patch, int(round(h * scale / patch)) * patch)
    new_w = max(patch, int(round(w * scale / patch)) * patch)
    image = Image.fromarray(rgb.astype(np.uint8)).resize((new_w, new_h), Image.BICUBIC)
    return np.asarray(image), scale


def dino_features_for_pixels(
    rgb: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    device: str,
    long_side: int,
    repo: str,
    model_name: str,
    allow_download: bool,
) -> np.ndarray:
    model = load_dinov2(device, repo=repo, model_name=model_name, allow_download=allow_download)
    resized, _ = resize_to_patch_multiple(rgb, long_side=long_side)
    h0, w0 = rgb.shape[:2]
    h, w = resized.shape[:2]

    image_dtype = torch.float16 if device == "cuda" else torch.float32
    img = torch.from_numpy(resized).to(device=device, dtype=image_dtype) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=image_dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=image_dtype).view(1, 3, 1, 1)
    img = (img - mean) / std

    with torch.inference_mode():
        out = model.forward_features(img)
        tokens = out["x_norm_patchtokens"].detach().float()[0]
    patch = 14
    ph, pw = h // patch, w // patch
    feat_map = tokens.reshape(ph, pw, -1)

    px = np.floor(xs.astype(np.float32) / max(w0 - 1, 1) * (pw - 1)).astype(np.int64)
    py = np.floor(ys.astype(np.float32) / max(h0 - 1, 1) * (ph - 1)).astype(np.int64)
    px = np.clip(px, 0, pw - 1)
    py = np.clip(py, 0, ph - 1)
    feats = feat_map[torch.from_numpy(py).to(device), torch.from_numpy(px).to(device)]
    feats = feats.detach().cpu().numpy().astype(np.float32)
    del model, img, out, tokens, feat_map
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats


def normalize_features_np(feats: np.ndarray) -> np.ndarray:
    feats_t = torch.tensor(feats, dtype=torch.float32)
    feats_t = F.normalize(feats_t, p=2, dim=1)
    return feats_t.cpu().numpy().astype(np.float32)


def load_memory(grim_root: str, memory_obj: str, memory_task: str, device: str):
    memory_json_path = Path(grim_root) / "memory" / "memory.json"
    with memory_json_path.open("r", encoding="utf-8") as f:
        memory_json = json.load(f)
    if memory_obj not in memory_json:
        raise RuntimeError(f"GRIM memory 中没有物体 '{memory_obj}'")
    if memory_task not in memory_json[memory_obj]:
        raise RuntimeError(
            f"GRIM memory 物体 '{memory_obj}' 中没有任务 '{memory_task}', "
            f"可用任务={sorted(memory_json[memory_obj].keys())}"
        )
    grasps_for_task = memory_json[memory_obj][memory_task]
    if not grasps_for_task:
        raise RuntimeError(f"GRIM memory '{memory_obj}/{memory_task}' 没有 grasp")

    feature_mesh_path = Path(grim_root) / "memory" / "fm_output" / memory_obj / "feature_mesh.pt"
    if not feature_mesh_path.exists():
        raise RuntimeError(f"缺少 GRIM feature mesh: {feature_mesh_path}")
    # memory 只用于取 vertices / DINO feature，保持在 CPU 可以避免和 IsaacSim/SAM3 抢 GPU 显存。
    memory_data = torch.load(str(feature_mesh_path), map_location="cpu", weights_only=False)
    return memory_data, grasps_for_task


def parse_memory_grasps(grasps_for_task: Dict) -> Tuple[List[np.ndarray], List[float], List[float], List[float]]:
    poses, widths, lengths, scales = [], [], [], []
    required = ("gripper_pose_4x4", "scale_factor", "gripper_width", "gripper_length")
    for grasp_id, info in grasps_for_task.items():
        if not all(key in info for key in required):
            print(f"[GRIM-Worker] 跳过 grasp {grasp_id}: 字段不完整", flush=True)
            continue
        poses.append(np.asarray(info["gripper_pose_4x4"], dtype=np.float64))
        scales.append(float(info["scale_factor"]))
        widths.append(float(info["gripper_width"]))
        lengths.append(float(info["gripper_length"]))
    if not poses:
        raise RuntimeError("没有可解析的 memory grasp pose")
    return poses, widths, lengths, scales


def align_and_transfer(
    target_points: np.ndarray,
    target_feats_norm: np.ndarray,
    memory_data: Dict,
    grasps_for_task: Dict,
    args: argparse.Namespace,
):
    downsample_with_features, calculate_combined_score = import_grim_utilities(args.grim_root)

    W_GEOM = 10
    W_FEAT = float(getattr(args, "feature_weight", 100.0))
    EVAL_K_NEIGHBORS = 3
    PCA_FEATURE_DIM = 4
    SCORING_VOXEL_SIZE_DIVISOR = 25.0
    INITIAL_CORR_DIST_FACTOR_PCA = 0.50
    INNER_ICP_MAX_CORR_DIST_FACTOR = 0.30
    FINAL_EVAL_DISTANCE_FACTOR = 1 / 3.0
    NB_NEIGHBORS_OUTLIER = 15
    STD_RATIO_OUTLIER = 0.5

    obj_vertices = memory_data["vertices"]
    if isinstance(obj_vertices, torch.Tensor):
        obj_vertices = obj_vertices.detach().cpu().numpy()
    obj_vertices = obj_vertices.astype(np.float64)
    raw_feats = memory_data["out"]["dino_feats"]
    if isinstance(raw_feats, torch.Tensor):
        raw_feats_t = raw_feats.detach().float().to(args.device)
    else:
        raw_feats_t = torch.tensor(raw_feats, dtype=torch.float32, device=args.device)
    source_feats_norm = F.normalize(raw_feats_t, p=2, dim=1).detach().cpu().numpy().astype(np.float32)
    del raw_feats_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if source_feats_norm.shape[0] != obj_vertices.shape[0]:
        raise RuntimeError(
            f"memory vertex/feature 数量不一致: {obj_vertices.shape[0]} vs {source_feats_norm.shape[0]}"
        )

    target_pcd_full = o3d.geometry.PointCloud()
    target_pcd_full.points = o3d.utility.Vector3dVector(target_points.astype(np.float64))
    print(f"[GRIM-Worker] Target cloud before outlier removal: {len(target_pcd_full.points)}", flush=True)
    target_pcd, ind = target_pcd_full.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS_OUTLIER, std_ratio=STD_RATIO_OUTLIER
    )
    target_feats_filtered = target_feats_norm[ind]
    print(f"[GRIM-Worker] Target cloud after outlier removal: {len(target_pcd.points)}", flush=True)
    if len(target_pcd.points) < 20:
        raise RuntimeError(f"目标点云过滤后太少: {len(target_pcd.points)}")
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(obj_vertices)
    if len(source_pcd.points) < 20:
        raise RuntimeError(f"memory source 点太少: {len(source_pcd.points)}")
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    memory_grasp_poses, widths, lengths, scales = parse_memory_grasps(grasps_for_task)

    pca_model = PCA(n_components=PCA_FEATURE_DIM)
    pca_model.fit(source_feats_norm)
    source_feats_pca = pca_model.transform(source_feats_norm).astype(np.float32)

    max_dim_target = (target_pcd.get_max_bound() - target_pcd.get_min_bound()).max()
    if max_dim_target <= 1e-6:
        max_dim_target = 1.0
    initial_eval_distance = max(0.01, max_dim_target * INITIAL_CORR_DIST_FACTOR_PCA)
    inner_icp_distance = max(0.005, max_dim_target * INNER_ICP_MAX_CORR_DIST_FACTOR)
    final_eval_distance = max(0.005, initial_eval_distance * FINAL_EVAL_DISTANCE_FACTOR)

    centroid_target = target_pcd.get_center()
    centroid_source = source_pcd.get_center()
    target_centered = np.asarray(target_pcd.points) - centroid_target
    source_centered = np.asarray(source_pcd.points) - centroid_source
    alignment_scale = 1.0
    if len(target_centered) >= 2 and len(source_centered) >= 2:
        pca_target = PCA(n_components=1).fit(target_centered)
        pca_source = PCA(n_components=1).fit(source_centered)
        var_target = pca_target.explained_variance_[0]
        var_source = pca_source.explained_variance_[0]
        if var_target > 1e-12 and var_source > 1e-12:
            alignment_scale = float(np.sqrt(var_target / var_source))
    print(f"[GRIM-Worker] Initial scale: {alignment_scale:.4f}", flush=True)

    scoring_voxel_target = max_dim_target / SCORING_VOXEL_SIZE_DIVISOR
    scoring_voxel_source = scoring_voxel_target / alignment_scale if alignment_scale > 1e-6 else scoring_voxel_target
    scoring_voxel_source = float(np.clip(scoring_voxel_source, 0.0005, 0.05))
    source_pcd_scoring, source_feats_scoring = downsample_with_features(source_pcd, source_feats_norm, scoring_voxel_source)
    target_pcd_scoring, target_feats_scoring = downsample_with_features(target_pcd, target_feats_filtered, scoring_voxel_target)
    if len(source_pcd_scoring.points) < max(2, PCA_FEATURE_DIM) or len(target_pcd_scoring.points) < max(2, PCA_FEATURE_DIM):
        raise RuntimeError(
            f"downsample 后点太少: src={len(source_pcd_scoring.points)}, tgt={len(target_pcd_scoring.points)}"
        )
    print(
        f"[GRIM-Worker] Downsampled sizes: src={len(source_pcd_scoring.points)}, tgt={len(target_pcd_scoring.points)}",
        flush=True,
    )

    source_feats_pca_scoring = pca_model.transform(source_feats_scoring).astype(np.float32)
    target_feats_pca_scoring = pca_model.transform(target_feats_scoring).astype(np.float32)
    target_kdtree_scoring = o3d.geometry.KDTreeFlann(target_pcd_scoring)

    candidates = []
    angles = np.linspace(0, 360, args.num_euler_steps, endpoint=False)
    cand_idx = 0
    for ax in angles:
        for ay in angles:
            for az in angles:
                R_sampled = R.from_euler("zyx", [az, ay, ax], degrees=True).as_matrix()
                T_init = np.eye(4)
                T_init[:3, :3] = alignment_scale * R_sampled
                T_init[:3, 3] = centroid_target - (T_init[:3, :3] @ centroid_source)
                if np.isfinite(T_init).all():
                    score = calculate_combined_score(
                        source_pcd_scoring,
                        target_pcd_scoring,
                        T_init,
                        source_feats_pca_scoring,
                        target_feats_pca_scoring,
                        target_kdtree_scoring,
                        W_GEOM,
                        W_FEAT,
                        initial_eval_distance,
                        EVAL_K_NEIGHBORS,
                    )
                    if np.isfinite(score):
                        candidates.append({"index": cand_idx, "T_init": T_init, "score": score})
                cand_idx += 1
    if not candidates:
        raise RuntimeError("没有有效初始姿态候选")
    candidates.sort(key=lambda x: x["score"])
    top_candidates = candidates[: args.top_k_orientations]
    print(f"[GRIM-Worker] Top orientations: {len(top_candidates)}, best score={top_candidates[0]['score']:.4f}", flush=True)

    target_for_icp = target_pcd
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=args.icp_iterations
    )
    refined = []
    for cand in top_candidates:
        reg = o3d.pipelines.registration.registration_icp(
            source=source_pcd,
            target=target_for_icp,
            max_correspondence_distance=inner_icp_distance,
            init=cand["T_init"],
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            criteria=icp_criteria,
        )
        if np.isfinite(reg.transformation).all() and reg.fitness > 1e-4:
            refined.append(
                {
                    "index": cand["index"],
                    "T_refined": reg.transformation.copy(),
                    "fitness": float(reg.fitness),
                    "rmse": float(reg.inlier_rmse),
                    "initial_score": float(cand["score"]),
                }
            )
    print(f"[GRIM-Worker] ICP refined: {len(refined)}", flush=True)
    if not refined:
        raise RuntimeError("ICP 没有成功 refined candidate")

    target_feats_pca_final = pca_model.transform(target_feats_filtered).astype(np.float32)
    target_kdtree_final = o3d.geometry.KDTreeFlann(target_pcd)
    for item in refined:
        score = calculate_combined_score(
            source_pcd,
            target_pcd,
            item["T_refined"],
            source_feats_pca,
            target_feats_pca_final,
            target_kdtree_final,
            W_GEOM,
            W_FEAT,
            final_eval_distance,
            EVAL_K_NEIGHBORS,
        )
        item["final_score"] = float(score) if np.isfinite(score) else float("inf")
    refined.sort(key=lambda x: x["final_score"])
    best = refined[0]
    T_final = best["T_refined"]
    print(
        f"[GRIM-Worker] Best alignment: cand={best['index']} score={best['final_score']:.4f} "
        f"fit={best['fitness']:.4f} rmse={best['rmse']:.4f}",
        flush=True,
    )

    correction = np.eye(4)
    correction[:3, :3] = R.from_euler("xyz", [-90, 0, 0], degrees=True).as_matrix()

    final_grasps = []
    for pose, scale in zip(memory_grasp_poses, scales):
        if abs(scale) < 1e-6:
            scale = 1.0
        try:
            grasp_pose_corrected = pose @ correction
            scale_inv = np.diag([1.0 / scale] * 3 + [1.0])
            grasp_unscaled = scale_inv @ grasp_pose_corrected
            grasp_scene = T_final @ grasp_unscaled
            if np.isfinite(grasp_scene).all():
                final_grasps.append(grasp_scene.astype(np.float64))
        except Exception as exc:
            print(f"[GRIM-Worker] grasp transfer 失败: {exc}", flush=True)
    if not final_grasps:
        raise RuntimeError("GRIM 对齐后没有可用 transferred grasp")

    grasp_index = int(np.clip(args.grasp_index, 0, len(final_grasps) - 1))
    best_grasp = final_grasps[grasp_index]
    return best_grasp, final_grasps, best


def save_failure(out_path: str, reason: str) -> None:
    np.savez(out_path, success=False, reason=np.array(reason))


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    try:
        data = np.load(args.in_data, allow_pickle=False)
        depth = clean_depth(data["depth"])
        K = np.asarray(data["K"], dtype=np.float32)
        mask = np.asarray(data["segmap"], dtype=np.uint8)
        rgb = np.asarray(data["rgb"], dtype=np.uint8)[..., :3]
        data.close()

        points, xs, ys = masked_point_cloud(depth, mask, K, max_points=args.max_points, seed=args.seed)
        print(f"[GRIM-Worker] Isaac target points: {len(points)}", flush=True)
        memory_data, grasps_for_task = load_memory(args.grim_root, args.memory_obj, args.memory_task, args.device)
        source_feats = memory_data["out"]["dino_feats"]
        source_feat_dim = int(source_feats.shape[1])

        if args.feature_mode == "geometry":
            print("[GRIM-Worker] 使用 geometry-only 对齐调试模式: 不计算 Isaac 目标 DINO 特征。", flush=True)
            dino_feats_norm = np.zeros((len(points), source_feat_dim), dtype=np.float32)
            args.feature_weight = 0.0
        else:
            try:
                dino_feats = dino_features_for_pixels(
                    rgb,
                    xs,
                    ys,
                    args.device,
                    args.dino_long_side,
                    repo=args.dinov2_repo,
                    model_name=args.dinov2_model,
                    allow_download=args.dinov2_allow_download,
                )
                if dino_feats.shape[1] != source_feat_dim:
                    raise RuntimeError(f"DINO feature dim 不匹配: target={dino_feats.shape[1]}, memory={source_feat_dim}")
                dino_feats_norm = normalize_features_np(dino_feats)
                args.feature_weight = 100.0
            except Exception as exc:
                if args.feature_mode == "dinov2":
                    raise
                print(
                    "[GRIM-Worker] ⚠️ DINOv2 目标特征不可用，自动退回 geometry-only 对齐。"
                    f"原因: {exc}",
                    flush=True,
                )
                dino_feats_norm = np.zeros((len(points), source_feat_dim), dtype=np.float32)
                args.feature_weight = 0.0

        best_grasp, all_grasps, alignment = align_and_transfer(
            target_points=points,
            target_feats_norm=dino_feats_norm,
            memory_data=memory_data,
            grasps_for_task=grasps_for_task,
            args=args,
        )

        np.savez(
            args.out_data,
            success=True,
            best_grasp=best_grasp.astype(np.float64),
            score=np.array(float(-alignment.get("final_score", 0.0))),
            candidate_count=np.array(len(all_grasps), dtype=np.int32),
            all_grasps=np.stack(all_grasps).astype(np.float64),
            memory_obj=np.array(args.memory_obj),
            memory_task=np.array(args.memory_task),
            alignment_score=np.array(float(alignment.get("final_score", float("inf")))),
            alignment_fitness=np.array(float(alignment.get("fitness", 0.0))),
            alignment_rmse=np.array(float(alignment.get("rmse", 0.0))),
        )
        print(
            f"✅ GRIM 输出 best grasp: obj={args.memory_obj}, task={args.memory_task}, candidates={len(all_grasps)}",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"❌ GRIM worker failed: {exc}", flush=True)
        save_failure(args.out_data, str(exc))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
