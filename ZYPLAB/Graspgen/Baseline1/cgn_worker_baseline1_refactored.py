#!/usr/bin/env python3
"""Contact-GraspNet worker for Baseline1 batch evaluation.

The IsaacSim process writes RGB-D + object mask to an npz file. This worker runs
inside the separate Contact-GraspNet environment and returns one camera-frame
4x4 grasp pose plus a score. Keeping CGN in a subprocess prevents its dependency
stack from contaminating IsaacSim/SAM3/Qwen imports.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

CGN_REPO_ROOT = Path("/home/zyp/pan1/contact_graspnet_pytorch")
CGN_SRC_DIR = CGN_REPO_ROOT / "contact_graspnet_pytorch"
for path in (CGN_REPO_ROOT, CGN_SRC_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.append(path_text)

from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

# Older Contact-GraspNet checkpoints may contain numpy scalar pickles.
torch.serialization.add_safe_globals([np.core.multiarray.scalar])


def enforce_orthogonal_grasps(grasps: np.ndarray) -> np.ndarray:
    """Project network rotations back to valid orthonormal frames.

    Contact-GraspNet can emit slightly non-orthogonal rotation matrices. IsaacSim
    IK is much happier when the axes are exactly orthogonal, so we preserve the
    approach axis and rebuild the other two axes by cross products.
    """
    fixed = np.copy(grasps)
    for i in range(len(fixed)):
        rot = fixed[i, :3, :3]
        x_raw = rot[:, 0]
        z_raw = rot[:, 2]
        z_new = z_raw / max(np.linalg.norm(z_raw), 1e-8)
        y_new = np.cross(z_new, x_raw)
        y_norm = np.linalg.norm(y_new)
        if y_norm < 1e-6:
            fallback_x = np.array([1.0, 0.0, 0.0]) if abs(z_new[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            y_new = np.cross(z_new, fallback_x)
            y_norm = np.linalg.norm(y_new)
        y_new = y_new / max(y_norm, 1e-8)
        x_new = np.cross(y_new, z_new)
        fixed[i, :3, 0] = x_new
        fixed[i, :3, 1] = y_new
        fixed[i, :3, 2] = z_new
    return fixed


def save_failure(out_path: str, reason: str) -> None:
    print(f"❌ {reason}", flush=True)
    np.savez(out_path, success=False, reason=np.array(reason))


def run_inference(args: argparse.Namespace) -> None:
    print("[Baseline1-CGN] 读取 RGB-D/mask 输入...", flush=True)
    data = np.load(args.in_data)
    depth = data["depth"]
    cam_k = data["K"]
    segmap = data["segmap"]
    rgb = data["rgb"] if "rgb" in data.files else None

    ckpt_dir = CGN_REPO_ROOT / "checkpoints" / "contact_graspnet"
    global_config = config_utils.load_config(str(ckpt_dir), batch_size=1)
    estimator = GraspEstimator(global_config)
    checkpoint_io = CheckpointIO(checkpoint_dir=str(ckpt_dir / "checkpoints"), model=estimator.model)
    checkpoint_io.load("model.pt")

    print("[Baseline1-CGN] 提取点云...", flush=True)
    pc_full, pc_segments, pc_colors = estimator.extract_point_clouds(
        depth,
        cam_k,
        segmap=segmap,
        rgb=rgb,
        z_range=[0.01, 2.0],
    )

    print("[Baseline1-CGN] 运行 Contact-GraspNet...", flush=True)
    with torch.no_grad():
        pred_grasps_cam, scores, contact_pts, gripper_openings = estimator.predict_scene_grasps(
            pc_full,
            pc_segments=pc_segments,
            local_regions=True,
            filter_grasps=True,
            forward_passes=args.forward_passes,
        )

    # 与旧版 cgn_worker_baseline1.py 保持一致：SAM3 segmap 对应的目标 id 是 1。
    obj_id = 1
    if obj_id not in pred_grasps_cam or len(pred_grasps_cam[obj_id]) == 0:
        save_failure(args.out_data, "未生成抓取")
        return

    raw_grasps = pred_grasps_cam[obj_id]
    raw_scores = scores[obj_id]
    raw_contact_pts = contact_pts[obj_id]
    obj_points = pc_segments[obj_id]

    valid_idxs = []
    for i, contact_point in enumerate(raw_contact_pts):
        dist = np.min(np.linalg.norm(obj_points - contact_point, axis=1))
        if dist < args.contact_threshold:
            valid_idxs.append(i)

    if not valid_idxs:
        save_failure(args.out_data, "过滤后无有效抓取点")
        return

    valid_idxs = np.asarray(valid_idxs, dtype=np.int64)
    final_grasps = enforce_orthogonal_grasps(raw_grasps[valid_idxs])
    final_scores = raw_scores[valid_idxs]
    final_openings = gripper_openings[obj_id][valid_idxs]
    best_idx = int(np.argmax(final_scores))

    if args.vis_data:
        np.savez_compressed(
            args.vis_data,
            pc_full=pc_full,
            pc_colors=pc_colors,
            grasps=final_grasps,
            scores=final_scores,
            openings=final_openings,
            best_idx=best_idx,
        )

    np.savez(
        args.out_data,
        success=True,
        best_grasp=final_grasps[best_idx],
        score=np.asarray(final_scores[best_idx], dtype=np.float32),
        candidate_count=np.asarray(len(final_grasps), dtype=np.int32),
    )
    print(
        f"✅ Baseline1-CGN 输出 best grasp: score={float(final_scores[best_idx]):.4f}, "
        f"candidates={len(final_grasps)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data", required=True)
    parser.add_argument("--out_data", required=True)
    parser.add_argument("--vis_data", default=None)
    parser.add_argument("--forward_passes", type=int, default=3)
    parser.add_argument("--contact_threshold", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
