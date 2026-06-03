#!/usr/bin/env python3
"""Run one Baseline5/GRIM trial per task/camera for fast debugging."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from auto_baseline5_six_objects_refactored import BASELINE_OBJECT_CONFIGS

SCRIPT_DIR = Path(__file__).resolve().parent
FULL_EVAL_SCRIPT = SCRIPT_DIR / "auto_baseline5_six_objects_refactored.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast Baseline5/GRIM debug run: one trial per selected object/task/camera.")
    parser.add_argument("--objects", nargs="+", choices=tuple(BASELINE_OBJECT_CONFIGS.keys()), default=tuple(BASELINE_OBJECT_CONFIGS.keys()))
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--cam-ids", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--output-root", default="batch_test_results_refactored/baseline5_debug_one_trial_views")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--close-distance", type=float, default=0.075)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--debug-visualize", action="store_true")
    parser.add_argument("--verbose-subprocess", action="store_true")
    parser.add_argument("--save-step-images", action="store_true")
    parser.add_argument("--grim-python", default="/home/zyp/pan1/conda/envs/grim/bin/python")
    parser.add_argument("--grim-worker", default=str(SCRIPT_DIR / "grim_worker_baseline5.py"))
    parser.add_argument("--grim-root", default="/home/zyp/pan1/GRIM")
    parser.add_argument("--grim-timeout", type=int, default=1200)
    parser.add_argument("--grim-max-points", type=int, default=22000)
    parser.add_argument("--grim-dino-long-side", type=int, default=700)
    parser.add_argument("--grim-feature-mode", choices=("auto", "dinov2", "geometry"), default="auto")
    parser.add_argument("--grim-dinov2-repo", default="")
    parser.add_argument("--grim-dinov2-model", default="dinov2_vitl14")
    parser.add_argument("--grim-dinov2-allow-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(FULL_EVAL_SCRIPT),
        "--objects", *args.objects,
        "--output-root", args.output_root,
        "--log-dir", str(Path(args.output_root) / "logs"),
        "--trials-per-task", "1",
        "--trial-ids", str(args.trial_id),
        "--cam-ids", *(str(cam_id) for cam_id in args.cam_ids),
        "--chunk-level", "camera",
        "--fast-sim",
        "--cooldown-seconds", "0",
        "--camera-width", str(args.camera_width),
        "--camera-height", str(args.camera_height),
        "--close-distance", str(args.close_distance),
        "--grim-python", args.grim_python,
        "--grim-worker", args.grim_worker,
        "--grim-root", args.grim_root,
        "--grim-timeout", str(args.grim_timeout),
        "--grim-max-points", str(args.grim_max_points),
        "--grim-dino-long-side", str(args.grim_dino_long_side),
        "--grim-feature-mode", args.grim_feature_mode,
        "--grim-dinov2-model", args.grim_dinov2_model,
    ]
    if args.grim_dinov2_repo:
        cmd.extend(["--grim-dinov2-repo", args.grim_dinov2_repo])
    if args.grim_dinov2_allow_download:
        cmd.append("--grim-dinov2-allow-download")
    if args.task_names:
        cmd.extend(["--task-names", *args.task_names])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.debug_visualize:
        cmd.append("--debug-visualize")
    if args.verbose_subprocess:
        cmd.append("--verbose-subprocess")
    if args.save_step_images:
        cmd.append("--save-step-images")

    print("🚀 启动 Baseline5/GRIM 一轮视角调试，每个任务/视角只跑 1 次")
    print(f"📁 输出目录: {args.output_root}")
    print("🧾 命令:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(SCRIPT_DIR))


if __name__ == "__main__":
    raise SystemExit(main())
