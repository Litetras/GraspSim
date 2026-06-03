#!/usr/bin/env python3
"""Run one Baseline2/GraspGPT trial per task/camera for fast view debugging.

This wrapper keeps Baseline2's original algorithm chain intact and writes to a
separate debug output folder by default, so quick USD/SAM/IK checks do not
pollute the full 14-trial evaluation results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from auto_baseline2_six_objects_refactored import BASELINE_OBJECT_CONFIGS


SCRIPT_DIR = Path(__file__).resolve().parent
FULL_EVAL_SCRIPT = SCRIPT_DIR / "auto_baseline2_six_objects_refactored.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast Baseline2 debug run: one trial per selected object/task/camera."
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=tuple(BASELINE_OBJECT_CONFIGS.keys()),
        default=tuple(BASELINE_OBJECT_CONFIGS.keys()),
        help="Objects to debug. Default: all configured objects.",
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        help="Only run these task names, such as drill_operate.",
    )
    parser.add_argument(
        "--cam-ids",
        nargs="+",
        type=int,
        default=list(range(1, 8)),
        help="Camera ids to debug. Default: 1..7.",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        default=1,
        help="Single trial id to run for each task/camera.",
    )
    parser.add_argument(
        "--output-root",
        default="batch_test_results_refactored/baseline2_debug_one_trial_views",
        help="Debug output root, separate from full evaluation results.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Camera width. Baseline2 defaults to the original 1280.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Camera height. Baseline2 defaults to the original 720.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip debug review images already present in the debug output folder.",
    )
    parser.add_argument(
        "--debug-visualize",
        action="store_true",
        help="Open IsaacSim GUI for visual debugging.",
    )
    parser.add_argument(
        "--verbose-subprocess",
        action="store_true",
        help="Print full child-process logs.",
    )
    parser.add_argument(
        "--save-step-images",
        action="store_true",
        help="Save step0/step1/step2 images in addition to review images.",
    )
    parser.add_argument("--contact-python", default="/home/zyp/anaconda3/envs/contact/bin/python")
    parser.add_argument("--cgn-worker", default=str(SCRIPT_DIR / "cgn_worker_baseline2.py"))
    parser.add_argument("--graspgpt-python", default="/home/zyp/anaconda3/envs/graspgpt/bin/python")
    parser.add_argument("--graspgpt-worker", default=str(SCRIPT_DIR / "graspgpt_worker_baseline2.py"))
    parser.add_argument("--cgn-timeout", type=int, default=420)
    parser.add_argument("--graspgpt-timeout", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(FULL_EVAL_SCRIPT),
        "--objects",
        *args.objects,
        "--output-root",
        args.output_root,
        "--log-dir",
        str(Path(args.output_root) / "logs"),
        "--trials-per-task",
        "1",
        "--trial-ids",
        str(args.trial_id),
        "--cam-ids",
        *(str(cam_id) for cam_id in args.cam_ids),
        "--chunk-level",
        "camera",
        "--fast-sim",
        "--cooldown-seconds",
        "0",
        "--camera-width",
        str(args.camera_width),
        "--camera-height",
        str(args.camera_height),
        "--contact-python",
        args.contact_python,
        "--cgn-worker",
        args.cgn_worker,
        "--graspgpt-python",
        args.graspgpt_python,
        "--graspgpt-worker",
        args.graspgpt_worker,
        "--cgn-timeout",
        str(args.cgn_timeout),
        "--graspgpt-timeout",
        str(args.graspgpt_timeout),
    ]
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

    print("🚀 启动 Baseline2 一轮视角调试，每个任务/视角只跑 1 次")
    print(f"📁 输出目录: {args.output_root}")
    print("🧾 命令:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(SCRIPT_DIR))


if __name__ == "__main__":
    raise SystemExit(main())
