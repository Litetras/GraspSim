#!/usr/bin/env python3
"""Six-object automated IsaacSim evaluation for Baseline2 / GraspGPT.

This keeps Baseline2's original algorithm chain intact:
SAM3 -> Contact-GraspNet candidate generation -> GraspGPT task ranking.
Only the batch orchestration and IsaacSim execution are shared with LODGrasp's
common evaluation layer.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOD_DIR = SCRIPT_DIR.parent / "LODGrasp"
if str(LOD_DIR) not in sys.path:
    sys.path.append(str(LOD_DIR))

from autoL_four_objects_refactored import OBJECT_CONFIGS as LOD_OBJECT_CONFIGS
from lod_eval_common import LodEvalConfig, LodEvalTask, run_lod_eval


BASELINE_OBJECT_CONFIGS = dict(LOD_OBJECT_CONFIGS)
BASELINE_OBJECT_CONFIGS["knife"] = {
    **LOD_OBJECT_CONFIGS["knife"],
    "scene_pattern": "cam{cam_id}.usd",
    "scene_id_pattern": "cam{cam_id}",
}

# GraspGPT 官方 TaskGrasp 词表和你的 12 个任务并不是一一同名。
# 这里显式保存映射，后续如果你想调整论文实验口径，只改这张表。
GRASPGPT_TASK_MAP = {
    "knife_cut": "cut",
    "knife_pass": "handover",
    "hammer_strike": "hammer",
    "hammer_pull": "hammer",
    "brush_clean": "clean",
    "brush_pass": "handover",
    "drill_operate": "screw",
    "drill_pass": "handover",
    "mug_pour": "pour",
    "mug_pass": "handover",
    "spoon_scoop": "scoop",
    "spoon_pass": "handover",
}

GRASPGPT_OBJECT_MAP = {
    "knife_cut": "knife",
    "knife_pass": "knife",
    "hammer_strike": "hammer",
    "hammer_pull": "hammer",
    "brush_clean": "brush",
    "brush_pass": "brush",
    # GraspGPT 官方 obj_gpt_v2 没有 drill，用 screwdriver 作为最近类别。
    "drill_operate": "screwdriver",
    "drill_pass": "screwdriver",
    "mug_pour": "mug",
    "mug_pass": "mug",
    # 官方没有普通 spoon 目录，tablespoon 比 wooden_spoon 更贴近你的勺子模型。
    "spoon_scoop": "tablespoon",
    "spoon_pass": "tablespoon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Baseline2/GraspGPT IsaacSim evaluations for six objects."
    )
    parser.add_argument("--objects", nargs="+", choices=tuple(BASELINE_OBJECT_CONFIGS.keys()), default=tuple(BASELINE_OBJECT_CONFIGS.keys()))
    parser.add_argument("--run-object", choices=tuple(BASELINE_OBJECT_CONFIGS.keys()), help=argparse.SUPPRESS)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--chunk-level", choices=("trial", "camera", "task", "object"), default="camera")
    parser.add_argument("--output-root", default="batch_test_results_refactored/baseline2_six_objects")
    parser.add_argument("--trials-per-task", type=int, default=14)
    parser.add_argument("--trial-ids", nargs="+", type=int, default=None)
    parser.add_argument("--cam-ids", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--keep-sam3-loaded", action="store_true")
    parser.add_argument("--sam3-in-process", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--fast-sim", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--pregrasp-steps", type=int, default=180)
    parser.add_argument("--insert-steps", type=int, default=80)
    parser.add_argument("--gripper-close-steps", type=int, default=80)
    parser.add_argument("--lift-move-steps", type=int, default=120)
    parser.add_argument("--lift-settle-steps", type=int, default=120)
    parser.add_argument("--log-dir", default="batch_test_results_refactored/baseline2_logs")
    parser.add_argument("--verbose-subprocess", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--debug-visualize", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-step-images", action="store_true")
    parser.add_argument("--contact-python", default="/home/zyp/anaconda3/envs/contact/bin/python")
    parser.add_argument("--cgn-worker", default=str(SCRIPT_DIR / "cgn_worker_baseline2.py"))
    parser.add_argument("--graspgpt-python", default="/home/zyp/anaconda3/envs/graspgpt/bin/python")
    parser.add_argument("--graspgpt-worker", default=str(SCRIPT_DIR / "graspgpt_worker_baseline2.py"))
    parser.add_argument("--cgn-timeout", type=int, default=420)
    parser.add_argument("--graspgpt-timeout", type=int, default=900)
    return parser.parse_args()


def apply_fast_sim_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not args.fast_sim:
        return args
    args.settle_steps = 50
    args.pregrasp_steps = 110
    args.insert_steps = 50
    args.gripper_close_steps = 45
    args.lift_move_steps = 80
    args.lift_settle_steps = 60
    return args


def make_config(object_name: str, args: argparse.Namespace) -> LodEvalConfig:
    spec = BASELINE_OBJECT_CONFIGS[object_name]
    tasks = spec["tasks"]
    if args.task_names:
        requested = set(args.task_names)
        tasks = tuple(task for task in tasks if task.task_name in requested)
        if not tasks:
            raise ValueError(f"No matching tasks for {object_name}: {sorted(requested)}")

    return LodEvalConfig(
        prompt=spec["prompt"],
        tasks=tasks,
        scene_pattern=spec["scene_pattern"],
        scene_id_pattern=spec["scene_id_pattern"],
        grasp_threshold=spec.get("grasp_threshold", 0.0),
        img_dir=str(Path(args.output_root) / object_name),
        baseline_name="Baseline2_GraspGPT",
        inference_backend="graspgpt",
        contact_graspnet_python=args.contact_python,
        contact_graspnet_worker=args.cgn_worker,
        contact_graspnet_timeout=args.cgn_timeout,
        graspgpt_python=args.graspgpt_python,
        graspgpt_worker=args.graspgpt_worker,
        graspgpt_timeout=args.graspgpt_timeout,
        graspgpt_task_map=GRASPGPT_TASK_MAP,
        graspgpt_object_map=GRASPGPT_OBJECT_MAP,
        # 与旧版 Baseline2 保持一致：插入执行点是 grasp_pos + grasp_dir * 0.115。
        close_distance_along_grasp=0.115,
        cam_ids=tuple(args.cam_ids),
        trials_per_task=args.trials_per_task,
        trial_ids=tuple(args.trial_ids) if args.trial_ids else None,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        unload_sam3_after_mask=not args.keep_sam3_loaded,
        sam3_subprocess=not args.sam3_in_process,
        settle_steps=args.settle_steps,
        pregrasp_steps=args.pregrasp_steps,
        insert_steps=args.insert_steps,
        gripper_close_steps=args.gripper_close_steps,
        lift_move_steps=args.lift_move_steps,
        lift_settle_steps=args.lift_settle_steps,
        headless=not args.debug_visualize,
        render_motion_steps=args.debug_visualize,
        save_step_images=args.save_step_images,
        split_output_by_task=True,
        skip_existing=args.skip_existing,
    )


def resolve_output_root(output_root: str, script_path: Path) -> Path:
    root = Path(output_root)
    return root if root.is_absolute() else script_path.parent / root


def review_image_path(args, script_path: Path, object_name: str, task: LodEvalTask, cam_id: int, trial_id: int) -> Path:
    root = resolve_output_root(args.output_root, script_path)
    scene_id = BASELINE_OBJECT_CONFIGS[object_name]["scene_id_pattern"].format(cam_id=cam_id)
    return root / object_name / task.task_name / "review" / f"trial_{trial_id:03d}_{scene_id}_{task.task_name}_review.jpg"


def missing_review_paths(args, script_path: Path, object_name: str, tasks, cam_ids, trial_ids):
    missing = []
    for task in tasks:
        for cam_id in cam_ids:
            for trial_id in trial_ids:
                path = review_image_path(args, script_path, object_name, task, cam_id, trial_id)
                if not path.exists():
                    missing.append(path)
    return missing


def child_log_path(args, script_path: Path, object_name: str, task_text: str, cam_text: str, trial_text: str) -> Path:
    log_root = resolve_output_root(args.log_dir, script_path)
    log_root.mkdir(parents=True, exist_ok=True)
    safe_name = f"baseline2_{object_name}_{task_text}_cam{cam_text}_trial{trial_text}".replace(",", "-")
    return log_root / f"{safe_name}.log"


def should_print_child_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    useful_tokens = (
        "▶️", "✅", "❌", "⚠️", "⏭️", "🎥", "🧾", "🌍", "🚀", "📊", "🧹",
        "正在进行 SAM3", "运行算法推理", "已保存审查图", "SAM3未检测到", "推理失败",
        "IK 求解失败", "所有", "计划总测试数", "任务映射", "最佳匹配概率",
    )
    return any(token in text for token in useful_tokens)


def run_child_process(cmd, log_path: Path, verbose: bool, env=None) -> int:
    last_lines = deque(maxlen=80)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            last_lines.append(line.rstrip())
            if verbose or should_print_child_line(line):
                print(line, end="")
        return_code = process.wait()

    if return_code != 0:
        print(f"📄 子进程完整日志: {log_path}")
        print("📄 子进程最后日志片段:")
        for line in last_lines:
            if line.strip():
                print(f"  {line}")
    return return_code


def get_selected_tasks(object_name: str, task_names):
    tasks = BASELINE_OBJECT_CONFIGS[object_name]["tasks"]
    if not task_names:
        return tasks
    requested = set(task_names)
    return tuple(task for task in tasks if task.task_name in requested)


def iter_chunks(args):
    for object_name in args.objects:
        tasks = get_selected_tasks(object_name, args.task_names)
        if not tasks:
            print(f"⚠️ {object_name} 没有匹配 --task-names 的任务，跳过。")
            continue
        if args.chunk_level == "object":
            yield object_name, tasks, tuple(args.cam_ids)
        elif args.chunk_level == "task":
            for task in tasks:
                yield object_name, (task,), tuple(args.cam_ids)
        elif args.chunk_level == "camera":
            for task in tasks:
                for cam_id in args.cam_ids:
                    yield object_name, (task,), (cam_id,)
        else:
            trial_ids = tuple(args.trial_ids or range(1, args.trials_per_task + 1))
            for task in tasks:
                for cam_id in args.cam_ids:
                    for trial_id in trial_ids:
                        yield object_name, (task,), (cam_id,), (trial_id,)


def print_output_layout(config: LodEvalConfig) -> None:
    print("📁 本轮输出目录:")
    for task in config.tasks:
        print(f"  - {Path(config.img_dir) / task.task_name}")


def print_review_summary(args, script_path: Path, object_names) -> None:
    root = resolve_output_root(args.output_root, script_path)
    expected_trials = len(args.trial_ids) if args.trial_ids else args.trials_per_task
    expected_per_task = len(args.cam_ids) * expected_trials
    print("\n📊 review 图片数量汇总:")
    for object_name in object_names:
        for task in BASELINE_OBJECT_CONFIGS[object_name]["tasks"]:
            review_dir = root / object_name / task.task_name / "review"
            count = len(list(review_dir.glob("*_review.jpg"))) if review_dir.exists() else 0
            print(f"  - {object_name}/{task.task_name}: {count}/{expected_per_task}")


def run_one_object(object_name: str, args: argparse.Namespace) -> None:
    config = make_config(object_name, args)
    print("\n" + "=" * 72)
    print(f"🚀 开始 Baseline2 评估物体: {object_name}")
    print(f"🧾 任务数: {len(config.tasks)} | 🎥 视角数: {len(config.cam_ids)} | 🔄 每任务每视角 trials: {config.trials_per_task}")
    print_output_layout(config)
    print("=" * 72)
    run_lod_eval(config)


def run_parent(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    failed = []
    start_time = time.time()

    for chunk in iter_chunks(args):
        if len(chunk) == 3:
            object_name, tasks, cam_ids = chunk
            trial_ids = tuple(args.trial_ids or range(1, args.trials_per_task + 1))
        else:
            object_name, tasks, cam_ids, trial_ids = chunk

        if args.skip_existing:
            missing = missing_review_paths(args, script_path, object_name, tasks, cam_ids, trial_ids)
            total = len(tasks) * len(cam_ids) * len(trial_ids)
            if not missing:
                print(
                    f"⏭️ 父进程整块跳过: object={object_name} "
                    f"task={','.join(t.task_name for t in tasks)} cam={','.join(map(str, cam_ids))} reviews={total}/{total}"
                )
                continue
            if len(missing) < total:
                print(f"↪️ 仅补跑缺失 review: {len(missing)}/{total} missing")

        cmd = [
            sys.executable,
            str(script_path),
            "--run-object", object_name,
            "--output-root", args.output_root,
            "--trials-per-task", str(args.trials_per_task),
            "--cam-ids", *(str(cam_id) for cam_id in cam_ids),
            "--task-names", *(task.task_name for task in tasks),
            "--trial-ids", *(str(trial_id) for trial_id in trial_ids),
            "--camera-width", str(args.camera_width),
            "--camera-height", str(args.camera_height),
            "--settle-steps", str(args.settle_steps),
            "--pregrasp-steps", str(args.pregrasp_steps),
            "--insert-steps", str(args.insert_steps),
            "--gripper-close-steps", str(args.gripper_close_steps),
            "--lift-move-steps", str(args.lift_move_steps),
            "--lift-settle-steps", str(args.lift_settle_steps),
            "--contact-python", args.contact_python,
            "--cgn-worker", args.cgn_worker,
            "--graspgpt-python", args.graspgpt_python,
            "--graspgpt-worker", args.graspgpt_worker,
            "--cgn-timeout", str(args.cgn_timeout),
            "--graspgpt-timeout", str(args.graspgpt_timeout),
        ]
        if args.debug_visualize:
            cmd.append("--debug-visualize")
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.save_step_images:
            cmd.append("--save-step-images")
        if args.keep_sam3_loaded:
            cmd.append("--keep-sam3-loaded")
        if args.sam3_in_process:
            cmd.append("--sam3-in-process")

        task_text = ",".join(task.task_name for task in tasks)
        cam_text = ",".join(str(cam_id) for cam_id in cam_ids)
        trial_text = ",".join(str(trial_id) for trial_id in trial_ids)
        print("\n" + "#" * 72)
        print(f"▶️  Baseline2 总脚本启动子任务: object={object_name} task={task_text} cam={cam_text} trial={trial_text}")
        print("#" * 72)
        log_path = child_log_path(args, script_path, object_name, task_text, cam_text, trial_text)
        print(f"📄 完整日志写入: {log_path}")

        old_cwd = os.getcwd()
        os.chdir(str(script_path.parent))
        try:
            return_code = run_child_process(cmd, log_path, args.verbose_subprocess, env=env)
        finally:
            os.chdir(old_cwd)

        if return_code != 0:
            failed.append(object_name)
            print(f"❌ Baseline2 子任务失败: object={object_name}, code={return_code}")
            if args.stop_on_error:
                return return_code
        else:
            print(f"✅ Baseline2 子任务完成: object={object_name} task={task_text} cam={cam_text} trial={trial_text}")
        print_review_summary(args, script_path, [object_name])
        if args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)

    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    if failed:
        print(f"⚠️ Baseline2 评估结束，但这些子任务失败: {', '.join(failed)}")
    else:
        print("🎉 Baseline2 选定物体评估全部完成。")
    print(f"📁 输出根目录: {args.output_root}")
    print(f"⏱️ 总耗时: {elapsed / 60.0:.1f} min")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    parsed = apply_fast_sim_preset(parse_args())
    if parsed.run_object:
        run_one_object(parsed.run_object, parsed)
    else:
        raise SystemExit(run_parent(parsed))
