import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from lod_eval_common import LodEvalConfig, LodEvalTask, run_lod_eval


OBJECT_CONFIGS = {
    "brush": {
        "prompt": "brush",
        "tasks": (
            LodEvalTask("brush_clean", "Grasp the brush to clean."),
            LodEvalTask("brush_pass", "Grasp the brush to pass."),
        ),
        "scene_pattern": "brush_cam{cam_id}.usd",
        "scene_id_pattern": "brush_cam{cam_id}",
        "grasp_threshold": 0.6,
    },
    "drill": {
        "prompt": "drill",
        "tasks": (
            LodEvalTask("drill_operate", "Grasp the drill to operate."),
            LodEvalTask("drill_pass", "Grasp the drill to pass."),
        ),
        "scene_pattern": "drill_cam{cam_id}.usd",
        "scene_id_pattern": "drill_cam{cam_id}",
        "grasp_threshold": 0.45,############
    },
    "mug": {
        "prompt": "mug",
        "tasks": (
            LodEvalTask("mug_pour", "Grasp the mug to pour."),
            LodEvalTask("mug_pass", "Grasp the mug to pass."),
        ),
        "scene_pattern": "mug_cam{cam_id}.usd",
        "scene_id_pattern": "mug_cam{cam_id}",
        "grasp_threshold": 0.6,
    },
    "spoon": {
        "prompt": "spoon",
        "tasks": (
            LodEvalTask("spoon_scoop", "Grasp the spoon to scoop."),
            LodEvalTask("spoon_pass", "Grasp the spoon to pass."),
        ),
        "scene_pattern": "spoon_cam{cam_id}.usd",
        "scene_id_pattern": "spoon_cam{cam_id}",
        "grasp_threshold": 0.6,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LODGrasp IsaacSim evaluations for brush, drill, mug, and spoon."
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=tuple(OBJECT_CONFIGS.keys()),
        default=tuple(OBJECT_CONFIGS.keys()),
        help="Objects to run in order.",
    )
    parser.add_argument(
        "--run-object",
        choices=tuple(OBJECT_CONFIGS.keys()),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        help="Only run these task names. Useful for resuming or debugging one task.",
    )
    parser.add_argument(
        "--chunk-level",
        choices=("trial", "camera", "task", "object"),
        default="camera",
        help=(
            "How the parent script splits subprocesses. 'camera' reuses one IsaacSim "
            "process for all trials of one object-task-camera chunk. Use 'trial' if "
            "memory pressure returns."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="batch_test_results_refactored/four_objects",
        help="Root output directory. Results are saved as output_root/object/task/.",
    )
    parser.add_argument(
        "--trials-per-task",
        type=int,
        default=14,
        help="Trials per task for each camera view.",
    )
    parser.add_argument(
        "--trial-ids",
        nargs="+",
        type=int,
        default=None,
        help="Only run these trial ids. Internal use for trial-level subprocess chunks.",
    )
    parser.add_argument(
        "--cam-ids",
        nargs="+",
        type=int,
        default=list(range(1, 8)),
        help="Camera ids to evaluate.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Camera RGB-D width. Lower values reduce SAM3/IsaacSim memory.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=360,
        help="Camera RGB-D height. Lower values reduce SAM3/IsaacSim memory.",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=120,
        help="Number of grasp candidates generated per inference. Lower values reduce memory/time.",
    )
    parser.add_argument(
        "--keep-sam3-loaded",
        action="store_true",
        help="Keep SAM3 in memory between trials. Faster, but uses more memory.",
    )
    parser.add_argument(
        "--sam3-in-process",
        action="store_true",
        help="Run SAM3 inside the IsaacSim process. Default isolates SAM3 in a subprocess to lower peak memory.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=2.0,
        help="Sleep after each subprocess so OS/GPU resources can settle.",
    )
    parser.add_argument(
        "--fast-sim",
        action="store_true",
        help="Use fewer physics/control steps for faster batch evaluation.",
    )
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--pregrasp-steps", type=int, default=180)
    parser.add_argument("--insert-steps", type=int, default=80)
    parser.add_argument("--gripper-close-steps", type=int, default=80)
    parser.add_argument("--lift-move-steps", type=int, default=120)
    parser.add_argument("--lift-settle-steps", type=int, default=120)
    parser.add_argument(
        "--log-dir",
        default="batch_test_results_refactored/logs",
        help="Directory for full child-process logs.",
    )
    parser.add_argument(
        "--verbose-subprocess",
        action="store_true",
        help="Print raw child-process logs, including IsaacSim startup spam.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one object process fails.",
    )
    parser.add_argument(
        "--debug-visualize",
        action="store_true",
        help="Open IsaacSim GUI and render motion steps for visual debugging.",
    )
    parser.add_argument(
        "--enable-meshcat",
        action="store_true",
        help="Also enable MeshCat visualization inside GraspGen inference.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trials whose review image already exists. Useful for resuming interrupted runs.",
    )
    parser.add_argument(
        "--save-step-images",
        action="store_true",
        help="Also save the separate step0/step1/step2 PNGs. By default only review images are saved.",
    )
    return parser.parse_args()


def apply_fast_sim_preset(args):
    if not args.fast_sim:
        return args
    args.settle_steps = 50
    args.pregrasp_steps = 110
    args.insert_steps = 50
    args.gripper_close_steps = 45
    args.lift_move_steps = 80
    args.lift_settle_steps = 60
    return args


def make_config(object_name: str, args) -> LodEvalConfig:
    spec = OBJECT_CONFIGS[object_name]
    tasks = spec["tasks"]
    if args.task_names:
        requested_tasks = set(args.task_names)
        tasks = tuple(task for task in tasks if task.task_name in requested_tasks)
        if not tasks:
            raise ValueError(f"No matching tasks for {object_name}: {sorted(requested_tasks)}")

    return LodEvalConfig(
        prompt=spec["prompt"],
        tasks=tasks,
        scene_pattern=spec["scene_pattern"],
        scene_id_pattern=spec["scene_id_pattern"],
        grasp_threshold=spec["grasp_threshold"],
        img_dir=str(Path(args.output_root) / object_name),
        cam_ids=tuple(args.cam_ids),
        trials_per_task=args.trials_per_task,
        trial_ids=tuple(args.trial_ids) if args.trial_ids else None,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        num_grasps=args.num_grasps,
        unload_sam3_after_mask=not args.keep_sam3_loaded,
        sam3_subprocess=not args.sam3_in_process,
        settle_steps=args.settle_steps,
        pregrasp_steps=args.pregrasp_steps,
        insert_steps=args.insert_steps,
        gripper_close_steps=args.gripper_close_steps,
        lift_move_steps=args.lift_move_steps,
        lift_settle_steps=args.lift_settle_steps,
        headless=not args.debug_visualize,
        enable_meshcat=args.enable_meshcat,
        render_motion_steps=args.debug_visualize,
        save_step_images=args.save_step_images,
        split_output_by_task=True,
        skip_existing=args.skip_existing,
        object_prim_path=None,
    )


def print_output_layout(config: LodEvalConfig):
    print("📁 本轮输出目录:")
    for task in config.tasks:
        print(f"  - {Path(config.img_dir) / task.task_name}")


def resolve_output_root(output_root: str, script_path: Path) -> Path:
    root = Path(output_root)
    if root.is_absolute():
        return root
    return script_path.parent / root


def print_review_summary(args, script_path: Path, object_names):
    root = resolve_output_root(args.output_root, script_path)
    expected_trials = len(args.trial_ids) if args.trial_ids else args.trials_per_task
    expected_per_task = len(args.cam_ids) * expected_trials
    print("\n📊 review 图片数量汇总:")
    for object_name in object_names:
        for task in OBJECT_CONFIGS[object_name]["tasks"]:
            review_dir = root / object_name / task.task_name / "review"
            count = len(list(review_dir.glob("*_review.jpg"))) if review_dir.exists() else 0
            print(f"  - {object_name}/{task.task_name}: {count}/{expected_per_task}")


def review_image_path(args, script_path: Path, object_name: str, task: LodEvalTask, cam_id: int, trial_id: int) -> Path:
    root = resolve_output_root(args.output_root, script_path)
    scene_id = OBJECT_CONFIGS[object_name]["scene_id_pattern"].format(cam_id=cam_id)
    filename = f"trial_{trial_id:03d}_{scene_id}_{task.task_name}_review.jpg"
    return root / object_name / task.task_name / "review" / filename


def child_log_path(args, script_path: Path, object_name: str, task_text: str, cam_text: str, trial_text: str) -> Path:
    log_root = resolve_output_root(args.log_dir, script_path)
    log_root.mkdir(parents=True, exist_ok=True)
    safe_name = f"{object_name}_{task_text}_cam{cam_text}_trial{trial_text}".replace(",", "-")
    return log_root / f"{safe_name}.log"


def should_print_child_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    useful_tokens = (
        "▶️", "✅", "❌", "⚠️", "⏭️", "🎥", "🧾", "🌍", "🚀", "📊", "🧹",
        "Starting collision-free", "Collision checking", "正在进行 SAM3", "运行大模型推理",
        "已保存审查图", "SAM3未检测到", "推理失败", "IK 求解失败", "所有", "计划总测试数",
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
    tasks = OBJECT_CONFIGS[object_name]["tasks"]
    if not task_names:
        return tasks
    requested_tasks = set(task_names)
    return tuple(task for task in tasks if task.task_name in requested_tasks)


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


def run_one_object(object_name: str, args):
    config = make_config(object_name, args)
    print("\n" + "=" * 72)
    print(f"🚀 开始评估物体: {object_name}")
    print(f"🧾 任务数: {len(config.tasks)} | 🎥 视角数: {len(config.cam_ids)} | 🔄 每任务每视角 trials: {config.trials_per_task}")
    print_output_layout(config)
    print("=" * 72)
    run_lod_eval(config)


def run_parent(args) -> int:
    script_path = Path(__file__).resolve()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    failed_objects = []
    start_time = time.time()

    for chunk in iter_chunks(args):
        if len(chunk) == 3:
            object_name, tasks, cam_ids = chunk
            trial_ids = tuple(args.trial_ids or range(1, args.trials_per_task + 1))
        else:
            object_name, tasks, cam_ids, trial_ids = chunk

        if args.skip_existing and len(tasks) == 1 and len(cam_ids) == 1 and len(trial_ids) == 1:
            existing_review = review_image_path(args, script_path, object_name, tasks[0], cam_ids[0], trial_ids[0])
            if existing_review.exists():
                print(f"⏭️ 父进程跳过已完成 trial: {existing_review}")
                continue

        cmd = [
            sys.executable,
            str(script_path),
            "--run-object",
            object_name,
            "--output-root",
            args.output_root,
            "--trials-per-task",
            str(args.trials_per_task),
            "--cam-ids",
            *(str(cam_id) for cam_id in cam_ids),
            "--task-names",
            *(task.task_name for task in tasks),
            "--trial-ids",
            *(str(trial_id) for trial_id in trial_ids),
            "--camera-width",
            str(args.camera_width),
            "--camera-height",
            str(args.camera_height),
            "--num-grasps",
            str(args.num_grasps),
            "--settle-steps",
            str(args.settle_steps),
            "--pregrasp-steps",
            str(args.pregrasp_steps),
            "--insert-steps",
            str(args.insert_steps),
            "--gripper-close-steps",
            str(args.gripper_close_steps),
            "--lift-move-steps",
            str(args.lift_move_steps),
            "--lift-settle-steps",
            str(args.lift_settle_steps),
        ]
        if args.debug_visualize:
            cmd.append("--debug-visualize")
        if args.enable_meshcat:
            cmd.append("--enable-meshcat")
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.save_step_images:
            cmd.append("--save-step-images")
        if args.keep_sam3_loaded:
            cmd.append("--keep-sam3-loaded")
        if args.sam3_in_process:
            cmd.append("--sam3-in-process")

        print("\n" + "#" * 72)
        task_text = ",".join(task.task_name for task in tasks)
        cam_text = ",".join(str(cam_id) for cam_id in cam_ids)
        trial_text = ",".join(str(trial_id) for trial_id in trial_ids)
        print(f"▶️  总脚本正在启动子任务: object={object_name} task={task_text} cam={cam_text} trial={trial_text}")
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
            failed_objects.append(object_name)
            print(f"❌ {object_name} 子任务失败，退出码: {return_code}")
            if args.stop_on_error:
                print("已停止后续物体。默认模式会继续跑完剩余物体。")
                return return_code
        else:
            print(f"✅ 子任务完成: object={object_name} task={task_text} cam={cam_text} trial={trial_text}")
        print_review_summary(args, script_path, [object_name])
        if args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)

    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    if failed_objects:
        print(f"⚠️  总评估结束，但这些物体失败: {', '.join(failed_objects)}")
    else:
        print("🎉 四个物体评估全部完成。")
    print(f"📁 输出根目录: {args.output_root}")
    print(f"⏱️ 总耗时: {elapsed / 60.0:.1f} min")
    print("=" * 72)
    return 1 if failed_objects else 0


if __name__ == "__main__":
    parsed_args = apply_fast_sim_preset(parse_args())
    if parsed_args.run_object:
        run_one_object(parsed_args.run_object, parsed_args)
    else:
        raise SystemExit(run_parent(parsed_args))
