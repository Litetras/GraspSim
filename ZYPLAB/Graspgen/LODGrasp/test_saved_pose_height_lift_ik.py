#!/usr/bin/env python3
"""Test whether raising saved grasp poses improves Franka IK feasibility.

This is a diagnostic script, not part of the LODGrasp evaluation pipeline. It
does not run SAM3/GraspGen and does not modify the original USD files. It reads
saved `best_pose_exec_world` matrices and translates the whole target grasp
pose upward in world z, which approximates testing the same grasp if the object
or table were higher.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "batch_test_results_refactored" / "four_objects"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "ik_height_lift_diagnostics.csv"
DEFAULT_SCENE_DIR = Path("/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib")
DEFAULT_LOG_DIR = SCRIPT_DIR / "batch_test_results_refactored" / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test IK feasibility after raising saved grasp poses.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--cam-ids", nargs="+", type=int, default=None)
    parser.add_argument("--trial-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--only-log-ik-failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--z-offsets", nargs="+", type=float, default=[0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
    parser.add_argument("--close-distance", type=float, default=0.03)
    parser.add_argument("--lift-z-distance", type=float, default=0.08)
    parser.add_argument("--roll-angles-deg", nargs="+", type=float, default=[0.0])
    parser.add_argument("--stop-after-first-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every-trials", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else SCRIPT_DIR / path


def int_suffix(text: str) -> int:
    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else 0


def parse_matrix(text: str) -> np.ndarray | None:
    if not text:
        return None
    try:
        matrix = np.array(ast.literal_eval(text), dtype=float)
    except Exception:
        return None
    if matrix.shape != (4, 4):
        return None
    return matrix


def parse_logs_for_ik_failures(log_dir: Path) -> set[tuple[str, str, str, str]]:
    trial_re = re.compile(r"场景:\s*([^|]+)\|\s*🧾 任务:\s*([^|]+)\|\s*🔄 测试轮次:\s*(\d+)\s*/")
    keys: set[tuple[str, str, str, str]] = set()
    log_dir = resolve_path(log_dir)
    if not log_dir.exists():
        return keys

    for log_path in sorted(log_dir.glob("*.log")):
        scene_id = task_name = trial_id = None
        with log_path.open(errors="replace") as handle:
            for line in handle:
                match = trial_re.search(line)
                if match:
                    scene_id = match.group(1).strip()
                    task_name = match.group(2).strip()
                    trial_id = match.group(3).strip()
                if "IK 求解失败" in line and scene_id and task_name and trial_id:
                    keys.add((scene_id.split("_cam")[0], task_name, scene_id, trial_id))
    return keys


def load_saved_pose_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    results_root = resolve_path(args.results_root)
    object_filter = set(args.objects) if args.objects else None
    task_filter = set(args.task_names) if args.task_names else None
    cam_filter = {f"cam{cam_id}" for cam_id in args.cam_ids} if args.cam_ids else None
    trial_filter = {str(trial_id) for trial_id in args.trial_ids} if args.trial_ids else None
    latest: OrderedDict[tuple[str, str, str, str], dict[str, str]] = OrderedDict()

    for csv_path in sorted(results_root.glob("*/*/trial_results.csv")):
        object_name = csv_path.parts[-3]
        task_name = csv_path.parts[-2]
        if object_filter and object_name not in object_filter:
            continue
        if task_filter and task_name not in task_filter:
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                scene_id = row.get("scene_id", "")
                trial_id = row.get("trial_id", "")
                if cam_filter and f"cam{int_suffix(scene_id)}" not in cam_filter:
                    continue
                if trial_filter and trial_id not in trial_filter:
                    continue
                if not row.get("best_pose_exec_world"):
                    continue
                row["_object"] = object_name
                row["_task"] = task_name
                latest[(object_name, task_name, scene_id, trial_id)] = row

    rows = list(latest.values())
    rows.sort(
        key=lambda row: (
            row["_object"],
            row["_task"],
            int_suffix(row["scene_id"]),
            int(row["trial_id"]) if row["trial_id"].isdigit() else 0,
        )
    )

    if args.only_log_ik_failures:
        ik_keys = parse_logs_for_ik_failures(args.log_dir)
        rows = [
            row
            for row in rows
            if (row["_object"], row["_task"], row["scene_id"], row["trial_id"]) in ik_keys
        ]
    if args.max_items is not None:
        rows = rows[: args.max_items]
    return rows


def local_z_roll_matrix(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    return np.array(
        [
            [cos_v, -sin_v, 0.0],
            [sin_v, cos_v, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def apply_local_z_roll(rotation: np.ndarray, degrees: float) -> np.ndarray:
    return rotation @ local_z_roll_matrix(degrees)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    args = parse_args()
    rows = load_saved_pose_rows(args)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = [(z_offset, roll_deg) for z_offset in args.z_offsets for roll_deg in args.roll_angles_deg]
    print(f"Loaded saved poses: {len(rows)}")
    print(f"Only log IK failures: {args.only_log_ik_failures}")
    print(f"Height candidates per pose: {len(candidates)}")
    print(f"Output: {output_path}")
    if not rows:
        return

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        from isaacsim.core.api import World
        from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver
        from omni.isaac.core.utils.rotations import rot_matrix_to_quat
        from omni.isaac.core.utils.stage import open_stage
        from omni.isaac.core.utils.types import ArticulationAction

        grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["scene_id"]].append(row)

        fieldnames = [
            "object", "task", "scene_id", "trial_id", "success", "fail_stage",
            "z_offset", "close_distance", "lift_z_distance", "roll_deg",
            "close_ik", "lift_ik", "target_x", "target_y", "target_z",
            "original_target_z", "approach_x", "approach_y", "approach_z",
            "grasp_score", "collision_free_count", "review_image",
        ]
        results: list[dict[str, Any]] = []

        def write_snapshot() -> None:
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        def reset_robot(franka, initial_joints, world) -> None:
            franka.set_joint_positions(initial_joints)
            world.step()

        def solve_and_apply(ik_solver, franka, world, target_pos, target_quat) -> bool:
            action, success = ik_solver.compute_inverse_kinematics(
                target_position=target_pos,
                target_orientation=target_quat,
            )
            if not success:
                return False
            current_joints = franka.get_joint_positions()
            target_joints = np.copy(current_joints)
            target_joints[:7] = action.joint_positions
            franka.apply_action(ArticulationAction(joint_positions=target_joints))
            world.step()
            return True

        total_trials = sum(len(scene_rows) for scene_rows in grouped.values())
        processed = 0
        start = time.monotonic()

        for scene_index, (scene_id, scene_rows) in enumerate(sorted(grouped.items(), key=lambda item: int_suffix(item[0])), 1):
            scene_path = resolve_path(args.scene_dir) / f"{scene_id}.usd"
            print(f"\n[{scene_index}/{len(grouped)}] Loading scene: {scene_path}")
            if World.instance() is not None:
                World.instance().clear_instance()
            open_stage(str(scene_path))
            world = World()
            franka = world.scene.add(Franka(prim_path="/Franka", name="franka"))
            world.reset()
            for _ in range(args.settle_steps):
                world.step()
            initial_joints = np.copy(franka.get_joint_positions())
            ik_solver = KinematicsSolver(robot_articulation=franka)

            for row in scene_rows:
                processed += 1
                trial_start = time.monotonic()
                matrix = parse_matrix(row.get("best_pose_exec_world", ""))
                if matrix is None:
                    continue
                grasp_pos = matrix[:3, 3].astype(float)
                grasp_dir = matrix[:3, 2].astype(float)
                original_close_pos = grasp_pos + grasp_dir * args.close_distance
                trial_success = False
                tried = 0

                for z_offset, roll_deg in candidates:
                    tried += 1
                    reset_robot(franka, initial_joints, world)
                    target_rot = apply_local_z_roll(matrix[:3, :3], roll_deg)
                    target_quat = rot_matrix_to_quat(target_rot)
                    close_pos = original_close_pos + np.array([0.0, 0.0, z_offset], dtype=float)
                    lift_pos = close_pos + np.array([0.0, 0.0, args.lift_z_distance], dtype=float)

                    close_ok = solve_and_apply(ik_solver, franka, world, close_pos, target_quat)
                    lift_ok = False
                    fail_stage = ""
                    if close_ok:
                        lift_ok = solve_and_apply(ik_solver, franka, world, lift_pos, target_quat)
                        if not lift_ok:
                            fail_stage = "lift"
                    else:
                        fail_stage = "close"
                    success = close_ok and lift_ok
                    if success:
                        trial_success = True

                    results.append(
                        {
                            "object": row["_object"],
                            "task": row["_task"],
                            "scene_id": row["scene_id"],
                            "trial_id": row["trial_id"],
                            "success": bool_text(success),
                            "fail_stage": fail_stage,
                            "z_offset": f"{z_offset:.3f}",
                            "close_distance": f"{args.close_distance:.3f}",
                            "lift_z_distance": f"{args.lift_z_distance:.3f}",
                            "roll_deg": f"{roll_deg:.1f}",
                            "close_ik": bool_text(close_ok),
                            "lift_ik": bool_text(lift_ok),
                            "target_x": f"{close_pos[0]:.6f}",
                            "target_y": f"{close_pos[1]:.6f}",
                            "target_z": f"{close_pos[2]:.6f}",
                            "original_target_z": f"{original_close_pos[2]:.6f}",
                            "approach_x": f"{grasp_dir[0]:.6f}",
                            "approach_y": f"{grasp_dir[1]:.6f}",
                            "approach_z": f"{grasp_dir[2]:.6f}",
                            "grasp_score": row.get("grasp_score", ""),
                            "collision_free_count": row.get("collision_free_count", ""),
                            "review_image": row.get("review_image", ""),
                        }
                    )
                    if success and args.stop_after_first_success:
                        break

                if args.progress_every_trials > 0 and processed % args.progress_every_trials == 0:
                    elapsed = time.monotonic() - start
                    print(
                        f"  [{processed}/{total_trials}] {row['_object']}/{row['_task']} "
                        f"{row['scene_id']} trial={row['trial_id']} "
                        f"{'success' if trial_success else 'failed'}; tried={tried}/{len(candidates)}; "
                        f"trial_time={time.monotonic() - trial_start:.1f}s; elapsed={elapsed / 60:.1f}min",
                        flush=True,
                    )
            write_snapshot()
            print(f"  snapshot saved: {output_path} ({len(results)} rows)", flush=True)

        trial_successes: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        all_trials: set[tuple[str, str, str, str]] = set()
        for result in results:
            key = (result["object"], result["task"], result["scene_id"], result["trial_id"])
            all_trials.add(key)
            if result["success"] == "true":
                trial_successes[key].append(result)

        print("\nHeight-lift IK diagnostic coverage:")
        print(f"  any success: {len(trial_successes)}/{len(all_trials)} ({len(trial_successes) / len(all_trials):.1%})")
        by_task_total = Counter((key[0], key[1]) for key in all_trials)
        by_task_success = Counter((key[0], key[1]) for key in trial_successes)
        for task_key in sorted(by_task_total):
            total = by_task_total[task_key]
            success = by_task_success[task_key]
            print(f"  {task_key[0]}/{task_key[1]}: {success}/{total} ({success / total:.1%})")

        chosen_rows = [success_rows[0] for success_rows in trial_successes.values()]
        print("\nChosen height offsets:")
        for z_offset, count in Counter(row["z_offset"] for row in chosen_rows).most_common():
            print(f"  z_offset={z_offset}: {count}")

        never = all_trials - set(trial_successes)
        if never:
            print("\nStill unreachable:")
            for key, count in sorted(Counter((item[0], item[1], item[2]) for item in never).items()):
                print(f"  {key}: {count}")
        print(f"\nSaved: {output_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
