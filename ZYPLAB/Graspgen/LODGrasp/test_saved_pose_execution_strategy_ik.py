#!/usr/bin/env python3
"""Sweep execution-side IK strategies for saved grasp poses.

This diagnostic does not run SAM3/GraspGen and does not edit USD files. It
keeps the predicted grasp pose fixed, then tests whether execution choices
around that pose make Franka IK feasible: shorter pregrasp offsets, world-z
lifting after closing, and optional height offsets that approximate a higher
object/table setup.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from test_saved_pose_height_lift_ik import (
    DEFAULT_LOG_DIR,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SCENE_DIR,
    apply_local_z_roll,
    bool_text,
    int_suffix,
    load_saved_pose_rows,
    parse_matrix,
    resolve_path,
)


DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "ik_execution_strategy_diagnostics.csv"


@dataclass(frozen=True)
class Strategy:
    name: str
    pre_distance: float | None
    lift_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep IK execution strategies for saved grasp poses.")
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
    parser.add_argument("--z-offsets", nargs="+", type=float, default=[0.0, 0.03, 0.05, 0.08, 0.10, 0.12])
    parser.add_argument("--close-distance", type=float, default=0.03)
    parser.add_argument("--lift-z-distance", type=float, default=0.08)
    parser.add_argument("--retreat-distance", type=float, default=0.08)
    parser.add_argument("--roll-angles-deg", nargs="+", type=float, default=[0.0])
    parser.add_argument("--stop-after-first-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every-trials", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_strategies() -> list[Strategy]:
    return [
        Strategy("original_pre010_retreat", pre_distance=0.10, lift_mode="approach_retreat"),
        Strategy("pre010_worldz", pre_distance=0.10, lift_mode="world_z"),
        Strategy("pre007_worldz", pre_distance=0.07, lift_mode="world_z"),
        Strategy("pre005_worldz", pre_distance=0.05, lift_mode="world_z"),
        Strategy("pre003_worldz", pre_distance=0.03, lift_mode="world_z"),
        Strategy("direct_close_worldz", pre_distance=None, lift_mode="world_z"),
    ]


def main() -> None:
    args = parse_args()
    rows = load_saved_pose_rows(args)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategies = build_strategies()
    candidates = [
        (z_offset, roll_deg, strategy)
        for z_offset in args.z_offsets
        for roll_deg in args.roll_angles_deg
        for strategy in strategies
    ]
    print(f"Loaded saved poses: {len(rows)}")
    print(f"Only log IK failures: {args.only_log_ik_failures}")
    print(f"Strategy candidates per pose: {len(candidates)}")
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
            "strategy", "z_offset", "pre_distance", "close_distance",
            "lift_mode", "lift_z_distance", "retreat_distance", "roll_deg",
            "pre_ik", "close_ik", "lift_ik",
            "pre_x", "pre_y", "pre_z", "close_x", "close_y", "close_z",
            "lift_x", "lift_y", "lift_z", "original_close_z",
            "approach_x", "approach_y", "approach_z",
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
            print(f"\n[{scene_index}/{len(grouped)}] Loading scene: {scene_path}", flush=True)
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
                base_grasp_pos = matrix[:3, 3].astype(float)
                grasp_dir = matrix[:3, 2].astype(float)
                grasp_dir = grasp_dir / np.linalg.norm(grasp_dir)
                original_close_pos = base_grasp_pos + grasp_dir * args.close_distance
                trial_success = False
                tried = 0

                for z_offset, roll_deg, strategy in candidates:
                    tried += 1
                    reset_robot(franka, initial_joints, world)
                    raised_grasp_pos = base_grasp_pos + np.array([0.0, 0.0, z_offset], dtype=float)
                    target_rot = apply_local_z_roll(matrix[:3, :3], roll_deg)
                    target_quat = rot_matrix_to_quat(target_rot)

                    pre_pos = None
                    if strategy.pre_distance is not None:
                        pre_pos = raised_grasp_pos - grasp_dir * strategy.pre_distance
                    close_pos = raised_grasp_pos + grasp_dir * args.close_distance
                    if strategy.lift_mode == "world_z":
                        lift_pos = close_pos + np.array([0.0, 0.0, args.lift_z_distance], dtype=float)
                    elif strategy.lift_mode == "approach_retreat":
                        lift_pos = raised_grasp_pos - grasp_dir * args.retreat_distance
                    else:
                        raise ValueError(f"Unknown lift mode: {strategy.lift_mode}")

                    pre_ok = True
                    close_ok = False
                    lift_ok = False
                    fail_stage = ""
                    if pre_pos is not None:
                        pre_ok = solve_and_apply(ik_solver, franka, world, pre_pos, target_quat)
                        if not pre_ok:
                            fail_stage = "pre"
                    if pre_ok:
                        close_ok = solve_and_apply(ik_solver, franka, world, close_pos, target_quat)
                        if not close_ok:
                            fail_stage = "close"
                    if close_ok:
                        lift_ok = solve_and_apply(ik_solver, franka, world, lift_pos, target_quat)
                        if not lift_ok:
                            fail_stage = "lift"

                    success = pre_ok and close_ok and lift_ok
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
                            "strategy": strategy.name,
                            "z_offset": f"{z_offset:.3f}",
                            "pre_distance": "" if strategy.pre_distance is None else f"{strategy.pre_distance:.3f}",
                            "close_distance": f"{args.close_distance:.3f}",
                            "lift_mode": strategy.lift_mode,
                            "lift_z_distance": f"{args.lift_z_distance:.3f}",
                            "retreat_distance": f"{args.retreat_distance:.3f}",
                            "roll_deg": f"{roll_deg:.1f}",
                            "pre_ik": bool_text(pre_ok),
                            "close_ik": bool_text(close_ok),
                            "lift_ik": bool_text(lift_ok),
                            "pre_x": "" if pre_pos is None else f"{pre_pos[0]:.6f}",
                            "pre_y": "" if pre_pos is None else f"{pre_pos[1]:.6f}",
                            "pre_z": "" if pre_pos is None else f"{pre_pos[2]:.6f}",
                            "close_x": f"{close_pos[0]:.6f}",
                            "close_y": f"{close_pos[1]:.6f}",
                            "close_z": f"{close_pos[2]:.6f}",
                            "lift_x": f"{lift_pos[0]:.6f}",
                            "lift_y": f"{lift_pos[1]:.6f}",
                            "lift_z": f"{lift_pos[2]:.6f}",
                            "original_close_z": f"{original_close_pos[2]:.6f}",
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

        print("\nExecution-strategy IK diagnostic coverage:")
        print(f"  any success: {len(trial_successes)}/{len(all_trials)} ({len(trial_successes) / len(all_trials):.1%})")
        by_task_total = Counter((key[0], key[1]) for key in all_trials)
        by_task_success = Counter((key[0], key[1]) for key in trial_successes)
        for task_key in sorted(by_task_total):
            total = by_task_total[task_key]
            success = by_task_success[task_key]
            print(f"  {task_key[0]}/{task_key[1]}: {success}/{total} ({success / total:.1%})")

        chosen_rows = [success_rows[0] for success_rows in trial_successes.values()]
        print("\nChosen strategies:")
        for strategy_name, count in Counter(row["strategy"] for row in chosen_rows).most_common():
            print(f"  {strategy_name}: {count}")
        print("\nChosen height offsets:")
        for z_offset, count in sorted(Counter(row["z_offset"] for row in chosen_rows).items(), key=lambda item: float(item[0])):
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
