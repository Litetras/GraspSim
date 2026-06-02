#!/usr/bin/env python3
"""Analyze IK failures by grasp approach direction.

This script is read-only with respect to IsaacSim scenes: it does not start
IsaacSim, does not run perception, and does not modify USD files. It reads the
saved LODGrasp CSV results plus logs, then groups trials by the world-frame
approach direction of the saved 6-DoF grasp.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "batch_test_results_refactored" / "four_objects"
DEFAULT_LOG_DIR = SCRIPT_DIR / "batch_test_results_refactored" / "logs"
DEFAULT_DIAG_CSV = DEFAULT_RESULTS_ROOT / "ik_workcell_diagnostics.csv"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "approach_ik_analysis.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group IK failures by saved grasp approach direction.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--diagnostics-csv", type=Path, default=DEFAULT_DIAG_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizontal-abs-z", type=float, default=0.35)
    parser.add_argument("--vertical-abs-z", type=float, default=0.75)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--task-names", nargs="+", default=None)
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


def parse_vector(text: str) -> np.ndarray | None:
    if not text:
        return None
    try:
        vector = np.array(ast.literal_eval(text), dtype=float)
    except Exception:
        return None
    if vector.shape != (3,):
        return None
    return vector


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def classify_approach(vector: np.ndarray, horizontal_abs_z: float, vertical_abs_z: float) -> str:
    z = float(vector[2])
    abs_z = abs(z)
    if abs_z < horizontal_abs_z:
        return "horizontal"
    if abs_z >= vertical_abs_z:
        return "vertical_up" if z > 0 else "vertical_down"
    return "oblique_up" if z > 0 else "oblique_down"


def azimuth_bin(vector: np.ndarray) -> str:
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    if -45.0 <= angle < 45.0:
        return "+x"
    if 45.0 <= angle < 135.0:
        return "+y"
    if -135.0 <= angle < -45.0:
        return "-y"
    return "-x"


def load_trials(args: argparse.Namespace) -> list[dict[str, str]]:
    results_root = resolve_path(args.results_root)
    object_filter = set(args.objects) if args.objects else None
    task_filter = set(args.task_names) if args.task_names else None
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
                if not row.get("best_pose_exec_world"):
                    continue
                row["_object"] = object_name
                row["_task"] = task_name
                latest[(object_name, task_name, row.get("scene_id", ""), row.get("trial_id", ""))] = row

    rows = list(latest.values())
    rows.sort(
        key=lambda row: (
            row["_object"],
            row["_task"],
            int_suffix(row.get("scene_id", "")),
            int(row.get("trial_id", "0")) if row.get("trial_id", "").isdigit() else 0,
        )
    )
    return rows


def parse_log_ik_failures(log_dir: Path) -> set[tuple[str, str, str, str]]:
    log_dir = resolve_path(log_dir)
    trial_re = re.compile(r"场景:\s*([^|]+)\|\s*🧾 任务:\s*([^|]+)\|\s*🔄 测试轮次:\s*(\d+)\s*/")
    failures: set[tuple[str, str, str, str]] = set()
    if not log_dir.exists():
        return failures

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
                    object_name = scene_id.split("_cam")[0]
                    failures.add((object_name, task_name, scene_id, trial_id))
    return failures


def load_diagnostic_successes(path: Path) -> dict[tuple[str, str, str, str], bool]:
    path = resolve_path(path)
    if not path.exists():
        return {}
    status: dict[tuple[str, str, str, str], bool] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["object"], row["task"], row["scene_id"], row["trial_id"])
            status[key] = status.get(key, False) or row.get("success") == "true"
    return status


def yes_no(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def print_group_summary(title: str, rows: list[dict[str, str]], group_fields: tuple[str, ...]) -> None:
    groups: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)

    print(f"\n{title}")
    for key, items in sorted(groups.items()):
        total = len(items)
        ik_failed = sum(item["ik_failed_log"] == "true" for item in items)
        diag_success_known = [item for item in items if item["diag_success_any"]]
        diag_success = sum(item["diag_success_any"] == "true" for item in diag_success_known)
        task_pose_known = [item for item in items if item["pose_correct_manual"]]
        task_pose = sum(item["pose_correct_manual"] == "true" for item in task_pose_known)
        diag_text = "n/a" if not diag_success_known else f"{diag_success}/{len(diag_success_known)} ({diag_success / len(diag_success_known):.1%})"
        pose_text = "n/a" if not task_pose_known else f"{task_pose}/{len(task_pose_known)} ({task_pose / len(task_pose_known):.1%})"
        print(
            f"  {'/'.join(key):32s} "
            f"count={total:3d} ik_fail={ik_failed:3d}/{total:<3d} ({ik_failed / total:5.1%}) "
            f"diag_rescue={diag_text:18s} pose_correct={pose_text}"
        )


def main() -> None:
    args = parse_args()
    trials = load_trials(args)
    ik_failures = parse_log_ik_failures(args.log_dir)
    diag_successes = load_diagnostic_successes(args.diagnostics_csv)

    output_rows: list[dict[str, str]] = []
    for row in trials:
        matrix = parse_matrix(row.get("best_pose_exec_world", ""))
        if matrix is None:
            continue
        approach = parse_vector(row.get("approach_dir_world", ""))
        if approach is None:
            approach = matrix[:3, 2]
        approach = normalize(approach.astype(float))
        key = (row["_object"], row["_task"], row.get("scene_id", ""), row.get("trial_id", ""))
        diag_known = key in diag_successes

        output_rows.append(
            {
                "object": row["_object"],
                "task": row["_task"],
                "scene_id": row.get("scene_id", ""),
                "trial_id": row.get("trial_id", ""),
                "approach_category": classify_approach(approach, args.horizontal_abs_z, args.vertical_abs_z),
                "azimuth_bin": azimuth_bin(approach),
                "approach_x": f"{approach[0]:.6f}",
                "approach_y": f"{approach[1]:.6f}",
                "approach_z": f"{approach[2]:.6f}",
                "abs_approach_z": f"{abs(float(approach[2])):.6f}",
                "xy_norm": f"{float(np.linalg.norm(approach[:2])):.6f}",
                "ik_failed_log": yes_no(key in ik_failures),
                "diag_success_any": yes_no(diag_successes[key]) if diag_known else "",
                "diag_still_unreachable": yes_no((not diag_successes[key]) if diag_known else None),
                "physics_success": row.get("physics_success", ""),
                "pose_correct_manual": row.get("pose_correct_manual", ""),
                "position_correct_manual": row.get("position_correct_manual", ""),
                "direction_correct_manual": row.get("direction_correct_manual", ""),
                "grasp_score": row.get("grasp_score", ""),
                "collision_free_count": row.get("collision_free_count", ""),
                "review_image": row.get("review_image", ""),
            }
        )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "object", "task", "scene_id", "trial_id", "approach_category", "azimuth_bin",
        "approach_x", "approach_y", "approach_z", "abs_approach_z", "xy_norm",
        "ik_failed_log", "diag_success_any", "diag_still_unreachable",
        "physics_success", "pose_correct_manual", "position_correct_manual",
        "direction_correct_manual", "grasp_score", "collision_free_count", "review_image",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Loaded trials with saved poses: {len(output_rows)}")
    print(f"Log IK failure trials: {len(ik_failures)}")
    print(f"Diagnostics-covered trials: {len(diag_successes)}")
    print(f"Saved: {output_path}")
    print_group_summary("By Approach Category", output_rows, ("approach_category",))
    print_group_summary("By Approach Category And Task", output_rows, ("approach_category", "object", "task"))
    print_group_summary("By Horizontal Azimuth", [r for r in output_rows if r["approach_category"] == "horizontal"], ("azimuth_bin",))

    still_unreachable = [row for row in output_rows if row["diag_still_unreachable"] == "true"]
    if still_unreachable:
        print_group_summary("Still Unreachable After Diagnostics", still_unreachable, ("approach_category", "object", "task"))


if __name__ == "__main__":
    main()
