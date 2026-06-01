#!/usr/bin/env python3
"""Local browser-based reviewer for LODGrasp IsaacSim review images.

The tool reads deduplicated trial_results.csv files, shows each review image in
the browser, and saves manual labels to a separate manual_review.csv file.
It uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import threading
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "batch_test_results_refactored" / "four_objects"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "manual_review.csv"

REVIEW_FIELDS = [
    "id",
    "object",
    "task",
    "scene_id",
    "trial_id",
    "prompt",
    "natural_instruction",
    "fail_reason",
    "review_image",
    "position_correct_manual",
    "direction_correct_manual",
    "pose_correct_manual",
    "grasp_success_manual",
    "task_success_manual",
    "uncertain_manual",
    "note",
    "updated_at",
]

OBJECT_ORDER = {
    "brush": 0,
    "drill": 1,
    "mug": 2,
    "spoon": 3,
}

VALID_TRI_STATE = {"unknown", "yes", "no"}
VALID_BOOL_STATE = {"no", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review LODGrasp result images in a local browser.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--task", action="append", help="Only load this task name. Can be repeated.")
    parser.add_argument("--object", action="append", dest="objects", help="Only load this object. Can be repeated.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def int_suffix(text: str) -> int:
    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else 0


def normalize_tri_state(value: Any) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return "yes"
    if value in {"false", "0", "no", "n"}:
        return "no"
    if value in {"unsure", "unknown", ""}:
        return "unknown"
    return "unknown"


def normalize_bool_state(value: Any) -> str:
    if value is None:
        return "no"
    value = str(value).strip().lower()
    return "yes" if value in {"true", "1", "yes", "y"} else "no"


def derive_pose(position: str, direction: str) -> str:
    if position == "yes" and direction == "yes":
        return "yes"
    if position == "no" or direction == "no":
        return "no"
    return "unknown"


def derive_task_success(pose: str) -> str:
    return pose


def is_review_complete(review: dict[str, str]) -> bool:
    required = ("position_correct_manual", "direction_correct_manual", "grasp_success_manual")
    return all(review.get(field, "unknown") in {"yes", "no"} for field in required)


@dataclass
class ReviewItem:
    item_id: str
    object_name: str
    task: str
    scene_id: str
    trial_id: str
    prompt: str
    natural_instruction: str
    fail_reason: str
    review_image: str
    image_path: Path

    def to_public_dict(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "id": self.item_id,
            "object": self.object_name,
            "task": self.task,
            "scene_id": self.scene_id,
            "trial_id": self.trial_id,
            "prompt": self.prompt,
            "natural_instruction": self.natural_instruction,
            "fail_reason": self.fail_reason,
            "review_image": self.review_image,
            "image_exists": self.image_path.exists(),
            "image_url": f"/api/image?idx={index}",
        }


class ReviewStore:
    def __init__(
        self,
        results_root: Path,
        output_path: Path,
        task_filter: set[str] | None = None,
        object_filter: set[str] | None = None,
    ) -> None:
        self.results_root = results_root.resolve()
        self.output_path = output_path.resolve()
        self.task_filter = task_filter
        self.object_filter = object_filter
        self.lock = threading.Lock()
        self.items = self._load_items()
        self.reviews = self._load_reviews()
        self._ensure_review_rows()

    def _load_items(self) -> list[ReviewItem]:
        latest: OrderedDict[tuple[str, str, str, str], dict[str, str]] = OrderedDict()
        csv_paths = sorted(self.results_root.glob("*/*/trial_results.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No trial_results.csv found under {self.results_root}")

        for csv_path in csv_paths:
            object_name = csv_path.parts[-3]
            task_name = csv_path.parts[-2]
            if self.object_filter and object_name not in self.object_filter:
                continue
            if self.task_filter and task_name not in self.task_filter:
                continue

            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    row["_object"] = object_name
                    row["_task"] = task_name
                    key = (object_name, task_name, row.get("scene_id", ""), row.get("trial_id", ""))
                    latest[key] = row

        items: list[ReviewItem] = []
        for (object_name, task_name, scene_id, trial_id), row in latest.items():
            review_image = row.get("review_image", "")
            image_path = resolve_path(Path(review_image), SCRIPT_DIR) if review_image else Path()
            item_id = f"{object_name}|{task_name}|{scene_id}|{trial_id}"
            items.append(
                ReviewItem(
                    item_id=item_id,
                    object_name=object_name,
                    task=task_name,
                    scene_id=scene_id,
                    trial_id=str(trial_id),
                    prompt=row.get("prompt", ""),
                    natural_instruction=row.get("natural_instruction", ""),
                    fail_reason=row.get("fail_reason", ""),
                    review_image=review_image,
                    image_path=image_path,
                )
            )

        items.sort(
            key=lambda item: (
                OBJECT_ORDER.get(item.object_name, 99),
                item.object_name,
                item.task,
                int_suffix(item.scene_id),
                int(item.trial_id) if item.trial_id.isdigit() else 0,
            )
        )
        return items

    def _load_reviews(self) -> dict[str, dict[str, str]]:
        reviews: dict[str, dict[str, str]] = {}
        if not self.output_path.exists():
            return reviews

        with self.output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item_id = row.get("id", "")
                if not item_id:
                    continue
                reviews[item_id] = self._normalize_review_row(row)
        return reviews

    def _normalize_review_row(self, row: dict[str, Any]) -> dict[str, str]:
        position = normalize_tri_state(row.get("position_correct_manual"))
        direction = normalize_tri_state(row.get("direction_correct_manual"))
        grasp = normalize_tri_state(row.get("grasp_success_manual"))
        pose = derive_pose(position, direction)
        task_success = derive_task_success(pose)
        return {
            "position_correct_manual": position,
            "direction_correct_manual": direction,
            "pose_correct_manual": pose,
            "grasp_success_manual": grasp,
            "task_success_manual": task_success,
            "uncertain_manual": normalize_bool_state(row.get("uncertain_manual")),
            "note": str(row.get("note", "") or ""),
            "updated_at": str(row.get("updated_at", "") or ""),
        }

    def _default_review(self) -> dict[str, str]:
        return {
            "position_correct_manual": "unknown",
            "direction_correct_manual": "unknown",
            "pose_correct_manual": "unknown",
            "grasp_success_manual": "unknown",
            "task_success_manual": "unknown",
            "uncertain_manual": "no",
            "note": "",
            "updated_at": "",
        }

    def _ensure_review_rows(self) -> None:
        for item in self.items:
            self.reviews.setdefault(item.item_id, self._default_review())

    def payload(self) -> dict[str, Any]:
        with self.lock:
            public_items = [item.to_public_dict(index) for index, item in enumerate(self.items)]
            reviews = {item.item_id: self.reviews[item.item_id] for item in self.items}
            complete = sum(is_review_complete(review) for review in reviews.values())
            return {
                "items": public_items,
                "reviews": reviews,
                "output_csv": str(self.output_path),
                "total": len(public_items),
                "complete": complete,
            }

    def update_review(self, item_id: str, updates: dict[str, Any]) -> dict[str, str]:
        with self.lock:
            if item_id not in self.reviews:
                raise KeyError(f"Unknown item id: {item_id}")

            review = dict(self.reviews[item_id])
            for field in ("position_correct_manual", "direction_correct_manual", "grasp_success_manual"):
                if field in updates:
                    review[field] = normalize_tri_state(updates[field])
            if "uncertain_manual" in updates:
                review["uncertain_manual"] = normalize_bool_state(updates["uncertain_manual"])
            if "note" in updates:
                review["note"] = str(updates["note"] or "")

            review["pose_correct_manual"] = derive_pose(
                review["position_correct_manual"],
                review["direction_correct_manual"],
            )
            review["task_success_manual"] = derive_task_success(review["pose_correct_manual"])
            review["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self.reviews[item_id] = review
            self.save_reviews_locked()
            return review

    def save_reviews_locked(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")

        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for item in self.items:
                review = self.reviews[item.item_id]
                writer.writerow(
                    {
                        "id": item.item_id,
                        "object": item.object_name,
                        "task": item.task,
                        "scene_id": item.scene_id,
                        "trial_id": item.trial_id,
                        "prompt": item.prompt,
                        "natural_instruction": item.natural_instruction,
                        "fail_reason": item.fail_reason,
                        "review_image": item.review_image,
                        **review,
                    }
                )
        os.replace(tmp_path, self.output_path)

    def image_path(self, index: int) -> Path:
        if index < 0 or index >= len(self.items):
            raise IndexError(index)
        return self.items[index].image_path


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LODGrasp Review</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111318;
      --panel: #1b2029;
      --panel2: #232a35;
      --text: #e9edf5;
      --muted: #9ca7b8;
      --line: #364052;
      --yes: #33c481;
      --no: #f05d5e;
      --unknown: #7b8495;
      --accent: #78a8ff;
      --warn: #ffcc66;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #151922;
    }
    .title { font-weight: 700; letter-spacing: .02em; }
    .progress-wrap { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 13px; }
    .bar { width: 220px; height: 8px; border-radius: 999px; background: #303746; overflow: hidden; }
    .bar > span { display: block; height: 100%; width: 0; background: var(--accent); }
    main {
      display: grid;
      grid-template-columns: 260px minmax(520px, 1fr) 330px;
      height: calc(100vh - 48px);
      min-height: 600px;
    }
    aside, section, .right {
      min-height: 0;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      flex-direction: column;
    }
    .filters { padding: 12px; border-bottom: 1px solid var(--line); display: grid; gap: 8px; }
    select, input, textarea {
      width: 100%;
      background: #10141c;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      font: inherit;
    }
    label.check { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }
    label.check input { width: auto; }
    .list { overflow: auto; padding: 8px; display: grid; gap: 6px; }
    .row {
      border: 1px solid transparent;
      background: #151a23;
      border-radius: 6px;
      padding: 8px;
      cursor: pointer;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .row.active { border-color: var(--accent); color: var(--text); background: #1f2837; }
    .row.done { border-left: 4px solid var(--yes); }
    .row.partial { border-left: 4px solid var(--warn); }
    .row .top { display: flex; justify-content: space-between; color: var(--text); font-weight: 650; }
    .viewer {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      background: #0e1117;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #121722;
    }
    .meta-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
    }
    .meta-card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
    .meta-card .value { margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .image-box {
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 12px;
      overflow: hidden;
    }
    img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 4px;
      background: #050608;
      box-shadow: 0 12px 40px rgba(0,0,0,.35);
    }
    .right {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      overflow: auto;
    }
    .panel-title { margin: 0 0 8px; font-size: 16px; }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.55; margin-bottom: 12px; }
    .field { margin: 14px 0; }
    .field-label { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 6px; }
    .field-label b { font-size: 13px; }
    .field-label span { color: var(--muted); font-size: 12px; }
    .tri {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
    }
    button {
      border: 1px solid var(--line);
      color: var(--text);
      background: var(--panel2);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
      font: inherit;
    }
    button:hover { border-color: var(--accent); }
    button.yes.active { background: rgba(51,196,129,.22); border-color: var(--yes); }
    button.no.active { background: rgba(240,93,94,.22); border-color: var(--no); }
    button.unknown.active { background: rgba(123,132,149,.22); border-color: var(--unknown); }
    button.full { width: 100%; margin-top: 8px; }
    .derived {
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 10px 0;
      margin: 14px 0;
      display: grid;
      gap: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      background: #303746;
      color: var(--muted);
    }
    .pill.yes { color: var(--yes); background: rgba(51,196,129,.12); }
    .pill.no { color: var(--no); background: rgba(240,93,94,.12); }
    .pill.unknown { color: var(--unknown); background: rgba(123,132,149,.12); }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
    .save-line { color: var(--muted); font-size: 12px; margin-top: 8px; min-height: 18px; }
    .fail {
      color: var(--warn);
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
      background: #17120a;
      border: 1px solid #4c3a12;
      border-radius: 6px;
      padding: 8px;
      margin-bottom: 12px;
    }
  </style>
</head>
<body>
  <header>
    <div class="title">LODGrasp Manual Review</div>
    <div class="progress-wrap">
      <span id="progressText">Loading...</span>
      <div class="bar"><span id="progressBar"></span></div>
      <span id="outputCsv"></span>
    </div>
  </header>
  <main>
    <aside>
      <div class="filters">
        <select id="taskFilter"></select>
        <label class="check"><input type="checkbox" id="unreviewedOnly"> show unreviewed only</label>
        <input id="search" placeholder="search task/cam/trial">
      </div>
      <div id="list" class="list"></div>
    </aside>
    <section class="viewer">
      <div class="meta">
        <div class="meta-card"><div class="label">object</div><div class="value" id="metaObject"></div></div>
        <div class="meta-card"><div class="label">task</div><div class="value" id="metaTask"></div></div>
        <div class="meta-card"><div class="label">camera</div><div class="value" id="metaCamera"></div></div>
        <div class="meta-card"><div class="label">trial</div><div class="value" id="metaTrial"></div></div>
      </div>
      <div class="image-box"><img id="reviewImage" alt="review"></div>
    </section>
    <div class="right">
      <h2 class="panel-title">Labels</h2>
      <div class="hint">
        Keys: 1 position, 2 direction, 3 grasp, 4 uncertain, A all yes+next,
        X all no+next, Space/Right next, Left previous, U next unreviewed.
      </div>
      <div id="failReason" class="fail" hidden></div>
      <div class="field">
        <div class="field-label"><b>Position correct</b><span>key 1</span></div>
        <div class="tri" data-field="position_correct_manual"></div>
      </div>
      <div class="field">
        <div class="field-label"><b>Direction correct</b><span>key 2</span></div>
        <div class="tri" data-field="direction_correct_manual"></div>
      </div>
      <div class="field">
        <div class="field-label"><b>Grasp success</b><span>key 3</span></div>
        <div class="tri" data-field="grasp_success_manual"></div>
      </div>
      <div class="derived">
        <div>Pose correct: <span id="posePill" class="pill unknown">unknown</span></div>
        <div>Task success (pos + dir): <span id="taskPill" class="pill unknown">unknown</span></div>
        <button id="uncertainBtn" class="full">Uncertain: no</button>
      </div>
      <div class="field">
        <div class="field-label"><b>Note</b><span>auto saved</span></div>
        <textarea id="note" rows="4" placeholder="optional note"></textarea>
      </div>
      <div class="actions">
        <button id="prevBtn">Previous</button>
        <button id="nextBtn">Next</button>
        <button id="allYesBtn">All yes + next</button>
        <button id="allNoBtn">All no + next</button>
        <button id="nextTodoBtn" class="full">Next unreviewed</button>
      </div>
      <div class="save-line" id="saveLine"></div>
    </div>
  </main>
  <script>
    const state = {
      items: [],
      reviews: {},
      visible: [],
      index: 0,
      saveTimer: null,
    };

    const triValues = ["unknown", "yes", "no"];
    const fields = ["position_correct_manual", "direction_correct_manual", "grasp_success_manual"];

    function reviewOf(item) {
      return state.reviews[item.id];
    }

    function isComplete(review) {
      return fields.every((field) => ["yes", "no"].includes(review[field]));
    }

    function derivePose(review) {
      if (review.position_correct_manual === "yes" && review.direction_correct_manual === "yes") return "yes";
      if (review.position_correct_manual === "no" || review.direction_correct_manual === "no") return "no";
      return "unknown";
    }

    function deriveTask(review) {
      return derivePose(review);
    }

    function currentItem() {
      return state.visible[state.index];
    }

    function refreshProgress() {
      const complete = state.items.filter((item) => isComplete(reviewOf(item))).length;
      const total = state.items.length;
      document.getElementById("progressText").textContent = `${complete}/${total} reviewed`;
      document.getElementById("progressBar").style.width = total ? `${(complete / total) * 100}%` : "0%";
    }

    function makeTriButtons() {
      document.querySelectorAll(".tri").forEach((box) => {
        const field = box.dataset.field;
        box.innerHTML = "";
        for (const value of triValues) {
          const button = document.createElement("button");
          button.textContent = value;
          button.className = value;
          button.addEventListener("click", () => setField(field, value));
          box.appendChild(button);
        }
      });
    }

    function setPill(el, value) {
      el.textContent = value;
      el.className = `pill ${value}`;
    }

    function renderList() {
      const list = document.getElementById("list");
      list.innerHTML = "";
      state.visible.forEach((item, visibleIndex) => {
        const review = reviewOf(item);
        const row = document.createElement("div");
        row.className = "row";
        if (visibleIndex === state.index) row.classList.add("active");
        if (isComplete(review)) row.classList.add("done");
        else if (fields.some((field) => review[field] !== "unknown")) row.classList.add("partial");
        row.innerHTML = `
          <div class="top"><span>${item.object}/${item.task}</span><span>#${item.trial_id}</span></div>
          <div>${item.scene_id}</div>
          <div>pos=${review.position_correct_manual}, dir=${review.direction_correct_manual}, grasp=${review.grasp_success_manual}</div>
        `;
        row.addEventListener("click", () => {
          state.index = visibleIndex;
          render();
        });
        list.appendChild(row);
      });
    }

    function applyFilters() {
      const task = document.getElementById("taskFilter").value;
      const unreviewedOnly = document.getElementById("unreviewedOnly").checked;
      const query = document.getElementById("search").value.trim().toLowerCase();
      const oldItem = currentItem();
      state.visible = state.items.filter((item) => {
        const review = reviewOf(item);
        if (task !== "all" && item.task !== task) return false;
        if (unreviewedOnly && isComplete(review)) return false;
        if (query) {
          const hay = `${item.object} ${item.task} ${item.scene_id} ${item.trial_id} ${item.fail_reason}`.toLowerCase();
          if (!hay.includes(query)) return false;
        }
        return true;
      });
      const oldIndex = oldItem ? state.visible.findIndex((item) => item.id === oldItem.id) : -1;
      state.index = oldIndex >= 0 ? oldIndex : 0;
      render();
    }

    function render() {
      if (!state.visible.length) {
        document.getElementById("reviewImage").removeAttribute("src");
        document.getElementById("metaObject").textContent = "none";
        document.getElementById("metaTask").textContent = "none";
        document.getElementById("metaCamera").textContent = "none";
        document.getElementById("metaTrial").textContent = "none";
        renderList();
        refreshProgress();
        return;
      }
      if (state.index < 0) state.index = 0;
      if (state.index >= state.visible.length) state.index = state.visible.length - 1;

      const item = currentItem();
      const review = reviewOf(item);
      document.getElementById("metaObject").textContent = item.object;
      document.getElementById("metaTask").textContent = item.task;
      document.getElementById("metaCamera").textContent = item.scene_id;
      document.getElementById("metaTrial").textContent = `${item.trial_id} (${state.index + 1}/${state.visible.length})`;
      document.getElementById("reviewImage").src = `${item.image_url}&t=${Date.now()}`;

      const fail = document.getElementById("failReason");
      if (item.fail_reason && item.fail_reason !== "not_lifted") {
        fail.hidden = false;
        fail.textContent = item.fail_reason;
      } else {
        fail.hidden = true;
        fail.textContent = "";
      }

      for (const field of fields) {
        document.querySelectorAll(`.tri[data-field="${field}"] button`).forEach((button) => {
          button.classList.toggle("active", button.textContent === review[field]);
        });
      }
      setPill(document.getElementById("posePill"), review.pose_correct_manual || derivePose(review));
      setPill(document.getElementById("taskPill"), review.task_success_manual || deriveTask(review));
      document.getElementById("uncertainBtn").textContent = `Uncertain: ${review.uncertain_manual || "no"}`;
      document.getElementById("uncertainBtn").classList.toggle("active", review.uncertain_manual === "yes");
      document.getElementById("note").value = review.note || "";
      renderList();
      refreshProgress();
    }

    function scheduleSave() {
      if (state.saveTimer) clearTimeout(state.saveTimer);
      state.saveTimer = setTimeout(saveCurrent, 180);
    }

    async function saveCurrent() {
      const item = currentItem();
      if (!item) return;
      const review = reviewOf(item);
      document.getElementById("saveLine").textContent = "saving...";
      const response = await fetch("/api/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: item.id, ...review }),
      });
      if (!response.ok) {
        document.getElementById("saveLine").textContent = "save failed";
        return;
      }
      const payload = await response.json();
      state.reviews[item.id] = payload.review;
      document.getElementById("saveLine").textContent = `saved ${new Date().toLocaleTimeString()}`;
      refreshProgress();
      renderList();
    }

    function setField(field, value) {
      const item = currentItem();
      if (!item) return;
      const review = reviewOf(item);
      review[field] = value;
      review.pose_correct_manual = derivePose(review);
      review.task_success_manual = deriveTask(review);
      scheduleSave();
      render();
    }

    function cycleField(field) {
      const item = currentItem();
      if (!item) return;
      const review = reviewOf(item);
      const current = review[field] || "unknown";
      const next = triValues[(triValues.indexOf(current) + 1) % triValues.length];
      setField(field, next);
    }

    function setAll(value, moveNext = false) {
      const item = currentItem();
      if (!item) return;
      const review = reviewOf(item);
      review.position_correct_manual = value;
      review.direction_correct_manual = value;
      review.grasp_success_manual = value;
      review.pose_correct_manual = derivePose(review);
      review.task_success_manual = deriveTask(review);
      saveCurrent().then(() => {
        if (moveNext) goNext();
      });
      render();
    }

    function toggleUncertain() {
      const item = currentItem();
      if (!item) return;
      const review = reviewOf(item);
      review.uncertain_manual = review.uncertain_manual === "yes" ? "no" : "yes";
      scheduleSave();
      render();
    }

    function goNext() {
      if (!state.visible.length) return;
      state.index = Math.min(state.index + 1, state.visible.length - 1);
      render();
    }

    function goPrev() {
      if (!state.visible.length) return;
      state.index = Math.max(state.index - 1, 0);
      render();
    }

    function nextUnreviewed() {
      if (!state.visible.length) return;
      for (let offset = 1; offset <= state.visible.length; offset++) {
        const idx = (state.index + offset) % state.visible.length;
        if (!isComplete(reviewOf(state.visible[idx]))) {
          state.index = idx;
          render();
          return;
        }
      }
    }

    async function init() {
      makeTriButtons();
      const response = await fetch("/api/items");
      const payload = await response.json();
      state.items = payload.items;
      state.reviews = payload.reviews;
      state.visible = [...state.items];
      document.getElementById("outputCsv").textContent = payload.output_csv;

      const taskSelect = document.getElementById("taskFilter");
      const tasks = [...new Set(state.items.map((item) => item.task))];
      taskSelect.innerHTML = '<option value="all">all tasks</option>' + tasks.map((task) => `<option value="${task}">${task}</option>`).join("");
      taskSelect.addEventListener("change", applyFilters);
      document.getElementById("unreviewedOnly").addEventListener("change", applyFilters);
      document.getElementById("search").addEventListener("input", applyFilters);
      document.getElementById("note").addEventListener("input", (event) => {
        const item = currentItem();
        if (!item) return;
        reviewOf(item).note = event.target.value;
        scheduleSave();
      });
      document.getElementById("uncertainBtn").addEventListener("click", toggleUncertain);
      document.getElementById("nextBtn").addEventListener("click", goNext);
      document.getElementById("prevBtn").addEventListener("click", goPrev);
      document.getElementById("allYesBtn").addEventListener("click", () => setAll("yes", true));
      document.getElementById("allNoBtn").addEventListener("click", () => setAll("no", true));
      document.getElementById("nextTodoBtn").addEventListener("click", nextUnreviewed);

      document.addEventListener("keydown", (event) => {
        const tag = (event.target.tagName || "").toLowerCase();
        if (tag === "textarea" || tag === "input" || tag === "select") return;
        if (event.key === "1") cycleField("position_correct_manual");
        else if (event.key === "2") cycleField("direction_correct_manual");
        else if (event.key === "3") cycleField("grasp_success_manual");
        else if (event.key === "4") toggleUncertain();
        else if (event.key === "ArrowRight" || event.key === " " || event.key.toLowerCase() === "n") goNext();
        else if (event.key === "ArrowLeft" || event.key.toLowerCase() === "b") goPrev();
        else if (event.key.toLowerCase() === "u") nextUnreviewed();
        else if (event.key.toLowerCase() === "a") setAll("yes", true);
        else if (event.key.toLowerCase() === "x") setAll("no", true);
      });

      render();
    }
    init();
  </script>
</body>
</html>
"""


def make_handler(store: ReviewStore) -> type[BaseHTTPRequestHandler]:
    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_bytes(body, "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/items":
                self.send_json(store.payload())
                return
            if parsed.path == "/api/image":
                query = parse_qs(parsed.query)
                try:
                    index = int(query.get("idx", ["-1"])[0])
                    image_path = store.image_path(index)
                except (ValueError, IndexError):
                    self.send_json({"error": "invalid image index"}, status=404)
                    return
                if not image_path.exists():
                    self.send_json({"error": f"missing image: {image_path}"}, status=404)
                    return
                content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                self.send_bytes(image_path.read_bytes(), content_type)
                return
            if parsed.path == "/api/manual_review.csv":
                if store.output_path.exists():
                    self.send_bytes(store.output_path.read_bytes(), "text/csv; charset=utf-8")
                else:
                    self.send_json({"error": "manual_review.csv does not exist"}, status=404)
                return
            self.send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/review":
                self.send_json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                item_id = payload.pop("id")
                review = store.update_review(item_id, payload)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
                return
            self.send_json({"review": review})

    return ReviewHandler


def main() -> None:
    args = parse_args()
    results_root = resolve_path(args.results_root, SCRIPT_DIR)
    output_path = resolve_path(args.output, SCRIPT_DIR)
    task_filter = set(args.task) if args.task else None
    object_filter = set(args.objects) if args.objects else None

    store = ReviewStore(
        results_root=results_root,
        output_path=output_path,
        task_filter=task_filter,
        object_filter=object_filter,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}/"

    print(f"Loaded {len(store.items)} review items")
    print(f"Writing labels to: {store.output_path}")
    print(f"Open: {url}")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Reviewer stopped")


if __name__ == "__main__":
    main()
