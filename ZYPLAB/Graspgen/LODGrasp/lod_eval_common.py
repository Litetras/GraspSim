import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass(frozen=True)
class LodEvalTask:
    task_name: str
    natural_instruction: str


@dataclass(frozen=True)
class LodEvalConfig:
    prompt: str
    tasks: Sequence[LodEvalTask]
    scene_pattern: str
    scene_id_pattern: str
    grasp_threshold: float

    scene_dir: str = "/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib"
    img_dir: str = "batch_test_results_refactored"
    baseline_name: str = "LODGrasp"
    gripper_config: str = "/home/zyp/Desktop/zyp_dataset7_clip/tutorial/models/tutorial_model_config.yaml"
    sam3_checkpoint: str = "/home/zyp/sam3/zypmodel/sam3/sam3.pt"
    graspgen_repo: str = "/home/zyp/GraspGen"

    headless: bool = True
    enable_meshcat: bool = False
    render_motion_steps: bool = False
    save_step_images: bool = True
    save_review_image: bool = True
    split_output_by_task: bool = False
    skip_existing: bool = False

    cam_ids: Sequence[int] = field(default_factory=lambda: tuple(range(1, 8)))
    trials_per_task: int = 14
    trial_ids: Optional[Sequence[int]] = None
    camera_width: int = 1280
    camera_height: int = 720
    axis_len: float = 0.15
    axis_thick: float = 0.005
    num_grasps: int = 200
    unload_sam3_after_mask: bool = False
    sam3_subprocess: bool = False
    lift_success_height_delta: float = 0.03
    settle_steps: int = 100
    pregrasp_steps: int = 180
    insert_steps: int = 80
    gripper_close_steps: int = 80
    lift_move_steps: int = 120
    lift_settle_steps: int = 120

    object_prim_path: Optional[str] = None
    object_prim_exclude_keywords: Sequence[str] = ("holder", "camera", "marker", "franka")


RESULT_FIELDS = [
    "task", "baseline", "scene_id", "trial_id", "prompt", "natural_instruction",
    "success", "physics_success", "grasp_score", "collision_free_count",
    "best_pose_raw_camera", "best_pose_exec_world", "approach_dir_world",
    "close_pos_world", "ik_strategy", "ik_fail_stage",
    "object_prim_path", "object_height_before",
    "object_height_after", "object_displacement", "image_before",
    "image_grasped", "image_final", "review_image", "fail_reason",
    "position_correct_manual", "direction_correct_manual", "pose_correct_manual",
]


def get_task_img_dir(config: LodEvalConfig, task: LodEvalTask) -> str:
    if config.split_output_by_task:
        return os.path.join(config.img_dir, task.task_name)
    return config.img_dir


def get_review_image_path(review_dir: str, trial: int, scene_id: str, task: LodEvalTask) -> str:
    return os.path.join(review_dir, f"trial_{trial:03d}_{scene_id}_{task.task_name}_review.jpg")


def should_skip_trial(config: LodEvalConfig, review_dir: str, trial: int, scene_id: str, task: LodEvalTask) -> bool:
    if not config.skip_existing:
        return False
    if not config.save_review_image:
        return False
    return os.path.exists(get_review_image_path(review_dir, trial, scene_id, task))


def get_cv2():
    import cv2

    return cv2


def get_T(translation, rotation_matrix):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


def move_along_grasp_dir(htm: np.ndarray, distance: float = 0.1) -> np.ndarray:
    grasp_dir = htm[:3, 2]
    grasp_dir_unit = grasp_dir / np.linalg.norm(grasp_dir)
    new_htm = np.eye(4)
    new_htm[:3, :3] = htm[:3, :3]
    new_htm[:3, 3] = htm[:3, 3] + grasp_dir_unit * distance
    return new_htm


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def make_base_record(config: LodEvalConfig, task: LodEvalTask, cam_id: int, trial: int, fail_reason=None):
    return {
        "task": task.task_name,
        "baseline": config.baseline_name,
        "scene_id": config.scene_id_pattern.format(cam_id=cam_id),
        "trial_id": trial,
        "prompt": config.prompt,
        "natural_instruction": task.natural_instruction,
        "success": None,
        "physics_success": None,
        "grasp_score": None,
        "collision_free_count": None,
        "best_pose_raw_camera": None,
        "best_pose_exec_world": None,
        "approach_dir_world": None,
        "close_pos_world": None,
        "ik_strategy": None,
        "ik_fail_stage": None,
        "object_prim_path": None,
        "object_height_before": None,
        "object_height_after": None,
        "object_displacement": None,
        "image_before": None,
        "image_grasped": None,
        "image_final": None,
        "review_image": None,
        "fail_reason": fail_reason,
        "position_correct_manual": None,
        "direction_correct_manual": None,
        "pose_correct_manual": None,
    }


def ensure_csv_schema(result_csv: str):
    if not os.path.exists(result_csv):
        return

    with open(result_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if fieldnames == RESULT_FIELDS:
        return

    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def write_trial_record(config: LodEvalConfig, record, img_dir=None):
    img_dir = img_dir or config.img_dir
    os.makedirs(img_dir, exist_ok=True)
    result_jsonl = os.path.join(img_dir, "trial_results.jsonl")
    result_csv = os.path.join(img_dir, "trial_results.csv")
    serializable = {field: to_jsonable(record.get(field)) for field in RESULT_FIELDS}

    with open(result_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    ensure_csv_schema(result_csv)
    csv_exists = os.path.exists(result_csv)
    csv_row = {}
    for key, value in serializable.items():
        if isinstance(value, (list, dict)):
            csv_row[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            csv_row[key] = ""
        else:
            csv_row[key] = value

    with open(result_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(csv_row)


def find_object_prim_path(config: LodEvalConfig, omni_usd, UsdGeom):
    if config.object_prim_path:
        return config.object_prim_path

    stage = omni_usd.get_context().get_stage()
    prompt_lower = config.prompt.lower()
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        text = path.lower()
        if prompt_lower not in text:
            continue
        if any(keyword in text for keyword in config.object_prim_exclude_keywords):
            continue
        if UsdGeom.Xformable(prim):
            candidates.append(path)

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.count("/"), len(item)))
    return candidates[0]


def get_prim_world_position(prim_path, omni_usd, Usd, UsdGeom):
    if not prim_path:
        return None
    stage = omni_usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    try:
        world_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_matrix.ExtractTranslation()
        return np.array([translation[0], translation[1], translation[2]], dtype=float)
    except Exception:
        return None


def capture_prim_pose(prim_path, SingleXFormPrim):
    if not prim_path:
        return None
    try:
        position, orientation = SingleXFormPrim(prim_path).get_world_pose()
        return np.array(position, dtype=float), np.array(orientation, dtype=float)
    except Exception:
        return None


def restore_prim_pose(prim_path, pose, SingleXFormPrim):
    if not prim_path or pose is None:
        return
    try:
        position, orientation = pose
        SingleXFormPrim(prim_path).set_world_pose(position=position, orientation=orientation)
    except Exception as exc:
        print(f"⚠️ 物体 pose 复位失败: {exc}")


def save_cam_img(camera_obj, save_path, world_obj=None):
    cv2 = get_cv2()
    img_rgb = capture_cam_rgb(camera_obj, world_obj)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)
    print(f"📸 已保存截图: {save_path}")
    return save_path


def capture_cam_rgb(camera_obj, world_obj=None):
    if world_obj is not None:
        world_obj.step(render=True)
    return camera_obj.get_rgb()[:, :, :3].astype(np.uint8)


def _read_review_panel(label, image_source, target_height=360):
    cv2 = get_cv2()
    if isinstance(image_source, np.ndarray):
        panel = cv2.cvtColor(image_source.astype(np.uint8), cv2.COLOR_RGB2BGR)
    elif image_source and os.path.exists(image_source):
        panel = cv2.imread(image_source)
    else:
        panel = np.zeros((target_height, int(target_height * 16 / 9), 3), dtype=np.uint8)

    scale = target_height / panel.shape[0]
    width = int(panel.shape[1] * scale)
    panel = cv2.resize(panel, (width, target_height), interpolation=cv2.INTER_AREA)
    panel = cv2.copyMakeBorder(panel, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(panel, label, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return panel


def save_review_image(image_paths, save_path, record):
    cv2 = get_cv2()
    labels = ["step0_pregrasp", "step1_grasped", "step2_final"]
    panels = [_read_review_panel(label, path) for label, path in zip(labels, image_paths)]
    review = cv2.hconcat(panels)

    header = np.zeros((96, review.shape[1], 3), dtype=np.uint8)
    score = record.get("grasp_score")
    score_text = "None" if score is None else f"{float(score):.3f}"
    physics_success = record.get("physics_success")
    approach_dir = record.get("approach_dir_world")
    if isinstance(approach_dir, np.ndarray):
        approach_dir = np.array2string(approach_dir, precision=3, suppress_small=True)

    line1 = (
        f"{record['baseline']} | {record['task']} | {record['scene_id']} "
        f"trial={record['trial_id']} | score={score_text} | physics_success={physics_success}"
    )
    line2 = (
        "manual labels: position_correct / direction_correct / pose_correct "
        f"| approach_dir={approach_dir}"
    )
    cv2.putText(header, line1, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(header, line2, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 210, 210), 1, cv2.LINE_AA)
    review = cv2.vconcat([header, review])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, review)
    print(f"🧾 已保存审查图: {save_path}")
    return save_path


def run_lod_eval(config: LodEvalConfig):
    # IsaacSim must be initialized before importing omni/isaac modules.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": config.headless})
    trial_ids = tuple(config.trial_ids or range(1, config.trials_per_task + 1))
    total_trials = len(config.cam_ids) * len(config.tasks) * len(trial_ids)

    try:
        import gc
        import subprocess
        import sys
        import tempfile
        import time

        import torch
        from PIL import Image
        from scipy.ndimage import zoom

        from omni.isaac.core.utils.stage import open_stage
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
        from omni.isaac.core.utils.types import ArticulationAction
        from omni.isaac.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
        from omni.isaac.sensor import Camera
        from omni.isaac.core.objects import VisualCuboid
        import omni.usd as omni_usd
        from pxr import Usd, UsdGeom

        lod_dir = str(Path(__file__).resolve().parent)
        for extra_path in (config.graspgen_repo, lod_dir):
            if extra_path not in sys.path:
                sys.path.append(extra_path)

        if config.sam3_subprocess:
            build_sam3_image_model = None
            Sam3Processor = None
        else:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        from demogen_LOD import demo_variable
        from grasp_gen.grasp_server_LOD import GraspGenSampler, load_grasp_cfg

        os.makedirs(config.img_dir, exist_ok=True)

        sam3_model = None
        sam3_processor = None

        def get_sam3_processor():
            nonlocal sam3_model, sam3_processor
            if sam3_model is None or sam3_processor is None:
                print("🚀 正在加载 SAM3 模型...")
                sam3_model = build_sam3_image_model(checkpoint_path=config.sam3_checkpoint)
                sam3_processor = Sam3Processor(sam3_model)
                sam3_model.to("cpu")
            return sam3_model, sam3_processor

        def unload_sam3_if_needed():
            nonlocal sam3_model, sam3_processor
            if not config.unload_sam3_after_mask:
                return
            if sam3_model is not None or sam3_processor is not None:
                print("🧹 正在卸载 SAM3，降低 GraspGen/Qwen 加载时的峰值内存...")
            del sam3_model
            del sam3_processor
            sam3_model = None
            sam3_processor = None
            gc.collect()
            torch.cuda.empty_cache()

        if not config.sam3_subprocess and not config.unload_sam3_after_mask:
            get_sam3_processor()

        def run_sam3_subprocess(rgb_array):
            with tempfile.TemporaryDirectory(prefix="lod_sam3_") as tmp_dir:
                rgb_path = os.path.join(tmp_dir, "rgb.npy")
                output_path = os.path.join(tmp_dir, "mask_scores.npz")
                np.save(rgb_path, rgb_array.astype(np.uint8))
                cmd = [
                    sys.executable,
                    os.path.join(lod_dir, "sam3_segment_subprocess.py"),
                    "--rgb-npy",
                    rgb_path,
                    "--prompt",
                    config.prompt,
                    "--checkpoint",
                    config.sam3_checkpoint,
                    "--output-npz",
                    output_path,
                ]
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.run(cmd, check=True, cwd=lod_dir, env=env)
                output = np.load(output_path)
                masks = output["masks"]
                scores = output["scores"]
                output.close()
                return masks, scores

        grasp_cfg = load_grasp_cfg(config.gripper_config)
        grasp_sampler = None

        def get_grasp_sampler():
            nonlocal grasp_sampler
            if grasp_sampler is None:
                print("🚀 正在首次加载 GraspGen/Qwen 模型，后续 trial 将复用该 sampler...")
                grasp_sampler = GraspGenSampler(grasp_cfg)
                print("✅ GraspGen/Qwen sampler 已加载完成。")
            return grasp_sampler

        for cam_id in config.cam_ids:
            scene_name = config.scene_pattern.format(cam_id=cam_id)
            scene_id = config.scene_id_pattern.format(cam_id=cam_id)
            usd_path = os.path.join(config.scene_dir, scene_name)
            print("\n" + "=" * 60)
            print(f"🌍 [阶段 1] 正在加载全新场景: {usd_path}")
            print("=" * 60)

            if World.instance() is not None:
                World.instance().clear_instance()

            open_stage(usd_path)
            world = World()

            franka = world.scene.add(Franka(prim_path="/Franka", name="franka"))
            camera_path = "/World/Camera"
            camera = Camera(prim_path=camera_path, resolution=(config.camera_width, config.camera_height))
            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()
            camera.add_rgb_to_frame()

            marker_x = world.scene.add(VisualCuboid(prim_path="/World/marker_x", name="marker_x", scale=np.array([config.axis_len, config.axis_thick, config.axis_thick]), color=np.array([1., 0., 0.])))
            marker_y = world.scene.add(VisualCuboid(prim_path="/World/marker_y", name="marker_y", scale=np.array([config.axis_thick, config.axis_len, config.axis_thick]), color=np.array([0., 1., 0.])))
            marker_z = world.scene.add(VisualCuboid(prim_path="/World/marker_z", name="marker_z", scale=np.array([config.axis_thick, config.axis_thick, config.axis_len]), color=np.array([0., 0., 1.])))
            marker_x.set_visibility(False)
            marker_y.set_visibility(False)
            marker_z.set_visibility(False)

            object_prim_path = find_object_prim_path(config, omni_usd, UsdGeom)
            if object_prim_path is None:
                print(f"⚠️ 未自动匹配到 '{config.prompt}' 对应的物体 prim，物理成功率字段会留空。")
            else:
                print(f"🎯 当前评估物体 prim: {object_prim_path}")

            world.reset()
            for _ in range(config.settle_steps):
                world.step()
            object_initial_pose = capture_prim_pose(object_prim_path, SingleXFormPrim)

            ik_solver = KinematicsSolver(robot_articulation=franka)

            def move_to_pose(target_pos, target_quat, step_count=150):
                action, success = ik_solver.compute_inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=target_quat,
                )
                if not success:
                    print("❌ IK 求解失败，跳过本次移动。")
                    return False

                current_joints = franka.get_joint_positions()
                target_joints = np.copy(current_joints)
                target_joints[:7] = action.joint_positions
                for i in range(step_count):
                    alpha = i / step_count
                    interp_joints = current_joints * (1 - alpha) + target_joints * alpha
                    franka.apply_action(ArticulationAction(joint_positions=interp_joints))
                    world.step(render=config.render_motion_steps)
                world.step(render=config.render_motion_steps)
                return True

            def probe_ik_pose(target_pos, target_quat):
                action, success = ik_solver.compute_inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=target_quat,
                )
                if not success:
                    return False
                target_joints = np.copy(franka.get_joint_positions())
                target_joints[:7] = action.joint_positions
                franka.set_joint_positions(target_joints)
                world.step(render=False)
                return True

            def reset_trial_state(initial_joints):
                franka.set_joint_positions(initial_joints)
                franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
                restore_prim_pose(object_prim_path, object_initial_pose, SingleXFormPrim)
                world.step(render=False)

            def choose_ik_strategy(grasp_pos, grasp_dir, grasp_quat, initial_joints):
                # 中文说明：
                # 主算法输出的抓取位姿保持不变，这里只在机器人执行层做 IK fallback。
                # 这样不会改变“位置是否正确、方向是否正确”的算法评测，只是尽量避免
                # Franka 因固定 10cm 预抓取距离或沿抓取轴后退而产生不必要的 IK 失败。
                # 每个候选策略会先做一次不开夹爪的 IK 预演；预演后立刻恢复本轮 trial
                # 的机器人关节和物体位姿，再真正执行第一个完整可行的策略。
                strategies = [
                    {
                        "name": "original_pre010_retreat",
                        "pre_distance": 0.10,
                        "lift_mode": "approach_retreat",
                    },
                    {
                        "name": "pre007_worldz",
                        "pre_distance": 0.07,
                        "lift_mode": "world_z",
                    },
                    {
                        "name": "pre005_worldz",
                        "pre_distance": 0.05,
                        "lift_mode": "world_z",
                    },
                    {
                        "name": "pre003_worldz",
                        "pre_distance": 0.03,
                        "lift_mode": "world_z",
                    },
                    {
                        "name": "direct_close_worldz",
                        "pre_distance": None,
                        "lift_mode": "world_z",
                    },
                ]
                close_pos = grasp_pos + grasp_dir * 0.03
                last_fail_stage = None

                for strategy in strategies:
                    reset_trial_state(initial_joints)
                    pre_distance = strategy["pre_distance"]
                    pre_pos = None if pre_distance is None else grasp_pos - grasp_dir * pre_distance
                    if strategy["lift_mode"] == "world_z":
                        lift_pos = close_pos + np.array([0.0, 0.0, 0.08])
                    else:
                        lift_pos = grasp_pos - grasp_dir * 0.08

                    if pre_pos is not None and not probe_ik_pose(pre_pos, grasp_quat):
                        last_fail_stage = "pre"
                        continue
                    if not probe_ik_pose(close_pos, grasp_quat):
                        last_fail_stage = "close"
                        continue
                    if not probe_ik_pose(lift_pos, grasp_quat):
                        last_fail_stage = "lift"
                        continue

                    reset_trial_state(initial_joints)
                    return {
                        **strategy,
                        "pre_pos": pre_pos,
                        "close_pos": close_pos,
                        "lift_pos": lift_pos,
                    }, None

                reset_trial_state(initial_joints)
                return None, last_fail_stage

            for task in config.tasks:
                task_img_dir = get_task_img_dir(config, task)
                review_dir = os.path.join(task_img_dir, "review")
                os.makedirs(review_dir, exist_ok=True)

                for trial in trial_ids:
                    print("\n" + "-" * 50)
                    print(f"🎥 场景: {scene_id} | 🧾 任务: {task.task_name} | 🔄 测试轮次: {trial} / {config.trials_per_task}")
                    print("-" * 50)

                    if should_skip_trial(config, review_dir, trial, scene_id, task):
                        print(f"⏭️ 已存在 review 图，跳过: {get_review_image_path(review_dir, trial, scene_id, task)}")
                        continue

                    world.reset()
                    restore_prim_pose(object_prim_path, object_initial_pose, SingleXFormPrim)
                    franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
                    for _ in range(config.settle_steps):
                        world.step()
                    object_pos_before = get_prim_world_position(object_prim_path, omni_usd, Usd, UsdGeom)

                    print(f"[{time.strftime('%H:%M:%S')}] 正在进行 SAM3 图像分割...")
                    world.step(render=True)
                    rgb_data = camera.get_rgb()
                    depth_data = camera.get_depth()
                    rgb_image = None

                    if config.sam3_subprocess:
                        masks, scores = run_sam3_subprocess(rgb_data)
                    else:
                        rgb_image = Image.fromarray(rgb_data.astype(np.uint8))
                        sam3_model, sam3_processor = get_sam3_processor()
                        sam3_model.to("cuda")
                        inference_state_obj = sam3_processor.set_image(rgb_image)
                        output_obj = sam3_processor.set_text_prompt(state=inference_state_obj, prompt=config.prompt)

                        masks = output_obj["masks"].cpu().numpy()
                        scores = output_obj["scores"].cpu().numpy()

                        del inference_state_obj
                        del output_obj
                        sam3_model.to("cpu")
                        gc.collect()
                        torch.cuda.empty_cache()
                        unload_sam3_if_needed()

                    if len(masks) == 0:
                        print(f"⚠️ SAM3未检测到'{config.prompt}'，跳过该 trial。")
                        failure_image = None
                        if config.save_step_images:
                            failure_image = save_cam_img(
                                camera,
                                os.path.join(task_img_dir, f"trial_{trial:03d}_{scene_id}_{task.task_name}_sam3_no_mask.png"),
                                world,
                            )
                        record = make_base_record(config, task, cam_id, trial, fail_reason="sam3_no_mask")
                        record["object_prim_path"] = object_prim_path
                        record["image_before"] = failure_image
                        if object_pos_before is not None:
                            record["object_height_before"] = float(object_pos_before[2])
                        if config.save_review_image:
                            review_path = get_review_image_path(review_dir, trial, scene_id, task)
                            record["review_image"] = save_review_image([failure_image or rgb_data, None, None], review_path, record)
                        write_trial_record(config, record, task_img_dir)
                        del rgb_data, depth_data, rgb_image, masks, scores
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    best_idx = np.argmax(scores)
                    best_mask = masks[best_idx]

                    if len(best_mask.shape) == 3:
                        best_mask = best_mask[0]
                    if best_mask.shape != rgb_data.shape[:2]:
                        scale_y = rgb_data.shape[0] / best_mask.shape[0]
                        scale_x = rgb_data.shape[1] / best_mask.shape[1]
                        best_mask = zoom(best_mask, (scale_y, scale_x), order=0) > 0.5

                    final_mask = (best_mask > 0.5).astype(np.uint8)

                    intrinsic_matrix = camera.get_intrinsics_matrix()
                    intrinsic = [
                        float(intrinsic_matrix[0, 0]),
                        float(intrinsic_matrix[1, 1]),
                        float(intrinsic_matrix[0, 2]),
                        float(intrinsic_matrix[1, 2]),
                    ]

                    print(f"[{time.strftime('%H:%M:%S')}] 🧠 运行大模型推理...")
                    try:
                        grasp = demo_variable(
                            rgb_data=rgb_data,
                            depth_data=depth_data,
                            mask=final_mask,
                            intrinsic=intrinsic,
                            natural_text=[task.natural_instruction],
                            strict_text=["nnn"],
                            gripper_config=config.gripper_config,
                            grasp_sampler=get_grasp_sampler(),
                            grasp_threshold=config.grasp_threshold,
                            num_grasps=config.num_grasps,
                            visualize=config.enable_meshcat,
                        )
                    except Exception as e:
                        print(f"⚠️ 推理失败 ({e})，跳过该 trial。")
                        record = make_base_record(config, task, cam_id, trial, fail_reason=f"inference_failed: {e}")
                        record["object_prim_path"] = object_prim_path
                        if object_pos_before is not None:
                            record["object_height_before"] = float(object_pos_before[2])
                        if config.save_review_image:
                            review_path = get_review_image_path(review_dir, trial, scene_id, task)
                            record["review_image"] = save_review_image([rgb_data, None, None], review_path, record)
                        write_trial_record(config, record, task_img_dir)
                        del rgb_data, depth_data, rgb_image, masks, scores, best_mask, final_mask
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    cam_trans, cam_quat = SingleXFormPrim(camera_path).get_world_pose()
                    T_world_cam = get_T(cam_trans, quat_to_rot_matrix(cam_quat))

                    T_world_grasp = (
                        T_world_cam
                        @ get_T([0, 0, 0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                        @ grasp.pose
                        @ get_T([0, 0, 0], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                    )
                    T_world_grasp = move_along_grasp_dir(T_world_grasp, distance=0.1)

                    grasp_pos = T_world_grasp[:3, 3]
                    grasp_quat = rot_matrix_to_quat(T_world_grasp[:3, :3])
                    grasp_dir = T_world_grasp[:3, 2]
                    close_pos = grasp_pos + grasp_dir * 0.03

                    R_mat = T_world_grasp[:3, :3]
                    axes_data = [
                        (marker_x, R_mat[:, 0]),
                        (marker_y, R_mat[:, 1]),
                        (marker_z, R_mat[:, 2]),
                    ]
                    for marker, direction in axes_data:
                        center = grasp_pos + direction * (config.axis_len / 2.0)
                        marker.set_world_pose(position=center, orientation=grasp_quat)
                        marker.set_visibility(True)
                    world.step(render=config.render_motion_steps)
                    print("🔍 坐标轴 Marker 已更新显示 (蓝色Z轴指向插入方向)")

                    trial_initial_joints = np.copy(franka.get_joint_positions())
                    ik_strategy, ik_fail_stage = choose_ik_strategy(
                        grasp_pos,
                        grasp_dir,
                        grasp_quat,
                        trial_initial_joints,
                    )
                    if ik_strategy is None:
                        print(f"❌ 所有 IK fallback 策略都失败，失败阶段: {ik_fail_stage}")
                        record = make_base_record(config, task, cam_id, trial, fail_reason=f"ik_failed:{ik_fail_stage}")
                        record.update({
                            "grasp_score": float(grasp.score),
                            "collision_free_count": (
                                int(len(grasp.all_collision_free_grasps))
                                if grasp.all_collision_free_grasps is not None else None
                            ),
                            "best_pose_raw_camera": grasp.pose,
                            "best_pose_exec_world": T_world_grasp,
                            "approach_dir_world": grasp_dir,
                            "close_pos_world": close_pos,
                            "ik_fail_stage": ik_fail_stage,
                            "object_prim_path": object_prim_path,
                        })
                        if object_pos_before is not None:
                            record["object_height_before"] = float(object_pos_before[2])
                        if config.save_review_image:
                            review_path = get_review_image_path(review_dir, trial, scene_id, task)
                            record["review_image"] = save_review_image([rgb_data, None, None], review_path, record)
                        write_trial_record(config, record, task_img_dir)
                        marker_x.set_visibility(False)
                        marker_y.set_visibility(False)
                        marker_z.set_visibility(False)
                        del rgb_data, depth_data, rgb_image, masks, scores, best_mask, final_mask, grasp
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue

                    close_pos = ik_strategy["close_pos"]
                    print(f"✅ 使用 IK fallback 策略: {ik_strategy['name']}")

                    print(">>> 步骤 0: 移动到预抓取点...")
                    if ik_strategy["pre_pos"] is not None:
                        move_to_pose(ik_strategy["pre_pos"], grasp_quat, step_count=config.pregrasp_steps)
                    else:
                        print(">>> 当前策略为 direct close，跳过预抓取点。")
                    image_before = None
                    image_grasped = None
                    image_final = None
                    review_before = None
                    review_grasped = None
                    review_final = None
                    if config.save_step_images:
                        image_before = save_cam_img(
                            camera,
                            os.path.join(task_img_dir, f"trial_{trial:03d}_{scene_id}_{task.task_name}_step0_before.png"),
                            world,
                        )
                    elif config.save_review_image:
                        review_before = capture_cam_rgb(camera, world)

                    print(">>> 步骤 1: 插入并闭合夹爪...")
                    move_to_pose(close_pos, grasp_quat, step_count=config.insert_steps)
                    franka.gripper.apply_action(ArticulationAction(joint_positions=franka.gripper.joint_closed_positions))
                    for _ in range(config.gripper_close_steps):
                        world.step(render=config.render_motion_steps)
                    if config.save_step_images:
                        image_grasped = save_cam_img(
                            camera,
                            os.path.join(task_img_dir, f"trial_{trial:03d}_{scene_id}_{task.task_name}_step1_grasped.png"),
                            world,
                        )
                    elif config.save_review_image:
                        review_grasped = capture_cam_rgb(camera, world)

                    print(">>> 步骤 2: 提起物体...")
                    move_to_pose(ik_strategy["lift_pos"], grasp_quat, step_count=config.lift_move_steps)
                    for _ in range(config.lift_settle_steps):
                        world.step(render=config.render_motion_steps)
                    if config.save_step_images:
                        image_final = save_cam_img(
                            camera,
                            os.path.join(task_img_dir, f"trial_{trial:03d}_{scene_id}_{task.task_name}_step2_final.png"),
                            world,
                        )
                    elif config.save_review_image:
                        review_final = capture_cam_rgb(camera, world)

                    object_pos_after = get_prim_world_position(object_prim_path, omni_usd, Usd, UsdGeom)
                    object_height_before = float(object_pos_before[2]) if object_pos_before is not None else None
                    object_height_after = float(object_pos_after[2]) if object_pos_after is not None else None
                    object_displacement = (
                        float(np.linalg.norm(object_pos_after - object_pos_before))
                        if object_pos_before is not None and object_pos_after is not None else None
                    )
                    physics_success = (
                        bool(object_height_after - object_height_before >= config.lift_success_height_delta)
                        if object_height_before is not None and object_height_after is not None else None
                    )

                    record = make_base_record(config, task, cam_id, trial)
                    record.update({
                        "success": physics_success,
                        "physics_success": physics_success,
                        "grasp_score": float(grasp.score),
                        "collision_free_count": (
                            int(len(grasp.all_collision_free_grasps))
                            if grasp.all_collision_free_grasps is not None else None
                        ),
                        "best_pose_raw_camera": grasp.pose,
                        "best_pose_exec_world": T_world_grasp,
                        "approach_dir_world": grasp_dir,
                        "close_pos_world": close_pos,
                        "ik_strategy": ik_strategy["name"],
                        "ik_fail_stage": None,
                        "object_prim_path": object_prim_path,
                        "object_height_before": object_height_before,
                        "object_height_after": object_height_after,
                        "object_displacement": object_displacement,
                        "image_before": image_before,
                        "image_grasped": image_grasped,
                        "image_final": image_final,
                        "fail_reason": None if physics_success is not False else "not_lifted",
                    })
                    review_sources = None
                    if config.save_review_image:
                        review_path = get_review_image_path(review_dir, trial, scene_id, task)
                        review_sources = [
                            image_before or review_before,
                            image_grasped or review_grasped,
                            image_final or review_final,
                        ]
                        record["review_image"] = save_review_image(review_sources, review_path, record)
                    write_trial_record(config, record, task_img_dir)

                    marker_x.set_visibility(False)
                    marker_y.set_visibility(False)
                    marker_z.set_visibility(False)

                    del (
                        rgb_data, depth_data, rgb_image, masks, scores, best_mask, final_mask,
                        grasp, review_before, review_grasped, review_final, review_sources,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()

            print(f"✅ 场景 {scene_id} 完成。")

            # Release references before loading the next stage.
            del world

            if World.instance() is not None:
                World.instance().clear_instance()

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"📊 计划总测试数: {total_trials}")

    finally:
        simulation_app.close()
        print(f"🎉 所有 {total_trials} 次测试执行完毕！")
