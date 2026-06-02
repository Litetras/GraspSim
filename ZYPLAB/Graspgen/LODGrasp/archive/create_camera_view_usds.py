"""
Create per-view USD files by copying an object-specific cam1 template scene and
transplanting calibrated camera poses.

Typical use:
    ./isaaclab.sh -p ZYPLAB/Graspgen/LODGrasp/archive/create_camera_view_usds.py --object mug --overwrite

The script reads camera xform ops from cam1_r.usd ... cam7_r.usd and writes them
into the selected object's output USD files. Run with IsaacSim's Python
environment so that `pxr` is available.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


DEFAULT_SCENE_DIR = Path("/home/zyp/SO-ARM100/Simulation/SO101/so101_new_calib")
DEFAULT_CAMERA_PRIM = "/World/Camera"
SUPPORTED_OBJECTS = ("knife", "hammer", "brush", "drill", "mug", "spoon")

# File naming follows the evaluation script. Knife uses the original calibrated
# camN_r.usd scenes; the other objects use object_camN.usd files.
OBJECT_USD_PATTERNS = {
    "knife": {
        "template": "cam1_r.usd",
        "output_pattern": "cam{cam_id}_r.usd",
    },
    "hammer": {
        "template": "hammer_cam1.usd",
        "output_pattern": "hammer_cam{cam_id}.usd",
    },
    "brush": {
        "template": "brush_cam1.usd",
        "output_pattern": "brush_cam{cam_id}.usd",
    },
    "drill": {
        "template": "drill_cam1.usd",
        "output_pattern": "drill_cam{cam_id}.usd",
    },
    "mug": {
        "template": "mug_cam1.usd",
        "output_pattern": "mug_cam{cam_id}.usd",
    },
    "spoon": {
        "template": "spoon_cam1.usd",
        "output_pattern": "spoon_cam{cam_id}.usd",
    },
}


def copy_xform_ops(src_prim, dst_prim):
    """Replace destination xform ops with the source xform ops."""
    for prop in list(dst_prim.GetProperties()):
        name = prop.GetName()
        if name == "xformOpOrder" or name.startswith("xformOp:"):
            dst_prim.RemoveProperty(name)

    for src_attr in src_prim.GetAttributes():
        name = src_attr.GetName()
        if name != "xformOpOrder" and not name.startswith("xformOp:"):
            continue

        dst_attr = dst_prim.CreateAttribute(
            name,
            src_attr.GetTypeName(),
            custom=src_attr.IsCustom(),
            variability=src_attr.GetVariability(),
        )

        if src_attr.GetNumTimeSamples() > 0:
            for time_code in src_attr.GetTimeSamples():
                dst_attr.Set(src_attr.Get(time_code), time_code)
        else:
            dst_attr.Set(src_attr.Get())


def get_camera_xform_summary(stage, camera_prim_path):
    from pxr import UsdGeom

    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim or not camera_prim.IsValid():
        return "<missing camera prim>"
    matrix = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
    translation = matrix.ExtractTranslation()
    return f"translation=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f})"


def transplant_camera_pose(source_usd, template_usd, output_usd, camera_prim_path, overwrite):
    from pxr import Usd

    source_usd = Path(source_usd)
    template_usd = Path(template_usd)
    output_usd = Path(output_usd)

    if not source_usd.exists():
        raise FileNotFoundError(f"Source camera USD not found: {source_usd}")
    if not template_usd.exists():
        raise FileNotFoundError(f"Template USD not found: {template_usd}")
    if output_usd.exists() and not overwrite:
        print(f"⏭️  Skip existing: {output_usd}")
        return

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    same_file_as_template = output_usd.resolve() == template_usd.resolve()
    if not same_file_as_template:
        shutil.copy2(template_usd, output_usd)

    source_stage = Usd.Stage.Open(str(source_usd))
    if same_file_as_template:
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f"{output_usd.stem}_",
            suffix=output_usd.suffix,
            dir=output_usd.parent,
            delete=False,
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        shutil.copy2(template_usd, temp_path)
        output_stage = Usd.Stage.Open(str(temp_path))
    else:
        temp_path = None
        output_stage = Usd.Stage.Open(str(output_usd))

    if source_stage is None:
        raise RuntimeError(f"Failed to open source USD: {source_usd}")
    if output_stage is None:
        raise RuntimeError(f"Failed to open output USD: {output_usd}")

    source_camera = source_stage.GetPrimAtPath(camera_prim_path)
    output_camera = output_stage.GetPrimAtPath(camera_prim_path)
    if not source_camera or not source_camera.IsValid():
        raise RuntimeError(f"Missing camera prim {camera_prim_path} in {source_usd}")
    if not output_camera or not output_camera.IsValid():
        raise RuntimeError(f"Missing camera prim {camera_prim_path} in {output_usd}")

    copy_xform_ops(source_camera, output_camera)
    summary = get_camera_xform_summary(output_stage, camera_prim_path)
    output_stage.GetRootLayer().Save()
    source_stage = None
    output_stage = None
    if temp_path is not None:
        shutil.move(str(temp_path), str(output_usd))
    print(f"✅ {output_usd.name}: copied {camera_prim_path} from {source_usd.name} ({summary})")


def resolve_object_patterns(args):
    patterns = OBJECT_USD_PATTERNS[args.object]
    template = args.template or patterns["template"]
    output_pattern = args.output_pattern or patterns["output_pattern"]
    return template, output_pattern


def parse_args():
    parser = argparse.ArgumentParser(description="Generate object_camN.usd files from calibrated camera view USDs.")
    parser.add_argument("--object", choices=SUPPORTED_OBJECTS, default="mug", help="Object USD set to generate.")
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--template", default=None, help="Template object scene USD. Defaults are selected by --object.")
    parser.add_argument("--source-pattern", default="cam{cam_id}_r.usd", help="USD pattern that stores camera poses.")
    parser.add_argument("--output-pattern", default=None, help="Output USD pattern. Defaults are selected by --object.")
    parser.add_argument("--camera-prim", default=DEFAULT_CAMERA_PRIM)
    parser.add_argument("--cam-ids", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7], help="Camera ids to generate. Default also rewrites cam1 from cam1_r.usd.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output USD files.")
    return parser.parse_args()


def main():
    args = parse_args()
    template_name, output_pattern = resolve_object_patterns(args)
    template_usd = args.scene_dir / template_name

    print(f"🎯 object={args.object}")
    print(f"📄 template={template_usd}")
    print(f"📄 output_pattern={output_pattern}")

    for cam_id in args.cam_ids:
        source_usd = args.scene_dir / args.source_pattern.format(cam_id=cam_id)
        output_usd = args.scene_dir / output_pattern.format(cam_id=cam_id)
        transplant_camera_pose(
            source_usd=source_usd,
            template_usd=template_usd,
            output_usd=output_usd,
            camera_prim_path=args.camera_prim,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
