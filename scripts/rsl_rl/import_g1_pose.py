"""Import the G1 safety squat captured on the robot into the training pose library.

Source: ``xr_teleoperate/teleop/safety/poses/safety_squat.yaml`` -- degrees, joint names without
the ``_joint`` suffix, captured from a hand-posed robot (branch ``teleop-safety``, see that repo's
``docs/HANDOFF.md`` S5). Target: ``configs/poses.yaml``, the pose library the training envs spawn
from (radians, URDF joint names, plus a floating-base pose).

Two things this script does beyond a unit conversion:

1. **Clamps into the URDF joint limits and says so.** The captured pose is a *maximal* fold: both
   knees read 1-2 deg PAST the model's hard limit and the hips/ankles sit within 0.005 rad of
   theirs. Feeding an unreachable target to ``track_joint_pose_exp`` caps that reward below 1
   forever; PhysX clamps the PD target anyway. Limits are read from the MJCF that ships with
   xr_teleoperate, which is the same model ``robot_arm_ik.py`` uses.
2. **Leaves the base pose unresolved.** A pose in the library stores the exact floor-snapped base
   height; a joint capture from the robot has no base pose at all. The written entry carries a
   placeholder base and ``needs_floor_snap: true`` in its note -- run ``pose_editor.py`` to snap it
   before training, exactly as the biped poses were authored.

Usage:
    python scripts/rsl_rl/import_g1_pose.py                      # write g1_squat, report clamps
    python scripts/rsl_rl/import_g1_pose.py --dry-run            # report only
    python scripts/rsl_rl/import_g1_pose.py --name g1_squat_v2   # alternate library key
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "humanoid_policy"))

from humanoid_policy import pose_lib  # noqa: E402

DEFAULT_XR_REPO = os.path.expanduser("~/xr_teleoperate")
CAPTURE_REL = "teleop/safety/poses/safety_squat.yaml"
MJCF_REL = "assets/g1/g1_body29_hand14.xml"

# Margin held back from the hard limit when clamping [rad]. 0.01 rad ~= 0.6 deg: enough that the
# PD target is inside the limit (so the joint can actually reach it and hold) without meaningfully
# changing the pose geometry.
LIMIT_MARGIN = 0.01


def read_joint_limits(mjcf_path: str) -> dict:
    """Return ``{joint_name: (lo, hi)}`` in radians from the G1 MJCF."""
    limits = {}
    for joint in ET.parse(mjcf_path).iter("joint"):
        name, rng = joint.get("name"), joint.get("range")
        if name and rng:
            lo, hi = (float(x) for x in rng.split())
            limits[name] = (lo, hi)
    return limits


def read_capture(capture_path: str) -> dict:
    """Return ``{urdf_joint_name: radians}`` from the degrees-and-short-names capture file."""
    data = yaml.safe_load(open(capture_path))
    units = (data.get("meta", {}) or {}).get("units", "degrees")
    if units != "degrees":
        raise SystemExit(f"unexpected units in capture: {units!r} (this script assumes degrees)")
    return {f"{name}_joint": math.radians(float(val)) for name, val in data["joints"].items()}


def clamp_to_limits(joint_pos: dict, limits: dict, margin: float = LIMIT_MARGIN):
    """Clamp each joint into ``[lo+margin, hi-margin]``; return ``(clamped, report_rows)``."""
    out, rows = {}, []
    for name, val in joint_pos.items():
        if name not in limits:
            out[name] = val
            rows.append((name, val, val, None, "NOT IN MODEL"))
            continue
        lo, hi = limits[name]
        clamped = pose_lib.clamp(val, lo + margin, hi - margin)
        headroom = min(val - lo, hi - val)
        if clamped != val:
            status = "CLAMPED (was past hard limit)" if headroom < 0 else "clamped into margin"
        elif headroom < 0.05:
            status = "at limit"
        else:
            status = ""
        out[name] = clamped
        rows.append((name, val, clamped, headroom, status))
    return out, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xr-repo", default=DEFAULT_XR_REPO, help="Path to the xr_teleoperate checkout.")
    parser.add_argument("--capture", default=None, help="Override the capture YAML path.")
    parser.add_argument("--mjcf", default=None, help="Override the G1 MJCF path (source of joint limits).")
    parser.add_argument("--poses-file", default=pose_lib.DEFAULT_LIBRARY_PATH, help="Pose library to write.")
    parser.add_argument("--name", default="g1_squat", help="Library key to write.")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write.")
    args = parser.parse_args()

    capture_path = args.capture or os.path.join(args.xr_repo, CAPTURE_REL)
    mjcf_path = args.mjcf or os.path.join(args.xr_repo, MJCF_REL)
    for path, what in ((capture_path, "capture"), (mjcf_path, "MJCF")):
        if not os.path.exists(path):
            raise SystemExit(f"{what} not found: {path}\n(is the xr_teleoperate teleop-safety branch checked out?)")

    joint_pos = read_capture(capture_path)
    limits = read_joint_limits(mjcf_path)
    clamped, rows = clamp_to_limits(joint_pos, limits)

    print(f"{'joint':26s}{'captured':>10s}{'written':>10s}{'headroom':>10s}  note")
    n_clamped = 0
    for name, before, after, headroom, status in rows:
        head = "     n/a" if headroom is None else f"{headroom:8.4f}"
        print(f"{name:26s}{before:10.4f}{after:10.4f}{head:>10s}  {status}")
        n_clamped += bool(status.startswith("CLAMPED"))
    print(f"\n{n_clamped} joint(s) were outside the model's hard limits and had to be clamped.")

    pose = pose_lib.Pose(
        name=args.name,
        # Placeholder base: yaw-neutral, nominal pelvis height. MUST be replaced by a floor snap.
        base_pos=[0.0, 0.0, 0.30],
        base_quat=[1.0, 0.0, 0.0, 0.0],
        joint_pos=clamped,
        note=(
            f"imported from {os.path.relpath(capture_path, args.xr_repo)} (captured on the robot); "
            f"clamped into URDF limits with {LIMIT_MARGIN} rad margin; needs_floor_snap: true "
            "-- run pose_editor.py to place and snap the base before training"
        ),
    )
    if args.dry_run:
        print(f"\n[dry-run] would write '{args.name}' to {args.poses_file}")
        return 0
    pose_lib.save_pose(pose, args.poses_file)
    print(f"\nwrote '{args.name}' to {args.poses_file}")
    print("NEXT: floor-snap the base pose --")
    print(f"  OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/rsl_rl/pose_editor.py "
          f"--variant standup-g1 --pose {args.name} --viz kit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
