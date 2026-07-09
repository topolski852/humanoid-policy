# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.
"""Named-pose library for the humanoid robot.

A *pose* is a base pose (position + wxyz quaternion) plus per-joint target positions
(radians). Poses are stored by name in a single YAML file (default: ``configs/poses.yaml``)
so the pose editor can add/load/remove them and training can spawn a chosen pose *by name* --
removing the need for spawn-height guessing + settle time (a saved pose stores the exact
snapped floor height in ``base_pos[2]``).

This module is deliberately Isaac-free (pure Python + PyYAML) so it is importable in unit
tests, inside the sim GUI, and by the training env without pulling heavy dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

DEFAULT_LIBRARY_PATH = "configs/poses.yaml"

_SIDE_A, _SIDE_B = "left", "right"


def mirror_joint_name(name: str) -> tuple[str, float]:
    """Map a joint to its sagittal-plane mirror.

    Returns ``(mirrored_name, sign)``. A sagittal reflection swaps ``left``<->``right`` and
    flips the sign of roll/yaw joints (their axes lie in the sagittal plane); pitch joints keep
    their sign. Joints with no side token map to themselves with sign ``+1``.
    """
    if f"_{_SIDE_A}_" in name:
        mirrored = name.replace(f"_{_SIDE_A}_", f"_{_SIDE_B}_")
    elif f"_{_SIDE_B}_" in name:
        mirrored = name.replace(f"_{_SIDE_B}_", f"_{_SIDE_A}_")
    else:
        mirrored = name
    sign = -1.0 if ("roll" in name or "yaw" in name) else 1.0
    return mirrored, sign


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``."""
    return lo if value < lo else hi if value > hi else value


def mirror_joint_pos(joint_pos: dict, source_side: str = _SIDE_A) -> dict:
    """Return a copy of ``joint_pos`` with ``source_side`` values mirrored onto the other side."""
    out = dict(joint_pos)
    tok = f"_{source_side}_"
    for name, val in joint_pos.items():
        if tok in name:
            mirrored, sign = mirror_joint_name(name)
            out[mirrored] = sign * float(val)
    return out


def clamp_pose_to_limits(joint_pos: dict, limits: dict) -> dict:
    """Clamp each joint to ``limits[name] = (lo, hi)``; joints absent from ``limits`` pass through."""
    out = {}
    for name, val in joint_pos.items():
        if name in limits:
            lo, hi = limits[name]
            out[name] = clamp(float(val), float(lo), float(hi))
        else:
            out[name] = float(val)
    return out


@dataclass
class Pose:
    """A named robot pose: floating-base pose + joint targets."""

    name: str
    base_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_quat: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # wxyz
    joint_pos: dict = field(default_factory=dict)  # joint_name -> radians
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "base_pos": [float(x) for x in self.base_pos],
            "base_quat": [float(x) for x in self.base_quat],
            "joint_pos": {k: float(v) for k, v in self.joint_pos.items()},
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Pose":
        return cls(
            name=name,
            base_pos=list(d.get("base_pos", [0.0, 0.0, 0.0])),
            base_quat=list(d.get("base_quat", [1.0, 0.0, 0.0, 0.0])),
            joint_pos=dict(d.get("joint_pos", {})),
            note=d.get("note", ""),
        )


def load_library(path: str = DEFAULT_LIBRARY_PATH) -> dict:
    """Load ``{name: Pose}`` from the YAML library (empty dict if the file is missing)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {name: Pose.from_dict(name, d) for name, d in (data.get("poses", {}) or {}).items()}


def save_library(lib: dict, path: str = DEFAULT_LIBRARY_PATH) -> None:
    """Write ``{name: Pose}`` to the YAML library (creates parent dirs)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {"poses": {name: p.to_dict() for name, p in lib.items()}}
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True, default_flow_style=False)


def get_pose(name: str, path: str = DEFAULT_LIBRARY_PATH) -> Optional[Pose]:
    return load_library(path).get(name)


def save_pose(pose: Pose, path: str = DEFAULT_LIBRARY_PATH) -> None:
    lib = load_library(path)
    lib[pose.name] = pose
    save_library(lib, path)


def delete_pose(name: str, path: str = DEFAULT_LIBRARY_PATH) -> bool:
    lib = load_library(path)
    if name in lib:
        del lib[name]
        save_library(lib, path)
        return True
    return False


def list_poses(path: str = DEFAULT_LIBRARY_PATH) -> list:
    return sorted(load_library(path).keys())
