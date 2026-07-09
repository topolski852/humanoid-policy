# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.
"""Interactive pose editor / placement tool for the humanoid robot.

Runs the real training robot (same USD, collision meshes, joint limits, ground) so a pose can be
authored and *verified* on the floor instead of guessed + settled. Edit each joint with sliders,
mirror one side onto the other, place the floating base, snap the robot exactly onto the floor,
and save named poses to a library (``configs/poses.yaml``) that training can later spawn by name.

Usage:
    # GUI (Kit window):
    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/rsl_rl/pose_editor.py --variant standup-biped --viz kit
    # headless self-test of the core logic (no GUI):
    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/rsl_rl/pose_editor.py --variant standup-biped --headless --selftest

The heavy Isaac logic lives in ``PoseEditorCore`` (exercised by --selftest); the omni.ui panel is
a thin layer on top.
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import variants  # isort: skip

# ----------------------------------------------------------------------------- CLI + app launch
parser = argparse.ArgumentParser(description="Humanoid pose editor / placement tool.")
parser.add_argument("--task", type=str, default=None, help="Gym task id (usually set via --variant).")
parser.add_argument("--poses_file", type=str, default="configs/poses.yaml", help="Pose library YAML path.")
parser.add_argument("--pose", type=str, default=None, help="Name of a pose to load on startup.")
parser.add_argument("--selftest", action="store_true", help="Run headless logic self-test and exit.")
variants.add_variant_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
variants.resolve_variant(args_cli)
if args_cli.task is None:
    args_cli.task = variants.VARIANTS["standup-biped"]
if args_cli.selftest:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------- deferred imports
import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation

import omni.usd
from pxr import UsdGeom, UsdPhysics, Usd

if not args_cli.headless:
    import omni.ui as ui

from isaaclab_tasks.utils import parse_env_cfg
import humanoid_policy.tasks  # noqa: F401  (registers gym tasks)
from humanoid_policy import pose_lib


def rpy_to_wxyz(roll, pitch, yaw):
    x, y, z, w = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()  # scipy: xyzw
    return [float(w), float(x), float(y), float(z)]


def wxyz_to_rpy(quat_wxyz):
    w, x, y, z = quat_wxyz
    r, p, y_ = Rotation.from_quat([x, y, z, w]).as_euler("xyz")
    return [float(r), float(p), float(y_)]


class PoseEditorCore:
    """All Isaac-side pose logic: read/write joint & base state, limits, floor snap, save/load."""

    def __init__(self, task, device, poses_file):
        self.poses_file = poses_file
        env_cfg = parse_env_cfg(task, device=device, num_envs=1)
        self.env = gym.make(task, cfg=env_cfg)
        self.env.reset()
        self.u = self.env.unwrapped
        self.sim = self.u.sim
        self.robot = self.u.scene["robot"]
        self.device = self.robot.device

        self.joint_names = list(self.robot.data.joint_names)
        self.n_joints = len(self.joint_names)
        # soft position limits per joint name -> (lo, hi)
        lim = self.robot.data.soft_joint_pos_limits[:][0].cpu().numpy()  # (n_joints, 2)
        self.limits = {n: (float(lim[i][0]), float(lim[i][1])) for i, n in enumerate(self.joint_names)}

        # robot root prim (parent of the base body)
        base_path = self.robot.root_physx_view.prim_paths[0]
        self.robot_prim = base_path.rsplit("/", 1)[0]
        self.stage = omni.usd.get_context().get_stage()
        self.body_names = list(self.robot.data.body_names)
        self._precompute_body_corners()
        self._disable_gravity()

        # Settle a few steps with gravity OFF so live data (esp. root_quat_w) is valid before we
        # capture the initial held state. Capturing pre-step gave a bogus base orientation that,
        # because the torso collider sits ~0.7 m above the base origin, swung min-z wildly.
        for _ in range(5):
            self.sim.step(render=False)
        self.robot.update(self.sim.get_physics_dt())
        self._q = self.robot.data.joint_pos[:][0].detach().clone()  # (n_joints,)
        self._base_pos = self.robot.data.root_pos_w[:][0].detach().clone()  # (3,)
        self._base_quat = self.robot.data.root_quat_w[:][0].detach().clone()  # (4,) wxyz

    def _precompute_body_corners(self):
        """Per body, cache the 8 corners of its collision AABB in the body's LOCAL frame.

        Physics runs in fabric and does not sync body transforms back to USD, so a USD world-bound
        is stale. Instead we take each rigid body's (pose-invariant) local AABB once, then transform
        its corners by the LIVE physics body pose to get the true world min-z at any pose.
        """
        # include 'guide'/'proxy' purposes: collision meshes are typically purpose='guide'
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        )
        self._body_corners = []  # list aligned with body_names; None if body has no collision geom
        for bn in self.body_names:
            body = self.stage.GetPrimAtPath(f"{self.robot_prim}/{bn}")
            corners = None
            if body and body.IsValid():
                # Each body Xform has 'visuals' + 'collisions' children. Floor contact is the
                # COLLISION geometry only (visuals overhang and would over-estimate the extent).
                # ComputeRelativeBound(collisions, body) gives that geometry in the body's frame,
                # which we then place with the LIVE physics body pose (body_pos_w / body_quat_w).
                col = self.stage.GetPrimAtPath(f"{self.robot_prim}/{bn}/collisions")
                target = col if (col and col.IsValid()) else body
                rng = cache.ComputeRelativeBound(target, body).ComputeAlignedRange()
                if not rng.IsEmpty():
                    mn, mx = rng.GetMin(), rng.GetMax()
                    corners = np.array([[mn[0] if (k & 1) else mx[0],
                                         mn[1] if (k & 2) else mx[1],
                                         mn[2] if (k & 4) else mx[2]] for k in range(8)])
            self._body_corners.append(corners)

    # ---- gravity: authoring holds pose kinematically; zero gravity avoids fighting it ----
    def _set_gravity(self, magnitude):
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                UsdPhysics.Scene(prim).GetGravityMagnitudeAttr().Set(float(magnitude))
                break

    def _disable_gravity(self):
        self._set_gravity(0.0)

    # ---- joint access (by name; sim order is interleaved so never assume it) ----
    def joint_index(self, name):
        return self.joint_names.index(name)

    def get_joint_pos(self):
        q = self._q.cpu().numpy()
        return {n: float(q[i]) for i, n in enumerate(self.joint_names)}

    def set_joint(self, name, value):
        lo, hi = self.limits[name]
        self._q[self.joint_index(name)] = pose_lib.clamp(float(value), lo, hi)

    def set_joint_pos(self, joint_pos):
        for n, v in joint_pos.items():
            if n in self.limits:
                self.set_joint(n, v)

    def mirror(self, source_side="left"):
        self.set_joint_pos(pose_lib.mirror_joint_pos(self.get_joint_pos(), source_side))

    # ---- base access ----
    def get_base(self):
        return self._base_pos.cpu().numpy().tolist(), self._base_quat.cpu().numpy().tolist()

    def set_base_pos(self, xyz):
        self._base_pos = torch.tensor(xyz, dtype=self._base_pos.dtype, device=self.device)

    def set_base_rpy(self, roll, pitch, yaw):
        self._base_quat = torch.tensor(rpy_to_wxyz(roll, pitch, yaw), dtype=self._base_quat.dtype, device=self.device)

    # ---- write target to sim + step PHYSICS ONLY (no render: safe inside UI callbacks) ----
    def apply(self, steps=1):
        q = self._q.unsqueeze(0)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.write_joint_position_to_sim(q)  # also hold via drive target
        pose = torch.cat([self._base_pos, self._base_quat]).unsqueeze(0)
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))
        for _ in range(steps):
            self.sim.step(render=False)  # rendering is driven once per frame by the main loop
        self.robot.update(self.sim.get_physics_dt())

    # ---- floor geometry: world min-z from LIVE physics body poses ----
    def min_z(self):
        self.robot.update(self.sim.get_physics_dt())
        bp = self.robot.data.body_pos_w[:][0].cpu().numpy()   # (n_bodies, 3)
        bq = self.robot.data.body_quat_w[:][0].cpu().numpy()  # (n_bodies, 4) wxyz
        zmin = float("inf")
        for i, corners in enumerate(self._body_corners):
            if corners is None:
                continue
            w, x, y, z = bq[i]
            rot = Rotation.from_quat([x, y, z, w]).as_matrix()
            world = corners @ rot.T + bp[i]
            zmin = min(zmin, float(world[:, 2].min()))
        return zmin

    def snap_to_floor(self, clearance=0.0):
        """Shift base z so the lowest robot point rests exactly on z=0 (+clearance)."""
        self.apply(steps=1)
        z = self.min_z()
        self._base_pos[2] += (clearance - z)
        self.apply(steps=1)
        return self.min_z()

    def floor_penetration(self, eps=1e-3):
        return self.min_z() < -eps

    # ---- stability test: hold the authored joints via actuators under gravity (rigid-hold) ----
    def base_tilt_deg(self):
        w, x, y, z = self.robot.data.root_quat_w[:][0].cpu().numpy()
        up = Rotation.from_quat([x, y, z, w]).as_matrix()[:, 2]  # body z-axis in world
        return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))

    def base_z(self):
        return float(self.robot.data.root_pos_w[:][0][2].cpu())

    def begin_stability_test(self):
        """Freeze the authored pose, switch gravity on, and prepare to hold joints via the actuators."""
        self._saved = (self._q.clone(), self._base_pos.clone(), self._base_quat.clone())
        self._set_gravity(9.81)
        self._test_z0 = self.base_z()
        self._test_max_tilt = self.base_tilt_deg()

    def step_stability_test(self):
        q = self._q.unsqueeze(0)
        self.robot.set_joint_position_target(q)  # actuators (real PD gains + effort limits) hold it
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(self.sim.get_physics_dt())
        self._test_max_tilt = max(self._test_max_tilt, self.base_tilt_deg())
        return self.base_tilt_deg(), self.base_z()

    def end_stability_test(self):
        z = self.base_z()
        fell = self._test_max_tilt > 45.0 or z < self._test_z0 - 0.15
        # restore authored pose and go back to kinematic hold
        self._set_gravity(0.0)
        self._q, self._base_pos, self._base_quat = self._saved
        self.apply(steps=1)
        return {"fell": fell, "max_tilt": self._test_max_tilt, "z": z, "z0": self._test_z0}

    # ---- pose library ----
    def current_pose(self, name, note=""):
        pos, quat = self.get_base()
        return pose_lib.Pose(name=name, base_pos=pos, base_quat=quat, joint_pos=self.get_joint_pos(), note=note)

    def save_current(self, name, note=""):
        pose_lib.save_pose(self.current_pose(name, note), self.poses_file)

    def load_pose(self, name):
        p = pose_lib.get_pose(name, self.poses_file)
        if p is None:
            return False
        self.set_base_pos(p.base_pos)
        self._base_quat = torch.tensor(p.base_quat, dtype=self._base_quat.dtype, device=self.device)
        self.set_joint_pos(p.joint_pos)
        self.apply(steps=1)
        return True


def run_selftest(core):
    print("\n==================== POSE EDITOR SELF-TEST ====================")
    ok = True

    # 1) hold: set a distinctive pose, step, read back
    core.set_joint("leg_left_hip_roll_joint", 0.30)
    core.apply(steps=30)
    got = core.get_joint_pos()["leg_left_hip_roll_joint"]
    held = abs(got - 0.30) < 0.02
    print(f"[hold]  left_hip_roll target=0.300 readback={got:.4f}  -> {'OK' if held else 'FAIL'}")
    ok &= held

    # 2) limit clamp
    hi = core.limits["leg_left_hip_roll_joint"][1]
    core.set_joint("leg_left_hip_roll_joint", hi + 5.0)
    clamped = abs(core.get_joint_pos()["leg_left_hip_roll_joint"] - hi) < 1e-6
    print(f"[clamp] over-limit clamped to {hi:.4f} -> {'OK' if clamped else 'FAIL'}")
    ok &= clamped
    core.set_joint("leg_left_hip_roll_joint", 0.0)

    # 3) AABB tracks base: raise base 0.5, min_z should rise ~0.5
    core.apply(steps=1)
    z0 = core.min_z()
    pos, _ = core.get_base()
    core.set_base_pos([pos[0], pos[1], pos[2] + 0.5])
    core.apply(steps=1)
    z1 = core.min_z()
    tracks = abs((z1 - z0) - 0.5) < 0.05
    print(f"[aabb]  min_z {z0:.4f} -> {z1:.4f} (delta {z1-z0:.4f}, expect ~0.5) -> {'OK' if tracks else 'FAIL'}")
    ok &= tracks

    # 4) snap to floor -> min_z ~ 0
    zsnap = core.snap_to_floor()
    snapped = abs(zsnap) < 5e-3
    _, _ = core.get_base()
    print(f"[snap]  min_z after snap = {zsnap:.5f} (base_z={core.get_base()[0][2]:.4f}) -> {'OK' if snapped else 'FAIL'}")
    ok &= snapped

    # 5) penetration flag: push base down 0.05 -> should report penetration
    pos, _ = core.get_base()
    core.set_base_pos([pos[0], pos[1], pos[2] - 0.05])
    core.apply(steps=1)
    pen = core.floor_penetration()
    print(f"[pen]   after -5cm, penetration={pen} -> {'OK' if pen else 'FAIL'}")
    ok &= pen
    core.snap_to_floor()

    # 5b) ABSOLUTE floor check: let gravity settle the robot onto the real floor, then our
    #     computed min_z must read ~0. This catches constant frame/offset bugs that relative
    #     tests (snap, delta) cannot, since it compares against sim physics ground truth.
    pos, _ = core.get_base()
    core.set_base_pos([pos[0], pos[1], pos[2] + 0.15])  # lift so it drops onto the floor
    core.apply(steps=1)
    core._set_gravity(9.81)
    for _ in range(400):
        core.sim.step(render=False)
    core.robot.update(core.sim.get_physics_dt())
    z_settled = core.min_z()
    # tolerance 5 cm: catches gross constant offsets (earlier bugs were -18..-20 cm) while allowing
    # real PhysX soft-contact penetration as the deep squat topples into a resting heap.
    settled_ok = abs(z_settled) < 0.05
    print(f"[abs]   min_z after gravity settle = {z_settled*100:+.2f} cm (expect ~0) -> {'OK' if settled_ok else 'FAIL'}")
    ok &= settled_ok
    core._set_gravity(0.0)

    # 6) save + reload from a temp library
    import tempfile, os
    core.poses_file = os.path.join(tempfile.mkdtemp(), "poses.yaml")
    core.save_current("selftest_pose", note="from selftest")
    names = pose_lib.list_poses(core.poses_file)
    reloaded = core.load_pose("selftest_pose")
    print(f"[lib]   saved+listed={names} reload={reloaded} -> {'OK' if (names==['selftest_pose'] and reloaded) else 'FAIL'}")
    ok &= (names == ["selftest_pose"] and reloaded)

    print(f"==================== SELF-TEST {'PASSED' if ok else 'FAILED'} ====================\n")
    return ok


_RED = 0xFF3B3BFF     # ABGR: penetration / error
_GREEN = 0xFF5BD65B   # ABGR: on the floor, ok


class PoseEditorUI:
    """omni.ui panel that drives PoseEditorCore. Only built in GUI mode."""

    def __init__(self, core):
        self.core = core
        self._updating = False           # guards programmatic model updates (mirror recursion)
        self.mirror_mode = False
        self.joint_models = {}           # joint name -> ui.SimpleFloatModel
        self.base_models = {}            # 'x'/'y'/'z'/'roll'/'pitch'/'yaw' -> model
        self.status = None
        self.testing = False             # stability-test state machine
        self._test_frames_left = 0
        self._test_secs = 3.0
        self.window = ui.Window("Humanoid Pose Editor", width=460, height=900)
        self.window.frame.set_build_fn(self._populate)  # rebuild() re-reads core + pose list

    # ---- side helpers ----
    @staticmethod
    def _side(name):
        return "left" if "_left_" in name else "right" if "_right_" in name else "other"

    def _rebuild(self):
        self.window.frame.rebuild()

    def _populate(self):
        """(Re)build the whole panel from current core state + pose library."""
        self.joint_models = {}
        self.base_models = {}
        names = self.core.joint_names
        with ui.ScrollingFrame():
            with ui.VStack(spacing=6, height=0):
                # ---- pose library ----
                ui.Label("POSE LIBRARY", height=20)
                self._pose_names = pose_lib.list_poses(self.core.poses_file)
                self.combo = ui.ComboBox(0, *(self._pose_names or ["<none>"]))
                with ui.HStack(height=24, spacing=4):
                    ui.Button("Load", clicked_fn=self._on_load)
                    ui.Button("Delete", clicked_fn=self._on_delete)
                    ui.Button("Refresh", clicked_fn=self._rebuild)
                with ui.HStack(height=24, spacing=4):
                    ui.Label("Save as:", width=60)
                    self.name_field = ui.StringField()
                    ui.Button("Save", width=70, clicked_fn=self._on_save)

                ui.Separator(height=6)
                # ---- status + snap + mirror ----
                self.status = ui.Label("status: --", height=22, style={"color": _GREEN})
                with ui.HStack(height=26, spacing=4):
                    ui.Button("Snap to floor", clicked_fn=self._on_snap)
                    self.mirror_cb = ui.CheckBox(width=20)
                    self.mirror_cb.model.set_value(self.mirror_mode)
                    self.mirror_cb.model.add_value_changed_fn(self._on_mirror_toggle)
                    ui.Label("live mirror (edit L -> R)")
                with ui.HStack(height=26, spacing=4):
                    ui.Button("Test stability (rigid hold, 3s)", clicked_fn=self._on_test_stability)
                    ui.Label("watch it hold or topple")

                ui.Separator(height=6)
                # ---- base 6-DOF ----
                ui.Label("BASE", height=20)
                (bx, by, bz), bq = self.core.get_base()
                br, bp, byaw = wxyz_to_rpy(bq)
                self._add_base_slider("x", bx, -3.0, 3.0)
                self._add_base_slider("y", by, -3.0, 3.0)
                self._add_base_slider("z", bz, 0.0, 1.5)
                self._add_base_slider("roll", br, -3.1416, 3.1416)
                self._add_base_slider("pitch", bp, -3.1416, 3.1416)
                self._add_base_slider("yaw", byaw, -3.1416, 3.1416)

                ui.Separator(height=6)
                # ---- joints, grouped by side ----
                q = self.core.get_joint_pos()
                for side in ("left", "right", "other"):
                    group = [n for n in names if self._side(n) == side]
                    if not group:
                        continue
                    ui.Label(f"{side.upper()} JOINTS", height=20)
                    for n in group:
                        self._add_joint_slider(n, q[n])

    # ---- widget builders ----
    def _add_base_slider(self, key, val, lo, hi):
        with ui.HStack(height=22, spacing=4):
            ui.Label(key, width=110)
            model = ui.SimpleFloatModel(val)
            ui.FloatSlider(model=model, min=lo, max=hi)
            model.add_value_changed_fn(lambda m, k=key: self._on_base_change(k, m.get_value_as_float()))
            self.base_models[key] = model

    def _add_joint_slider(self, name, val):
        lo, hi = self.core.limits[name]
        with ui.HStack(height=22, spacing=4):
            ui.Label(name.replace("_joint", "").replace("leg_", "").replace("arm_", ""), width=180)
            model = ui.SimpleFloatModel(val)
            ui.FloatSlider(model=model, min=lo, max=hi)
            model.add_value_changed_fn(lambda m, n=name: self._on_joint_change(n, m.get_value_as_float()))
            self.joint_models[name] = model

    # ---- callbacks ----
    def _on_joint_change(self, name, value):
        if self._updating:
            return
        self.core.set_joint(name, value)
        if self.mirror_mode and self._side(name) == "left":
            mirrored, sign = pose_lib.mirror_joint_name(name)
            if mirrored in self.joint_models:
                self.core.set_joint(mirrored, sign * value)
                self._set_model(self.joint_models[mirrored], self.core.get_joint_pos()[mirrored])

    def _on_base_change(self, key, value):
        if self._updating:
            return
        pos, quat = self.core.get_base()
        rpy = wxyz_to_rpy(quat)
        if key in ("x", "y", "z"):
            idx = {"x": 0, "y": 1, "z": 2}[key]
            pos[idx] = value
            self.core.set_base_pos(pos)
        else:
            idx = {"roll": 0, "pitch": 1, "yaw": 2}[key]
            rpy[idx] = value
            self.core.set_base_rpy(*rpy)

    def _on_mirror_toggle(self, model):
        self.mirror_mode = model.get_value_as_bool()

    def _on_test_stability(self):
        if self.testing:
            return
        self.core.begin_stability_test()
        self._test_frames_left = int(self._test_secs / self.core.sim.get_physics_dt())
        self.testing = True

    def step_test(self):
        """Advance the stability test one frame (called from the main loop so render isn't nested)."""
        tilt, z = self.core.step_stability_test()
        self._test_frames_left -= 1
        if self.status is not None:
            self.status.text = f"TESTING... tilt={tilt:4.1f} deg  z={z:+.3f} m"
            self.status.set_style({"color": _RED if tilt > 45 else _GREEN})
        if self._test_frames_left <= 0:
            res = self.core.end_stability_test()
            self.testing = False
            if self.status is not None:
                verdict = "FELL / UNSTABLE" if res["fell"] else "STABLE (held upright)"
                self.status.text = f"{verdict}  |  max tilt={res['max_tilt']:.1f} deg  final z={res['z']:+.3f} m"
                self.status.set_style({"color": _RED if res["fell"] else _GREEN})

    def _on_snap(self):
        self.core.snap_to_floor()
        self._rebuild()

    def _on_load(self):
        name = self._selected_name()
        if name and self.core.load_pose(name):
            self._rebuild()

    def _on_save(self):
        name = self.name_field.model.get_value_as_string().strip()
        if not name:
            return
        self.core.save_current(name)
        self._rebuild()

    def _on_delete(self):
        name = self._selected_name()
        if name:
            pose_lib.delete_pose(name, self.core.poses_file)
            self._rebuild()

    # ---- helpers ----
    def _selected_name(self):
        if not self._pose_names:
            return None
        idx = self.combo.model.get_item_value_model().get_value_as_int()
        return self._pose_names[idx] if 0 <= idx < len(self._pose_names) else None

    def _set_model(self, model, value):
        self._updating = True
        model.set_value(float(value))
        self._updating = False

    def update_status(self):
        z = self.core.min_z()
        pen = z < -1e-3
        self.status.text = f"min_z = {z*100:+.2f} cm   {'PENETRATION' if pen else 'on floor / clear'}"
        self.status.set_style({"color": _RED if pen else _GREEN})


def main():
    core = PoseEditorCore(args_cli.task, args_cli.device, args_cli.poses_file)
    if args_cli.pose:
        core.load_pose(args_cli.pose)
    if args_cli.selftest:
        ok = run_selftest(core)
        core.env.close()
        simulation_app.close()
        sys.exit(0 if ok else 1)

    core.snap_to_floor()
    editor = PoseEditorUI(core)
    frame = 0
    while simulation_app.is_running():
        if editor.testing:
            editor.step_test()   # let gravity act, hold joints via actuators, watch it hold/topple
        else:
            core.apply(steps=1)  # physics-only: re-holds the authored pose kinematically
        core.sim.render()        # single render + UI pump per frame (no nested frames)
        frame += 1
        if not editor.testing and frame % 10 == 0 and editor.status is not None:
            editor.update_status()
    core.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
