"""Load the vendored bench-fitted motor models (configs/actuators/*.json).

Thin loader so the robot cfg reads friction/inertia straight from the validated JSONs
instead of hardcoding magic numbers. See configs/actuators/PROVENANCE.md for the two
fields we deliberately DON'T take from the JSON (latency_s and torque_limit).
"""

from __future__ import annotations

import json
import os

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "actuators")
)


def load_actuator_model(name: str) -> dict:
    """Return the fitted-model dict for ``name`` (e.g. ``"m6c12_pitch"``)."""
    path = os.path.join(_CONFIG_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)
