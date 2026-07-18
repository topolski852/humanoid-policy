"""TD-MPC2 model primitives, adapted from the official MIT repo (github.com/nicklashansen/tdmpc2).

Simplified for our single-task, state-only, non-episodic, non-multitask setting and stripped of
the torchrl/tensordict `from_modules`+`vmap` ensemble machinery (replaced by a plain ModuleList
Q-ensemble) so it runs against torch 2.11 / no-torchrl. See configs/actuators/PROVENANCE-style
note in docs/proactive_world_model/DESIGN.md.
"""
