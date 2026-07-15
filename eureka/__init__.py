"""Eureka-style automated reward-weight tuner for the humanoid-policy trainer.

API-free: a local evolutionary search over the walk reward-term weights, graded by
a weight-invariant ground-truth fitness read from each run's TensorBoard, with
plateau-based early-stop of every training run. Wraps the existing rsl_rl trainer;
does not modify it. See README.md.
"""
