# Humanoid-Policy — copy-paste command sheet

Self-contained repo. Everything runs from this repo's own `.venv` (created by `uv`); no external
Berkeley/Isaac install is required. Run everything from the repo root:
`cd /home/nse/humanoid/humanoid-policy`

---

## 0. One-time setup (fresh clone)

Create the repo-local environment (pulls Isaac Sim / Isaac Lab / torch from the pinned indexes;
uses `uv.lock` for exact versions):

```bash
cd /home/nse/humanoid/humanoid-policy
uv sync
```

Isaac Sim's Omniverse Kit shows a one-time EULA prompt on first launch. For headless/background
runs, accept it non-interactively by exporting this once (add it to your shell profile if you like):

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

---

## 1. Variants, profiles, and log locations

`--variant` picks the task; `--profile` picks the scale. Any variant runs at either scale.

| `--variant` | log directory |
|---|---|
| `walk-biped` | `logs/rsl_rl/biped/` |
| `standup-biped` | `logs/rsl_rl/standup_biped/` |
| `squat-biped` | `logs/rsl_rl/squat_biped/` |
| `walk-humanoid`, `standup-humanoid` | (full-humanoid variants, rarely used) |

| `--profile` | envs | steps/env | iters | wall time |
|---|---|---|---|---|
| `fast` | 4096 | 24 | 6000 | ~2 h |
| `full` | 24576 | 64 | 6000 | much longer |

Explicit `--num_envs` / `--max_iterations` always override the profile. With **no** `--profile`,
the task's own default is used (750 iters for the biped cfg) — so always pass one for a real run.

Source of truth: `scripts/rsl_rl/profiles.py`, `scripts/rsl_rl/variants.py`.

---

## 2. Environment variables — READ BEFORE ANY WALK RUN

The walk task's plant and rewards are selected by env var, **not** by CLI flag. Getting these
wrong silently trains a different thing, and the run looks perfectly healthy while doing it.
Every one of them is echoed at startup — check the `[INFO]` lines before walking away.

| variable | default | values | what it changes |
|---|---|---|---|
| `HUMANOID_SMOOTH_PRESET` | **`off`** | `off,a,b,c,d` | action_rate / action_l2 / dof_vel weights |
| `HUMANOID_OBS_MODEL` | `measured` | `measured,legacy` | observation staleness + encoder quantisation + noise scales |
| `HUMANOID_GAIN_PRESET` | `tuned` | `tuned,berkeley,tuned_kd3` | kp/kd only |
| `HUMANOID_ACTUATOR_MODEL` | `1` | `1,0` | `1` = bench stick-slip + delayed PD; `0` = friction-free implicit baseline |

⚠️ **The `HUMANOID_SMOOTH_PRESET` trap.** The default is `off` (action_rate −0.014, the other two
zero). The deployed Smooth A and Smooth B bundles were trained at `a` and `b`. A run without the
variable set is **not comparable to anything previously captured on hardware**. Set it explicitly,
every time.

`measured` vs `legacy` observations is not a simple noise-scale swap — see
`.../velocity/mdp/observations.py`. Use `legacy` only to reproduce a bundle trained before
2026-09-23.

---

## 3. Train

### Walk (the current line of work)

```bash
cd /home/nse/humanoid/humanoid-policy
OMNI_KIT_ACCEPT_EULA=YES \
HUMANOID_OBS_MODEL=measured HUMANOID_SMOOTH_PRESET=a \
HUMANOID_GAIN_PRESET=tuned HUMANOID_ACTUATOR_MODEL=1 \
.venv/bin/python scripts/rsl_rl/train.py \
  --variant walk-biped --profile full --headless \
  --seed 42 --run_name measA-full --plateau
```

- `--run_name` suffixes the log dir (`logs/rsl_rl/biped/<timestamp>_measA-full/`). Always set it;
  a bare timestamp tells you nothing six weeks later.
- `--seed 42` matches the existing comparison runs. Keep it fixed when A/B-ing.
- `--plateau` keeps `model_best.pt` and early-stops at the reward plateau. Use it on long runs;
  **omit it when doing a like-for-like comparison against a fixed-iteration run**, or an early
  stop will confound the result.
- Checkpoints save every 100 iters.

### Stand-up

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/rsl_rl/train.py \
  --variant standup-biped --profile full --headless
```

### Quick dev iteration (~2 h)

Swap `--profile full` for `--profile fast`. Use this to sanity-check a config change before
committing to a full run.

---

## 4. Watch the PLAYBACK (Kit window)

`--viz kit` is **REQUIRED** to open the GUI window — in Isaac Lab 3.0 the default is headless, so
without a `--viz` backend no window appears. (`--viz` is an alias of `--visualizer`; CSV backends
are `kit,newton,rerun,viser`.)

Use 16 envs and NO `--profile` so it keeps cheap GPU buffers. Auto-loads the newest checkpoint.

```bash
cd /home/nse/humanoid/humanoid-policy
OMNI_KIT_ACCEPT_EULA=YES HUMANOID_OBS_MODEL=measured HUMANOID_SMOOTH_PRESET=a \
.venv/bin/python scripts/rsl_rl/play.py --variant walk-biped --num_envs 16 --viz kit
```

Pin a specific run/checkpoint with `--load_run <TIMESTAMP_DIR> --checkpoint model_XXXX.pt`.
Set the same env vars you trained with, or you are watching a policy run on a different plant.
`--cmd_vx <m/s>` fixes the forward command instead of sampling it.

**Side effect: `play.py` also EXPORTS.** It writes `logs/.../exported/policy.onnx` + `policy.pt`,
and `configs/policy_latest.yaml` + `configs/leg_policy_contract.json` (relative to CWD). That is
how a bundle gets produced — see §6.

---

## 5. Evaluate — compare two policies, or sim against hardware

```bash
OMNI_KIT_ACCEPT_EULA=YES EVAL_GAIT=1 HUMANOID_SMOOTH_PRESET=a HUMANOID_OBS_MODEL=measured \
.venv/bin/python scripts/rsl_rl/eval_plant_compare.py --plant modeled \
  --num_envs 128 --steps 600 --cmd_vx 0.3 --headless \
  --load_run <TIMESTAMP_DIR> --checkpoint model_5999.pt --out eval.json
```

`--plant modeled|baseline` selects the actuator plant (overrides `HUMANOID_ACTUATOR_MODEL`).
`EVAL_GAIT=1` adds the gait metrics; `EVAL_GAIT=0` or unset disables them.

Reports `fall_rate_per_min`, `base_accel_rms`, `rocking_rms`, `action_rate_rms`,
`gait_hz_median`, `knee_corr_median`, and per-joint `torque_sat_frac` / `torque_p95_nm` /
`torque_peak_nm`. The per-joint torque figures are directly comparable to the hardware tables in
`humanoid-control/docs/measurements/`.

**Judge runs on this, not on reward,** and not on `scripts/screen_gait.py` — that screener gives
false negatives (`docs/walk-smoothness-sweep.md` §"ideal-loop screener").

---

## 6. Export a bundle for the robot

1. Run `play.py` (§4) on the checkpoint you want. It writes `exported/policy.onnx`, `policy.pt`,
   and `configs/policy_latest.yaml` + `configs/leg_policy_contract.json`.
2. **Copy them into `deploy/walk/` by hand.** There is no automation for this step, and
   `deploy/walk/` silently goes stale otherwise.
3. Confirm `deploy/walk/leg_policy_contract.json` carries the current effort limits (12.0 N·m on
   the 8 M6C12 joints, 7.0 on the 4 MAD5010 ankles). They flow automatically from
   `_CONTRACT_EFFORT` in `humanoid_policy_assets/robots/humanoid.py`, but only on a re-export.
4. Archive the previous bundle under `deploy/walk/archive/<date>_<name>/`.

Hardware-side flashing (ESC `torque_limit` / `current_limit`) is covered in
`humanoid-control/docs/HARDWARE_PLAN_2026-09-23.md`.

---

## 7. RESUME after a crash (don't lose progress)

Find the latest run + checkpoint:
```bash
ls -dt logs/rsl_rl/biped/*/ | head -1
ls -tr logs/rsl_rl/biped/*/model_*.pt | tail -3
```

Resume. Set `--max_iterations` to the REMAINING count (profile total 6000 minus the checkpoint
number, e.g. crashed near 2100 → 6000 − 2100 = 3900). **Re-export the same env vars** — they are
not stored in the resumed state:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
HUMANOID_OBS_MODEL=measured HUMANOID_SMOOTH_PRESET=a \
.venv/bin/python scripts/rsl_rl/train.py \
  --variant walk-biped --profile full --headless \
  --resume True --load_run <TIMESTAMP_DIR> --checkpoint model_XXXX.pt \
  --max_iterations <REMAINING>
```

---

## 8. Cleaning up runs

Delete the **specific run directory**, never the parent task folder — that is where every trained
policy for the task lives:

```bash
rm -rf logs/rsl_rl/biped/<TIMESTAMP_DIR>      # one throwaway run
```

---

## 9. GPU housekeeping / diagnostics

Enable persistence mode (reduces transient GPU channel-stall crashes; re-run after reboot):
```bash
sudo nvidia-smi -pm 1
```

Live GPU status:
```bash
nvidia-smi
```

Check for GPU faults after a crash (Xid = hardware/driver fault; 702 launch-timeout = engine hang):
```bash
sudo dmesg -T | grep -iE "xid|nvrm|fell off" | tail -20
```

**Do not raise the PhysX collision-pair buffers to "fix" a large-env-count crash.** Enlarging
`found_lost` / `aggregate` to 2**27 pre-allocates ~4 GB of VRAM and starves PPO into CUDA OOM. A
`foundLostPairs` overflow at high env counts is a physics blow-up symptom, not a buffer shortage —
see the note at the bottom of `scripts/rsl_rl/profiles.py`.

**Intermittent Kit startup crash in `XOpenDisplay`.** Seen once on a `--headless` run launched
from a VS Code / agent shell: Kit's platforminfo plugin crashes trying to open the X display, and
the process dumps core before training starts. It does not reproduce reliably. If you hit it,
relaunch with the display vars stripped:

```bash
env -u DISPLAY -u XAUTHORITY OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/rsl_rl/train.py ...
```

Never do that for `play.py --viz kit` — it needs the display.
