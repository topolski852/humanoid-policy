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

## 1. Start a FRESH FULL training run (stand-up biped)

16384 envs x 48 steps x 3500 iters = ~2.75B samples. Auto-enlarges GPU buffers for the big env count.

```bash
cd /home/nse/humanoid/humanoid-policy
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/rsl_rl/train.py \
  --variant standup-biped --profile full --headless
```

---

## 2. Watch the PLAYBACK (latest checkpoint, Kit window)

Auto-loads the newest checkpoint. Use 16 envs and NO --profile so it keeps cheap GPU buffers.
`--viz kit` is REQUIRED to open the GUI window — in Isaac Lab 3.0 the default is headless, so
without a `--viz` backend no window appears. (`--viz` is an alias of `--visualizer`; CSV backends
are `kit,newton,rerun,viser`.)

```bash
cd /home/nse/humanoid/humanoid-policy
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/rsl_rl/play.py --variant standup-biped --num_envs 16 --viz kit
```

To pin a specific run/checkpoint instead of the newest, add:
`--load_run <TIMESTAMP_DIR> --checkpoint model_XXXX.pt`

---

## 3. RESUME after a crash (don't lose progress)

Checkpoints save every 100 iters. Find the latest, then resume from it.

Find the latest run + checkpoint:
```bash
ls -dt /home/nse/humanoid/humanoid-policy/logs/rsl_rl/standup_biped/*/ | head -1
ls -tr /home/nse/humanoid/humanoid-policy/logs/rsl_rl/standup_biped/*/model_*.pt | tail -3
```

Resume (fill in the run dir + checkpoint; set max_iterations = 3500 minus the checkpoint number,
e.g. crashed near 2100 -> 3500 - 2100 = 1400):
```bash
cd /home/nse/humanoid/humanoid-policy
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/rsl_rl/train.py \
  --variant standup-biped --profile full --headless \
  --resume True --load_run <TIMESTAMP_DIR> --checkpoint model_XXXX.pt \
  --max_iterations <REMAINING>
```

---

## 4. Clear logs to start over clean

```bash
rm -rf /home/nse/humanoid/humanoid-policy/logs/rsl_rl/standup_biped
```

---

## 5. GPU housekeeping / diagnostics

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
