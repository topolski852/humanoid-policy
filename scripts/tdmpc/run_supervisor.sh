#!/usr/bin/env bash
# nohup + auto-restart wrapper for the autonomous TD-MPC2 walk supervisor (Layer 1).
#
# Keeps supervisor.py alive across crashes for a week-long unattended run: if it exits
# non-zero it is relaunched (with backoff); if it exits cleanly at idle-at-cap it drops the
# DONE sentinel and this wrapper stops. Survives logout (run under nohup) but NOT a full
# machine reboot — after a reboot just re-run this same command (state is persisted, so it
# picks up where it left off).
#
# Start (detached, survives logout):
#   cd /home/nse/humanoid/humanoid-policy
#   nohup bash scripts/tdmpc/run_supervisor.sh > logs/tdmpc/_runlogs/wrapper.log 2>&1 &
#
# Stop everything:
#   touch logs/tdmpc/SUPERVISOR_DONE        # tell the wrapper to stop restarting
#   pkill -f scripts/tdmpc/supervisor.py    # kill the current supervisor + its training child
#
# Any args to this script are forwarded to supervisor.py (e.g. --max_wins 5).

set -u
cd "$(dirname "$0")/../.." || exit 1     # repo root
REPO_ROOT="$(pwd)"
PY="$REPO_ROOT/.venv/bin/python"
SUP="scripts/tdmpc/supervisor.py"
DONE="logs/tdmpc/SUPERVISOR_DONE"
LOG="logs/tdmpc/_runlogs/supervisor.log"
mkdir -p logs/tdmpc/_runlogs

backoff=15
max_backoff=300
echo "[wrapper $(date '+%F %T')] starting supervisor wrapper (repo=$REPO_ROOT)"

while true; do
  if [ -f "$DONE" ]; then
    echo "[wrapper $(date '+%F %T')] DONE sentinel present ($DONE) — supervisor finished; wrapper exiting."
    break
  fi

  echo "[wrapper $(date '+%F %T')] launching supervisor (log -> $LOG)"
  OMNI_KIT_ACCEPT_EULA=YES "$PY" "$SUP" "$@" >> "$LOG" 2>&1
  rc=$?
  echo "[wrapper $(date '+%F %T')] supervisor exited rc=$rc"

  if [ -f "$DONE" ]; then
    echo "[wrapper $(date '+%F %T')] idle-at-cap reached; wrapper exiting."
    break
  fi
  if [ "$rc" -eq 0 ]; then
    # clean exit without a DONE sentinel (e.g. queue grace elapsed) — don't hot-loop.
    echo "[wrapper $(date '+%F %T')] clean exit, no more work; wrapper exiting."
    break
  fi

  echo "[wrapper $(date '+%F %T')] crash (rc=$rc) — restarting in ${backoff}s"
  sleep "$backoff"
  backoff=$(( backoff * 2 )); [ "$backoff" -gt "$max_backoff" ] && backoff=$max_backoff
done
