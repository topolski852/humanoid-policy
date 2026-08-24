"""Screen an exported walk policy for a hardware-executable gait. No robot required.

See docs/walk-smoothness-sweep.md sec 5. Judge on gait frequency and whether it still
steps -- NOT on reward: a policy that has collapsed into standing scores well on
smoothness and badly on nothing obvious.

    python scripts/screen_gait.py <path/to/policy.onnx>
"""
import sys
import numpy as np, onnxruntime as ort

ONNX = sys.argv[1] if len(sys.argv) > 1 else "exported/policy.onnx"
DT, ASCALE = 0.04, 0.25
# canonical joint order (L then R): hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
KL, KR, HL, HR = 3, 9, 2, 8
DEFAULT = np.array([0.11, 0.0, -0.24, 0.83, -0.56, -0.07,
                    -0.11, -0.0, -0.24, 0.83, -0.56, 0.07], np.float32)
LO = np.array([-0.1745, -0.9817, -1.8980, 0.0, -0.7854, -0.2618,
               -1.5708, -0.5890, -1.8980, 0.0, -0.7854, -0.2618], np.float32)
HI = np.array([1.5708, 0.5890, 0.9817, 2.4435, 0.7854, 0.2618,
               0.1745, 0.9817, 0.9817, 2.4435, 0.7854, 0.2618], np.float32)
SIGN = np.ones(12, np.float32); SIGN[[6, 7, 11]] = -1.0   # URDF-mirrored right roll/yaw joints

s = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"]); inp = s.get_inputs()[0].name

def rollout(vx, n=500, track=0.9):
    pos, prev, P, A, V = DEFAULT.copy(), np.zeros(12, np.float32), [], [], []
    for k in range(n):
        vel = np.zeros(12, np.float32) if k == 0 else (pos - P[-1]) / DT
        obs = np.concatenate([[vx, 0, 0], [0, 0, 0], [0, 0, -1],
                              SIGN * (pos - DEFAULT), SIGN * vel, prev]).astype(np.float32)
        a = s.run(None, {inp: obs.reshape(1, 45)})[0].ravel()
        P.append(pos.copy()); A.append(a.copy()); V.append(vel.copy())
        a = np.clip(a, -4.0, 4.0)                       # trainer action_limit
        tgt = np.clip(SIGN * (a * ASCALE) + DEFAULT, LO, HI)
        pos = pos + track * (tgt - pos); prev = a
    sl = slice(100, None)
    return np.array(P)[sl], np.array(A)[sl], np.array(V)[sl]

def dom_hz(x):
    x = x - x.mean(); f = np.fft.rfftfreq(len(x), DT); Pw = np.abs(np.fft.rfft(x)) ** 2
    return float(f[1:][np.argmax(Pw[1:])])

def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

print(f"{ONNX}\n{'vx':>5s} {'gait Hz':>8s} {'knee corr':>10s} {'hip corr':>9s} "
      f"{'knee swing':>11s} {'max|v|':>7s} {'S(da^2)':>8s}  verdict")
for vx in (0.3, 0.4, 0.5, 0.6):
    P, A, V = rollout(vx)
    hz, kc, hc = dom_hz(P[:, KL]), corr(P[:, KL], P[:, KR]), corr(P[:, HL], P[:, HR])
    sw, mv = np.ptp(P[:, KL]), np.abs(V).max()
    ar = (np.diff(A, axis=0) ** 2).sum(axis=1).mean()
    if sw < 0.05:                       v = "COLLAPSED (standing)"
    elif hz > 3.5:                      v = "still too fast"
    elif kc > -0.3 and hc > -0.3:       v = "no gait phase"
    elif 1.0 <= hz <= 3.0:              v = "*** GOOD ***"
    else:                               v = "marginal"
    print(f"{vx:5.2f} {hz:8.2f} {kc:+10.2f} {hc:+9.2f} {sw:11.3f} {mv:7.2f} {ar:8.2f}  {v}")
