"""Plot actor/critic loss convergence from the CSV produced by loss_run.py."""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = sys.argv[1] if len(sys.argv) > 1 else "p3_results/loss_full.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "p3_results/fig_loss_full.png"

S, A, C = [], [], []
for r in csv.DictReader(open(f)):
    S.append(float(r["step"])); A.append(float(r["loss_actor"])); C.append(float(r["loss_critic"]))
S, A, C = np.array(S), np.array(A), np.array(C)


def sm(y, k=300):
    k = min(k, max(1, len(y) // 4))
    return np.convolve(y, np.ones(k) / k, mode="valid"), k


cA, kA = sm(A)
cC, kC = sm(C)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
ax[0].plot(S[len(S) - len(cC):] / 1e6, cC, color="#C44E52")
ax[0].set_title("Critic loss convergence"); ax[0].set_xlabel("training steps (millions)")
ax[0].set_ylabel("loss"); ax[0].grid(alpha=0.3)
ax[1].plot(S[len(S) - len(cA):] / 1e6, cA, color="#4C72B0")
ax[1].set_title("Actor loss convergence"); ax[1].set_xlabel("training steps (millions)")
ax[1].set_ylabel("loss"); ax[1].grid(alpha=0.3)
plt.suptitle("MADDPG loss convergence — full method @medium")
plt.tight_layout(); plt.savefig(out); print("saved", out)
