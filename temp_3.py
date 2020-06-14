import numpy as np

x = np.linspace(0, 1, 100)
v = []

for team_1_v in x:
    _v = []
    for team_2_v in x:
        if team_2_v > team_1_v + 0.2:
            score = 1
        elif team_1_v > team_2_v + 0.2:
            score = -1
        else:
            score = (team_2_v - team_1_v) * 3
        _v.append(score)
    v.append(_v)

v = np.array(v)

import matplotlib.pyplot as plt

plt.imshow(
    v,
    aspect='auto',
    cmap='viridis',
    origin='lower',
    extent=(0, 1, 0, 1)
)
cbar = plt.colorbar()
cbar.ax.set_ylabel("Advantage")

plt.xlabel("Team $1$ Speed")
plt.ylabel("Team $-1$ Speed")
plt.axis('scaled')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.tight_layout()
plt.show()


plt.plot(x, -(np.exp(2* x)) / np.exp(2) + 1)
plt.xlabel("Team Speed")
plt.ylabel("Team Skill")
plt.axis('scaled')

plt.xlim((0, 1))
plt.ylim((0, 1))
plt.tight_layout()
plt.show()
