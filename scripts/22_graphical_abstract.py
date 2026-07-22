"""
Graphical abstract for Applied Energy submission.
Spec: landscape, readable at 5 x 13 cm, 300 dpi -> ~1535 x 590 px.
Pipeline: real data -> net load -> ML forecast -> CQR intervals -> robust MILP -> findings.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(P, "outputs", "figures")

fig, ax = plt.subplots(figsize=(13/2.54, 5/2.54), dpi=420)
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis('off')

boxes = [
    (1,    'DATA', 'CAISO 2023\nEIA + NOAA\n115k records', '#E3F2FD', '#1565C0'),
    (17.4, 'NET LOAD', 'demand − solar\n− wind\n−10.4 / +44.9 GW', '#FFF8E1', '#F57F17'),
    (33.8, 'FORECAST', 'XGBoost 24 h\nR² = 0.854\n+18.5% skill', '#E8F5E9', '#2E7D32'),
    (50.2, 'CQR', '90% intervals\n88.8% coverage\ndistribution-free', '#F3E5F5', '#6A1B9A'),
    (66.6, 'ROBUST MILP', 'two-tier gas UC\n5 GW BESS\nγ ∈ [0, 1]', '#FFEBEE', '#B71C1C'),
]
BW, BH, Y = 13.6, 26, 7

for x, title, body, fc, ec in boxes:
    ax.add_patch(FancyBboxPatch((x, Y), BW, BH, boxstyle="round,pad=0.4",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + BW/2, Y + BH - 3.2, title, ha='center', va='center',
            fontsize=5.6, fontweight='bold', color=ec)
    ax.text(x + BW/2, Y + BH/2 - 3, body, ha='center', va='center',
            fontsize=4.6, color='#333333', linespacing=1.5)

for i in range(len(boxes) - 1):
    x0 = boxes[i][0] + BW + 0.6
    x1 = boxes[i + 1][0] - 0.6
    ax.add_patch(FancyArrowPatch((x0, Y + BH/2), (x1, Y + BH/2),
                                 arrowstyle='-|>', mutation_scale=7,
                                 color='#616161', lw=1.1))

# Findings panel
fx = 83
ax.add_patch(FancyBboxPatch((fx, 3), 15.5, 34, boxstyle="round,pad=0.4",
                            fc='#263238', ec='#263238', lw=1.2))
ax.text(fx + 7.75, 33.2, 'FINDINGS', ha='center', va='center',
        fontsize=5.6, fontweight='bold', color='white')
ax.text(fx + 7.75, 18,
        'no nuclear:\n+13.2% cost\n+17.3% CO₂\n\nrobust (γ=1):\n+4.2% cost\n+21.8% CO₂\n\nVOPI: 0.74%',
        ha='center', va='center', fontsize=4.6, color='white', linespacing=1.55)
ax.add_patch(FancyArrowPatch((boxes[-1][0] + BW + 0.6, Y + BH/2),
                             (fx - 0.6, Y + BH/2),
                             arrowstyle='-|>', mutation_scale=7,
                             color='#616161', lw=1.1))

ax.text(1, 2.2, 'Duck curve dispatch under uncertainty: conformalized ML + robust MILP',
        fontsize=4.8, style='italic', color='#555555')

fig.savefig(os.path.join(FIG, "graphical_abstract.png"),
            bbox_inches='tight', pad_inches=0.05)
print("-> graphical_abstract.png")
