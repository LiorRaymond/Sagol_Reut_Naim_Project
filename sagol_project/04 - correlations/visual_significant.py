import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# 1) Locate results directory
BASE = Path(__file__).parent
RES  = BASE / "results"

# 2) Read the “significant” CSVs
df_ari = (
    pd.read_csv(RES / "significant_feature_ARI.csv")
      .rename(columns={"Feature_1": "Feature", "r_with_ARI": "r"})
)
df_sc = (
    pd.read_csv(RES / "significant_feature_SCARED.csv")
      .rename(columns={"Feature_1": "Feature", "r_with_SCARED": "r"})
)

df_ari["test"] = "ARI"
df_sc["test"]  = "SCARED"
df = pd.concat(
    [df_ari[["Feature","r","test"]],
     df_sc[["Feature","r","test"]]],
    ignore_index=True
)

# 3) Prepare pivot and feature lists
pivot     = df.pivot(index="Feature", columns="test", values="r")
ari_feats = set(df_ari["Feature"])
sc_feats  = set(df_sc["Feature"])
features  = list(pivot.index)
y_pos     = np.arange(len(features))
offset    = 0.2

# determine horizontal span for full-lines
all_rs = pivot.values.flatten()
x_min = np.nanmin(all_rs) * 1.1
x_max = np.nanmax(all_rs) * 1.1

# 4) Plot with horizontal guide-lines and overlap diamonds
fig, ax = plt.subplots(figsize=(8, max(4, len(features)*0.3)))

# horizontal background lines
for y in y_pos:
    ax.hlines(y, x_min, x_max, color="#ddd", linewidth=0.8)

# plot markers and connectors
for i, feat in enumerate(features):
    for test, dy, marker in [
        ("ARI",    -offset, "o"),
        ("SCARED", +offset, "^"),
    ]:
        if feat not in (ari_feats if test=="ARI" else sc_feats):
            continue
        r = pivot.loc[feat, test]
        color = "red" if r > 0 else "blue"
        # vertical connector from baseline to marker
        ax.vlines(r, i, i+dy, linestyles='--', colors=color, linewidth=1, alpha=0.6)
        ax.scatter(r, i+dy,
                   color=color, marker=marker,
                   s=80, edgecolor="k", linewidth=0.8)

    # overlap diamond at central y=i
    if feat in (ari_feats & sc_feats):
        r1, r2 = pivot.loc[feat, ["ARI","SCARED"]].values
        avg_r = np.nanmean([r1, r2])
        ax.scatter(avg_r, i,
                   marker="D", color="black",
                   s=100, alpha=0.8, label="_nolegend_")

# vertical zero line
ax.axvline(0, color="grey", lw=1)

# labels
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.set_xlim(x_min, x_max)
ax.set_xlabel("Correlation coefficient (r)")
ax.set_title("Significant Features: ARI vs SCARED")

# legend
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="grey",
           markersize=8, label="ARI"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor="grey",
           markersize=8, label="SCARED"),
    Line2D([0],[0], marker="D", color="black", markersize=8,
           label="Overlap"),
    Line2D([0],[0], linestyle='--', color="grey", label="Connector"),
    Line2D([0],[0], marker="o", color="red", label="Positive r"),
    Line2D([0],[0], marker="o", color="blue", label="Negative r"),
]
ax.legend(handles=legend_elems, loc="best")

plt.tight_layout()
plt.show()


