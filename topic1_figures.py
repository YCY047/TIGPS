import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# topic1_figures.py
# 所有變項建構、樣本與模型設定都和 topic1_final.py 相同：
#   - 問卷題目：負數 → 遺漏；depression / well-being 只保留 0–4
#   - 量表分數：所有題目都有有效答案才計算平均
#   - Gender：1 = Female, 2 = Male → female = 1 / 0
#   - SES (bs3)：1–10；HOUWGT > 0
#   - 模型：WLS (HOUWGT) + school-clustered SE (nschool_id)
# 係數全部在本程式中估計，不手動輸入。
#
# Figure 1      → final RQ1（Selective → Online / Offline self-image）
# Figure RQ2    → final RQ2B（Selective Low/Medium/High × self-image group）
# Figure 3A/3B  → final RQ3B（Online–Offline gap size → psychological outcomes）
# ============================================================


# ------------------------------------------------------------
# 0. Data preparation（同 topic1_final.py 第 1–14 節）
# ------------------------------------------------------------
df = pd.read_csv("TIGPSw2_s.csv", low_memory=False)

selective_vars = ["bs23a", "bs23b", "bs23c"]
online_image_vars = ["bs25a", "bs25b", "bs25c"]
offline_image_vars = ["bs25d", "bs25e", "bs25f"]
self_esteem_vars = ["bs53a", "bs53b", "bs53c"]
depression_vars = [f"bs56{c}" for c in "abcdefghijklmn"]
wellbeing_vars = [f"bs58{c}" for c in "abcde"]
background_vars = ["bs1", "bs3", "HOUWGT", "nschool_id"]

questionnaire_vars = (
    selective_vars + online_image_vars + offline_image_vars
    + self_esteem_vars + depression_vars + wellbeing_vars
)

topic1 = df[questionnaire_vars + background_vars].copy()

for col in questionnaire_vars:
    topic1[col] = pd.to_numeric(topic1[col], errors="coerce")
    topic1[col] = topic1[col].where(topic1[col] >= 0)

for col in depression_vars + wellbeing_vars:
    topic1[col] = topic1[col].where(topic1[col].between(0, 4))

for col in background_vars:
    topic1[col] = pd.to_numeric(topic1[col], errors="coerce")

topic1["bs1"] = topic1["bs1"].where(topic1["bs1"].isin([1, 2]))
topic1["bs3"] = topic1["bs3"].where(topic1["bs3"].between(1, 10))
topic1["HOUWGT"] = topic1["HOUWGT"].where(topic1["HOUWGT"] > 0)
topic1["female"] = topic1["bs1"].map({1: 1, 2: 0})


def scale_score(items):
    return (
        topic1[items].mean(axis=1)
        .where(topic1[items].notna().sum(axis=1) == len(items))
    )


topic1["selective_presentation_score"] = scale_score(selective_vars)
topic1["online_self_image_score"] = scale_score(online_image_vars)
topic1["offline_self_image_score"] = scale_score(offline_image_vars)
topic1["self_esteem_score"] = scale_score(self_esteem_vars)
topic1["wellbeing_score"] = scale_score(wellbeing_vars)
topic1["depression_score"] = scale_score(depression_vars)


# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------
def zscore(x):
    return (x - x.mean()) / x.std()


def fit_weighted_cluster(formula, data):
    return smf.wls(
        formula=formula, data=data, weights=data["HOUWGT"]
    ).fit(cov_type="cluster", cov_kwds={"groups": data["nschool_id"]})


def fmt_beta(b):
    # APA style: no leading zero, typographic minus sign
    s = f"{abs(b):.3f}".lstrip("0")
    return ("−" if b < 0 else "") + s


def fmt_p(p):
    return "p < .001" if p < .001 else "p = " + f"{p:.3f}".lstrip("0")


# ============================================================
# FIGURE 1  (final RQ1)
# Selective Positive Self-presentation → Online / Offline Self-image
# ============================================================
rq1_vars = [
    "selective_presentation_score", "online_self_image_score",
    "offline_self_image_score", "female", "bs3", "HOUWGT", "nschool_id",
]
rq1 = topic1[rq1_vars].dropna().copy()

for col in ["selective_presentation_score", "online_self_image_score",
            "offline_self_image_score", "bs3"]:
    rq1[col + "_z"] = zscore(rq1[col])

model_online = fit_weighted_cluster(
    "online_self_image_score_z ~ selective_presentation_score_z + female + bs3_z",
    rq1,
)
model_offline = fit_weighted_cluster(
    "offline_self_image_score_z ~ selective_presentation_score_z + female + bs3_z",
    rq1,
)

online_beta = model_online.params["selective_presentation_score_z"]
offline_beta = model_offline.params["selective_presentation_score_z"]

print("Figure 1 analytic N =", len(rq1))
print(f"Figure 1: online β = {online_beta:.4f}, offline β = {offline_beta:.4f}")

# Predicted lines on the original 1–4 scale.
# Gender is held at the sample proportion of girls; SES at its mean (z = 0).
sel_mean, sel_sd = rq1["selective_presentation_score"].agg(["mean", "std"])
on_mean, on_sd = rq1["online_self_image_score"].agg(["mean", "std"])
off_mean, off_sd = rq1["offline_self_image_score"].agg(["mean", "std"])
female_share = rq1["female"].mean()

x_raw = np.linspace(1, 4, 200)
x_z = (x_raw - sel_mean) / sel_sd


def predict_raw(model, y_mean, y_sd):
    b = model.params
    pred_z = (
        b["Intercept"]
        + b["selective_presentation_score_z"] * x_z
        + b["female"] * female_share
    )
    return y_mean + pred_z * y_sd


online_pred_raw = predict_raw(model_online, on_mean, on_sd)
offline_pred_raw = predict_raw(model_offline, off_mean, off_sd)

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x_raw, online_pred_raw, linewidth=3, color="tab:blue",
        label="Online Positive Self-image")
ax.plot(x_raw, offline_pred_raw, linewidth=3, linestyle="--",
        color="tab:red", label="Offline Positive Self-image")

ax.set_xlim(1, 4)
ax.set_xticks(np.arange(1, 4.01, 0.5))
ax.set_xticklabels(["1\nLow", "1.5", "2", "2.5", "3", "3.5", "4\nHigh"])
ax.set_ylim(1, 4)
ax.set_yticks(np.arange(1, 4.01, 0.5))

ax.set_xlabel("Selective Positive Self-presentation", fontsize=12)
ax.set_ylabel("Model-estimated Positive Self-image", fontsize=12)
ax.set_title("Selective Self-presentation and\n"
             "Online vs. Offline Positive Self-image", fontsize=14, pad=12)

# β labels: placed above / below their line at the right end
label_x = 3.05
span = x_raw >= label_x
ax.text(label_x, online_pred_raw[span].max() + 0.15,
        f"Online: β = {fmt_beta(online_beta)}",
        color="tab:blue", fontsize=11, fontweight="bold",
        ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3))
ax.text(label_x, offline_pred_raw[span].min() - 0.2,
        f"Offline: β = {fmt_beta(offline_beta)}",
        color="tab:red", fontsize=11, fontweight="bold",
        ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

ax.legend(frameon=False, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.25)

fig.text(
    0.01, -0.01,
    "Note. Lines are model-estimated values from weighted (HOUWGT) regressions "
    "with school-clustered SEs, controlling for gender and family SES\n"
    "(gender held at the sample proportion of girls, SES at its mean). "
    "β = standardized coefficient.",
    fontsize=9, color="dimgray", ha="left", va="top",
)

plt.tight_layout()
plt.savefig("figure1_selective_self_presentation.png", dpi=300,
            bbox_inches="tight")
plt.show()


# ============================================================
# FIGURE RQ2  (final RQ2A / RQ2B)
# Selective Low / Medium / High × Online–Offline self-image group
# 100% stacked bar chart (unweighted, same as final)
# ============================================================
rq2_gap_tolerance = 1e-9

rq2_plot = topic1[[
    "selective_presentation_score", "online_self_image_score",
    "offline_self_image_score",
]].copy()
rq2_plot["self_image_gap"] = (
    rq2_plot["online_self_image_score"] - rq2_plot["offline_self_image_score"]
)
rq2_plot = rq2_plot[["selective_presentation_score", "self_image_gap"]].dropna()

rq2_plot["image_group"] = np.select(
    [
        rq2_plot["self_image_gap"] > rq2_gap_tolerance,
        rq2_plot["self_image_gap"] < -rq2_gap_tolerance,
    ],
    ["Online > Offline", "Online < Offline"],
    default="Online = Offline",
)

# Exploratory fixed cut-points (same as final):
# Low <= 2, 2 < Medium <= 3, 3 < High <= 4
rq2_plot["selective_group"] = pd.cut(
    rq2_plot["selective_presentation_score"],
    bins=[0, 2, 3, 4],
    labels=["Low", "Medium", "High"],
    include_lowest=True,
)

group_order = ["Low", "Medium", "High"]
image_order = ["Online > Offline", "Online = Offline", "Online < Offline"]

rq2_pct = pd.crosstab(
    rq2_plot["selective_group"], rq2_plot["image_group"], normalize="index"
).reindex(index=group_order, columns=image_order, fill_value=0) * 100

print("\nRQ2 Figure N =", len(rq2_plot))
print(rq2_pct.round(2))

colors = ["#347EB2", "#8999AA", "#D9965F"]
fig, ax = plt.subplots(figsize=(10, 5.5))
left = np.zeros(len(group_order))

for group, color in zip(image_order, colors):
    values = rq2_pct[group].to_numpy()
    bars = ax.barh(group_order, values, left=left, color=color, height=0.6,
                   label=group, edgecolor="white", linewidth=1)
    for bar, value in zip(bars, values):
        if value >= 5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%", ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold")
    left += values

ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_xticklabels([f"{x}%" for x in range(0, 101, 20)])
ax.invert_yaxis()  # Low → Medium → High (top to bottom)

ax.set_xlabel("Percentage of Students", fontsize=12)
ax.set_ylabel("Selective Positive Self-presentation", fontsize=12)
ax.set_title("Online–Offline Self-image Profiles\n"
             "by Selective Positive Self-presentation", fontsize=14, pad=15)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
          frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig("figure_rq2_stacked_bar.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# FIGURE 3A–3B  (final RQ3B, main specification)
# Online / Offline self-image direction → Psychological outcomes
#   sample  = Equal group (gap = 0, anchor) + students in that direction
#   outcome ~ gap_size + level_z + female + bs3_z
#   WLS with HOUWGT, school-clustered SE (nschool_id)
# Lines = model-estimated values (covariates at weighted sample means),
# bands = 95% CI, dots = unadjusted weighted means per gap value (n >= 30).
# β = change in the standardized outcome per 1-point gap.
# ============================================================
rq3_outcomes = [
    {"var": "self_esteem_score", "name": "Self-esteem", "color": "#7B3294"},
    {"var": "wellbeing_score", "name": "Well-being", "color": "#1B9E77"},
    {"var": "depression_score", "name": "Depressive symptoms", "color": "#D95F02"},
]

rq3_base = topic1[[
    "online_self_image_score", "offline_self_image_score",
    "self_esteem_score", "wellbeing_score", "depression_score",
    "female", "bs3", "HOUWGT", "nschool_id",
]].dropna(subset=[
    "online_self_image_score", "offline_self_image_score",
    "female", "bs3", "HOUWGT", "nschool_id",
]).copy()

rq3_base["gap"] = (
    rq3_base["online_self_image_score"] - rq3_base["offline_self_image_score"]
).round(4)
rq3_base["gap_group"] = np.select(
    [rq3_base["gap"] > 0, rq3_base["gap"] < 0],
    ["Online > Offline", "Offline > Online"],
    default="Equal",
)
rq3_base["gap_size"] = rq3_base["gap"].abs()
rq3_base["self_image_level"] = (
    rq3_base["online_self_image_score"] + rq3_base["offline_self_image_score"]
) / 2

print("\nFigure 3 group sizes:")
print(rq3_base["gap_group"].value_counts()
      .reindex(["Online > Offline", "Equal", "Offline > Online"]))


def prepare_rq3_sample(data, outcome):
    """Complete cases for one outcome; standardize within that sample (same as final)."""
    sample = data.dropna(subset=[outcome]).copy()
    sample["outcome_z"] = zscore(sample[outcome])
    sample["level_z"] = zscore(sample["self_image_level"])
    sample["bs3_z"] = zscore(sample["bs3"])
    return sample


fig3_specs = [
    {
        "direction": "Online > Offline",
        "file": "figure3a_online_higher_outcomes.png",
        "title": "When Online Self-image Exceeds Offline Self-image",
        "xlabel": "Online − Offline self-image gap (scale points)",
    },
    {
        "direction": "Offline > Online",
        "file": "figure3b_offline_higher_outcomes.png",
        "title": "When Offline Self-image Exceeds Online Self-image",
        "xlabel": "Offline − Online self-image gap (scale points)",
    },
]
FIG3_MIN_N_DOT = 30
Y_LO, Y_HI = 0, 4   # shared axis: self-esteem 1–4; well-being & depression 0–4

for spec in fig3_specs:
    subset = rq3_base[rq3_base["gap_group"].isin(["Equal", spec["direction"]])]

    fig, ax = plt.subplots(figsize=(11, 7))
    end_labels = []   # (y at x = 3, text, colour)

    print(f"\nFigure 3 ({spec['direction']}):")
    for out in rq3_outcomes:
        d = prepare_rq3_sample(subset, out["var"])

        rhs = "gap_size + level_z + female + bs3_z"
        m_raw = fit_weighted_cluster(f"{out['var']} ~ {rhs}", d)   # original scale → line
        m_std = fit_weighted_cluster(f"outcome_z ~ {rhs}", d)      # standardized → β

        b = m_std.params["gap_size"]
        p = m_std.pvalues["gap_size"]
        print(f"  {out['name']}: β = {b:.4f}, p = {p:.3g}, N = {int(m_std.nobs)}")

        grid = pd.DataFrame({
            "gap_size": np.linspace(0, 3, 61),
            "level_z": np.average(d["level_z"], weights=d["HOUWGT"]),
            "female": np.average(d["female"], weights=d["HOUWGT"]),
            "bs3_z": np.average(d["bs3_z"], weights=d["HOUWGT"]),
        })
        pred = m_raw.get_prediction(grid).summary_frame(alpha=0.05)

        ax.fill_between(grid["gap_size"], pred["mean_ci_lower"],
                        pred["mean_ci_upper"], color=out["color"],
                        alpha=0.15, linewidth=0)
        ax.plot(grid["gap_size"], pred["mean"], color=out["color"], linewidth=3)

        obs = (
            d.groupby("gap_size")
            .apply(lambda g: pd.Series({
                "mean": np.average(g[out["var"]], weights=g["HOUWGT"]),
                "n": len(g),
            }), include_groups=False)
        )
        obs = obs[obs["n"] >= FIG3_MIN_N_DOT]
        ax.scatter(obs.index, obs["mean"], s=30, color=out["color"],
                   alpha=0.55, edgecolor="white", linewidth=0.8, zorder=3)

        end_labels.append((
            pred["mean"].iloc[-1],
            f"{out['name']}: β = {fmt_beta(b)}, {fmt_p(p)}",
            out["color"],
        ))

    # x 軸畫到 4；落差最大只可能是 3（兩個量表都是 1–4），所以線停在 3
    ax.set_xlim(0, 4)
    ax.set_xticks(np.arange(0, 4.01, 0.5))
    ax.set_xticklabels(["0\n(equal)", "0.5", "1", "1.5", "2", "2.5", "3",
                        "3.5", "4"])
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks(np.arange(Y_LO, Y_HI + 0.01, 0.5))

    ax.set_xlabel(spec["xlabel"], fontsize=12)
    ax.set_ylabel("Model-estimated Outcome Score (original scale)", fontsize=12)
    ax.set_title(spec["title"] + "\nand Three Psychological Outcomes",
                 fontsize=14, pad=12)

    # Labels in the right margin, sorted by height and pushed apart
    min_gap = 0.22
    end_labels.sort(key=lambda t: t[0])
    label_y = [t[0] for t in end_labels]
    for i in range(1, len(label_y)):
        label_y[i] = max(label_y[i], label_y[i - 1] + min_gap)
    overflow = label_y[-1] - (Y_HI - 0.05)
    if overflow > 0:
        label_y = [v - overflow for v in label_y]
        for i in range(len(label_y) - 2, -1, -1):
            label_y[i] = min(label_y[i], label_y[i + 1] - min_gap)

    for (y_end, text, color), y_lab in zip(end_labels, label_y):
        ax.annotate(
            text, xy=(3, y_end), xytext=(3.08, y_lab),
            textcoords="data", annotation_clip=False,
            color=color, fontsize=10.5, fontweight="bold",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                            shrinkA=0, shrinkB=2),
        )

    mark_handles = [
        Line2D([0], [0], color="dimgray", lw=3, label="Model-estimated value"),
        Patch(facecolor="dimgray", alpha=0.2, label="95% CI"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="dimgray",
               markeredgecolor="white", alpha=0.6, markersize=7,
               label="Observed weighted mean"),
    ]
    ax.legend(handles=mark_handles, frameon=False, loc="lower left", ncol=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.25)

    # 終止線：落差最大可能值 = 3，3–4 之間不可能有資料
    ax.axvspan(3, 4, color="#F2F2F2", zorder=0, linewidth=0)
    ax.axvline(3, color="dimgray", linestyle="--", linewidth=1.2, zorder=1)
    # 文字預設放在頂端；若和右側 β 標籤重疊，改放在離標籤最遠的高度
    note_y = Y_HI - 0.25
    if min(abs(note_y - y) for y in label_y) < 0.45:
        candidates = np.arange(0.5, Y_HI - 0.2, 0.05)
        note_y = max(candidates,
                     key=lambda c: min(abs(c - y) for y in label_y))
    ax.text(3.5, note_y, "Beyond maximum\npossible gap (3)",
            ha="center", va="center", fontsize=9.5, color="dimgray",
            style="italic")

    n_dir = (subset["gap_group"] == spec["direction"]).sum()
    n_eq = (subset["gap_group"] == "Equal").sum()
    fig.text(
        0.01, -0.01,
        f"Note. Sample = students with equal online/offline self-image (gap = 0, n = {n_eq:,}) "
        f"plus students in the {spec['direction']} group (n = {n_dir:,}).\n"
        "Lines are model-estimated values from weighted (HOUWGT) regressions with school-clustered SEs, "
        "controlling for overall self-image level\n"
        "[(online + offline) / 2], gender and family SES (held at weighted sample means). "
        f"Dots = unadjusted weighted means at each gap value (n ≥ {FIG3_MIN_N_DOT}).\n"
        "Outcomes use different scales (self-esteem 1–4; well-being and depressive symptoms 0–4), "
        "so compare slopes rather than line heights.\n"
        "Higher depressive-symptom scores indicate worse mental health. "
        "β = change in the standardized outcome per 1-point gap.\n"
        "The maximum possible gap is 3 because both self-image scales range from 1 to 4 "
        "(dashed line); the shaded area beyond 3 contains no possible values.",
        fontsize=9, color="dimgray", ha="left", va="top",
    )

    plt.tight_layout()
    plt.savefig(spec["file"], dpi=300, bbox_inches="tight")
    plt.show()