import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf


# ============================================================
# FIGURE 1
# Selective Positive Self-presentation
# → Online / Offline Positive Self-image
# ============================================================

# Read raw data for original-scale plotting
df = pd.read_csv("TIGPSw2_s.csv", low_memory=False)

# ------------------------------------------------------------
# 1. Construct the three scales
# ------------------------------------------------------------

selective_vars = ["bs23a", "bs23b", "bs23c"]
online_vars = ["bs25a", "bs25b", "bs25c"]
offline_vars = ["bs25d", "bs25e", "bs25f"]

# Special missing values → NaN
for col in selective_vars + online_vars + offline_vars:
    df[col] = df[col].where(df[col] > 0)

df["selective_presentation_score"] = (
    df[selective_vars].mean(axis=1)
    .where(df[selective_vars].notna().sum(axis=1) == 3)
)

df["online_self_image_score"] = (
    df[online_vars].mean(axis=1)
    .where(df[online_vars].notna().sum(axis=1) == 3)
)

df["offline_self_image_score"] = (
    df[offline_vars].mean(axis=1)
    .where(df[offline_vars].notna().sum(axis=1) == 3)
)

# Prepare the same analytic sample as FINAL RQ1
# ------------------------------------------------------------
# Prepare the same analytic sample as FINAL RQ1
# ------------------------------------------------------------

# Convert variables to numeric
for col in ["bs1", "bs3", "HOUWGT", "nschool_id"]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

plot_data = df[
    [
        "selective_presentation_score",
        "online_self_image_score",
        "offline_self_image_score",
        "bs1",
        "bs3",
        "HOUWGT",
        "nschool_id"
    ]
].copy()

# Gender: valid responses = 1, 2
plot_data["bs1"] = plot_data["bs1"].where(
    plot_data["bs1"].isin([1, 2])
)

# Subjective family SES: valid range = 1–10
plot_data["bs3"] = plot_data["bs3"].where(
    plot_data["bs3"].between(1, 10)
)

# Weight must be positive
plot_data["HOUWGT"] = plot_data["HOUWGT"].where(
    plot_data["HOUWGT"] > 0
)

plot_data = plot_data.dropna()

print("Figure 1 analytic N =", len(plot_data))
# ------------------------------------------------------------
# 2. Means and SDs
# ------------------------------------------------------------

selective_mean = plot_data["selective_presentation_score"].mean()
selective_sd = plot_data["selective_presentation_score"].std()

online_mean = plot_data["online_self_image_score"].mean()
online_sd = plot_data["online_self_image_score"].std()

offline_mean = plot_data["offline_self_image_score"].mean()
offline_sd = plot_data["offline_self_image_score"].std()

# ------------------------------------------------------------
# 3. Final standardized coefficients
# ------------------------------------------------------------

online_beta = 0.4477
offline_beta = 0.1949

# ------------------------------------------------------------
# 4. Convert 1–4 selective-presentation scale to z
# ------------------------------------------------------------

x_raw = np.linspace(1, 4, 200)

x_z = (
    x_raw - selective_mean
) / selective_sd

# ------------------------------------------------------------
# 5. Convert predicted standardized outcomes
#    back to original 1–4 scales
#
# We focus on the slope here.
# ------------------------------------------------------------
online_intercept = 0.0201
online_beta = 0.4477

offline_intercept = 0.0483
offline_beta = 0.1949

online_pred_z = (
    online_intercept
    + online_beta * x_z
)

offline_pred_z = (
    offline_intercept
    + offline_beta * x_z
)

online_pred_raw = (
    online_mean
    +online_pred_z*online_sd
)

offline_pred_raw = (
    offline_mean
    + offline_pred_z*offline_sd
)

# ------------------------------------------------------------
# 6. Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    x_raw,
    online_pred_raw,
    linewidth=3,
    color="tab:blue",
    label="Online Positive Self-image"
)

ax.plot(
    x_raw,
    offline_pred_raw,
    linewidth=3,
    linestyle="--",
    color="tab:red",
    label="Offline Positive Self-image"
)

# ------------------------------------------------------------
# Axis
# ------------------------------------------------------------

ax.set_xlim(1, 4)
ax.set_xticks(np.arange(1, 4.01, 0.5))

ax.set_xticklabels([
    "1\nLow",
    "1.5",
    "2",
    "2.5",
    "3",
    "3.5",
    "4\nHigh"
])
ax.set_ylim(1, 4)
ax.set_yticks(np.arange(1, 4.01, 0.5))

ax.set_xlabel(
    "Selective Positive Self-presentation",
    fontsize=12
)

ax.set_ylim(1, 4)

ax.set_ylabel(
    "Model-estimated Positive Self-image",
    fontsize=12
)

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

ax.set_title(
    "Selective Self-presentation and\n"
    "Online vs. Offline Positive Self-image",
    fontsize=14,
    pad=12
)

# ------------------------------------------------------------
# β labels
# ------------------------------------------------------------

ax.text(
    3.05,
    3.25,
    "Online: β = .448",
    color="tab:blue",
    fontsize=11,
    fontweight="bold",
    ha="left",
    va="center",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.85,
        pad=3
    )
)

ax.text(
    3.05,
    2.73,
    "Offline: β = .195",
    color="tab:red",
    fontsize=11,
    fontweight="bold",
    ha="left",
    va="center",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.85,
        pad=2
    )
)

# ------------------------------------------------------------
# Appearance
# ------------------------------------------------------------

ax.legend(
    frameon=False,
    loc="upper left"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linestyle=":",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "figure1_selective_self_presentation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FIGURE 2A–2C
# Online / Offline Positive Self-image → Psychological outcomes
# Same style as Figure 1; one figure per outcome.
#
# Model (same as topic1.py FORMAL ANALYSIS 5C, RQ2):
#   outcome_z ~ online_z + offline_z + female + family_ses_z
#   WLS with HOUWGT, school-clustered SE (nschool_id)
#
# Coefficients are estimated here rather than hard-coded,
# so the figures stay in sync with the data.
# ============================================================

def scale_mean(data, items, valid):
    x = data[items].where(valid(data[items]))
    return x.mean(axis=1).where(x.notna().sum(axis=1) == len(items))


def zscore(x):
    return (x - x.mean()) / x.std()


def fmt_beta(b):
    # APA style: no leading zero, typographic minus sign
    s = f"{abs(b):.3f}".lstrip("0")
    return ("\u2212" if b < 0 else "") + s


# Outcome scales (same rules as topic1.py).
# online/offline self-image scores were already built for Figure 1.
fig2_data = df[[
    "online_self_image_score", "offline_self_image_score",
    "bs1", "bs3", "HOUWGT", "nschool_id"
]].copy()

fig2_data["self_esteem_score"] = scale_mean(
    df, ["bs53a", "bs53b", "bs53c"], lambda x: x > 0
)
fig2_data["depression_score"] = scale_mean(
    df, [f"bs56{c}" for c in "abcdefghijklmn"], lambda x: x >= 0
)
fig2_data["wellbeing_score"] = scale_mean(
    df, [f"bs58{c}" for c in "abcde"], lambda x: x >= 0
)
fig2_data["female"] = (fig2_data["bs1"] == 1).astype(int)
fig2_data["family_ses"] = fig2_data["bs3"].where(fig2_data["bs3"].between(1, 10))

fig2_outcomes = [
    {
        "var": "self_esteem_score",
        "name": "Self-esteem",
        "ylabel": "Model-estimated Self-esteem",
        "ylim": (1, 4),
        "file": "figure2a_selfimage_selfesteem.png",
    },
    {
        "var": "wellbeing_score",
        "name": "Well-being",
        "ylabel": "Model-estimated Well-being (WHO-5)",
        "ylim": (0, 4),
        "file": "figure2b_selfimage_wellbeing.png",
    },
    {
        "var": "depression_score",
        "name": "Depressive Symptoms",
        "ylabel": "Model-estimated Depressive Symptoms",
        "ylim": (0, 4),
        "file": "figure2c_selfimage_depression.png",
    },
]


def fit_figure2(spec):
    """Fit the RQ2 model for one outcome and return predicted lines."""
    y = spec["var"]

    d = fig2_data[[
        y, "online_self_image_score", "offline_self_image_score",
        "female", "family_ses", "HOUWGT", "nschool_id"
    ]].dropna().copy()

    for col in [y, "online_self_image_score",
                "offline_self_image_score", "family_ses"]:
        d[col + "_z"] = zscore(d[col])

    model = smf.wls(
        f"{y}_z ~ online_self_image_score_z + offline_self_image_score_z"
        " + female + family_ses_z",
        data=d,
        weights=d["HOUWGT"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["nschool_id"]})

    b = model.params
    b_on = b["online_self_image_score_z"]
    b_off = b["offline_self_image_score_z"]

    print(f"Figure 2 ({spec['name']}): N = {int(model.nobs)}, "
          f"online β = {b_on:.4f}, offline β = {b_off:.4f}")

    # Predictions on the original 1–4 self-image scale.
    # The other self-image score and family SES are held at their means
    # (z = 0); gender is held at the sample proportion of girls.
    x_raw = np.linspace(1, 4, 200)
    on_mean, on_sd = d["online_self_image_score"].agg(["mean", "std"])
    off_mean, off_sd = d["offline_self_image_score"].agg(["mean", "std"])
    y_mean, y_sd = d[y].agg(["mean", "std"])

    base_z = b["Intercept"] + b["female"] * d["female"].mean()

    return {
        "x": x_raw,
        "online_pred": y_mean + (base_z + b_on * (x_raw - on_mean) / on_sd) * y_sd,
        "offline_pred": y_mean + (base_z + b_off * (x_raw - off_mean) / off_sd) * y_sd,
        "b_on": b_on,
        "b_off": b_off,
    }


def plot_figure2_panel(ax, spec, res, title, label_size=11, legend=True):
    """Draw one outcome panel in Figure 1 style on the given axes."""
    x_raw = res["x"]
    online_pred, offline_pred = res["online_pred"], res["offline_pred"]
    b_on, b_off = res["b_on"], res["b_off"]

    ax.plot(x_raw, online_pred, linewidth=3, color="tab:blue",
            label="Online Positive Self-image")
    ax.plot(x_raw, offline_pred, linewidth=3, linestyle="--",
            color="tab:red", label="Offline Positive Self-image")

    ax.set_xlim(1, 4)
    ax.set_xticks(np.arange(1, 4.01, 0.5))
    ax.set_xticklabels(["1\nLow", "1.5", "2", "2.5", "3", "3.5", "4\nHigh"])

    lo, hi = spec["ylim"]
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + 0.01, 0.5))

    ax.set_xlabel("Positive Self-image", fontsize=12)
    ax.set_ylabel(spec["ylabel"], fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    # β labels: the upper line gets its label above the line,
    # the lower line gets its label below, measured over the
    # x-range the label occupies so text never sits on a line.
    label_x = 3.05 if label_size >= 11 else 2.75
    span = x_raw >= label_x
    on_seg, off_seg = online_pred[span], offline_pred[span]
    online_is_upper = on_seg.mean() >= off_seg.mean()
    pad = 0.07 * (hi - lo)

    labels = [
        (f"Online: β = {fmt_beta(b_on)}", on_seg, online_is_upper, "tab:blue"),
        (f"Offline: β = {fmt_beta(b_off)}", off_seg, not online_is_upper, "tab:red"),
    ]
    positions = {
        t: (seg.max() + pad if up else seg.min() - pad)
        for t, seg, up, _ in labels
    }

    # If the lower label would fall off the bottom of the plot,
    # stack both labels above the upper line instead.
    lower_text = [t for t, _, up, _ in labels if not up][0]
    upper_text = [t for t, _, up, _ in labels if up][0]
    if positions[lower_text] < lo + 0.6 * pad:
        positions[lower_text] = positions[upper_text]
        positions[upper_text] = positions[upper_text] + 1.3 * pad

    for text, seg, upper, color in labels:
        ax.text(label_x, positions[text], text, color=color,
                fontsize=label_size, fontweight="bold",
                ha="left", va="center",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=3))

    if legend:
        ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.25)


# ---------- Figures 2A–2C: one figure per outcome ----------

fig2_results = []
for spec in fig2_outcomes:
    res = fit_figure2(spec)
    fig2_results.append(res)

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_figure2_panel(
        ax, spec, res,
        title=f"Online vs. Offline Positive Self-image\nand {spec['name']}",
    )
    plt.tight_layout()
    plt.savefig(spec["file"], dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# FIGURE 2D
# Combined figure: lines from 2A + 2B + 2C overlaid on one plot.
#
# Colour = outcome; line style = self-image type
# (solid = online, dashed = offline, as in Figure 1).
# All three outcomes are drawn on a shared 0–4 axis, which
# contains every outcome's original range (self-esteem 1–4,
# well-being 0–4, depressive symptoms 0–4).
# ============================================================

outcome_colors = {
    "Self-esteem": "#7b3294",          # purple
    "Well-being": "#1b9e77",           # teal
    "Depressive Symptoms": "#d95f02",  # orange
}

fig, ax = plt.subplots(figsize=(11, 7))

end_labels = []   # (y at x = 4, text, colour)
for spec, res in zip(fig2_outcomes, fig2_results):
    color = outcome_colors[spec["name"]]
    ax.plot(res["x"], res["online_pred"], linewidth=3, color=color)
    ax.plot(res["x"], res["offline_pred"], linewidth=3, linestyle="--",
            color=color)
    end_labels.append((res["online_pred"][-1],
                       f"{spec['name']} – Online: β = {fmt_beta(res['b_on'])}",
                       color))
    end_labels.append((res["offline_pred"][-1],
                       f"{spec['name']} – Offline: β = {fmt_beta(res['b_off'])}",
                       color))

ax.set_xlim(1, 4)
ax.set_xticks(np.arange(1, 4.01, 0.5))
ax.set_xticklabels(["1\nLow", "1.5", "2", "2.5", "3", "3.5", "4\nHigh"])
ax.set_ylim(0, 4)
ax.set_yticks(np.arange(0, 4.01, 0.5))

ax.set_xlabel("Positive Self-image", fontsize=12)
ax.set_ylabel("Model-estimated Outcome Score (original scale)", fontsize=12)
ax.set_title("Online vs. Offline Positive Self-image\n"
             "and Three Psychological Outcomes", fontsize=14, pad=12)

# Labels in the right margin, at the end of each line.
# Sorted by height and pushed apart so they never overlap.
min_gap = 0.17
end_labels.sort(key=lambda t: t[0])
label_y = [t[0] for t in end_labels]
for i in range(1, len(label_y)):
    label_y[i] = max(label_y[i], label_y[i - 1] + min_gap)
overflow = label_y[-1] - 3.95          # keep the top label inside the plot
if overflow > 0:
    label_y = [v - overflow for v in label_y]
    for i in range(len(label_y) - 2, -1, -1):
        label_y[i] = min(label_y[i], label_y[i + 1] - min_gap)

for (y_end, text, color), y_lab in zip(end_labels, label_y):
    ax.annotate(
        text, xy=(4, y_end), xytext=(4.08, y_lab),
        textcoords="data", annotation_clip=False,
        color=color, fontsize=10.5, fontweight="bold",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                        shrinkA=0, shrinkB=2),
    )

# Legend for line style only (outcomes are named by the end labels)
from matplotlib.lines import Line2D
style_handles = [
    Line2D([0], [0], color="dimgray", lw=3, label="Online Positive Self-image"),
    Line2D([0], [0], color="dimgray", lw=3, linestyle="--",
           label="Offline Positive Self-image"),
]
ax.legend(handles=style_handles, frameon=False, loc="upper left")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.25)

fig.text(
    0.01, -0.01,
    "Note. Lines are model-estimated values from weighted (HOUWGT) regressions "
    "with school-clustered SEs; online and offline self-image entered together,\n"
    "controlling for gender and family SES. When one line is drawn, the other "
    "self-image score and family SES are held at their means.\n"
    "Outcomes use different scales (self-esteem 1–4; well-being and depressive "
    "symptoms 0–4), so compare slopes rather than line heights across outcomes.\n"
    "Higher depressive-symptom scores indicate worse mental health. "
    "β = standardized coefficient.",
    fontsize=9, color="dimgray", ha="left", va="top",
)

plt.tight_layout()
plt.savefig("figure2d_selfimage_outcomes_combined.png",
            dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# FIGURE 3A–3B（修改版 RQ3）
# Online / Offline self-image direction → Psychological outcomes
#
# 取代原本的性別調節圖（Gender moderation）。
#   Figure 3A：線上自我形象 > 線下自我形象 的學生，落差越大，心理狀態如何變化
#   Figure 3B：線下自我形象 > 線上自我形象 的學生，落差越大，心理狀態如何變化
#
# Model (same as topic1_final_v2.py RQ3B):
#   sample  = Equal group (gap = 0, anchor) + students in that direction
#   outcome ~ gap_size + self_image_level_z + female + family_ses_z
#   WLS with HOUWGT, school-clustered SE (nschool_id)
#
# Lines = model-estimated values (covariates at weighted sample means),
# bands = 95% CI, grey dots = unadjusted weighted means per gap value
# (n >= 30 only). β = change in standardized outcome per 1-point gap.
# ============================================================

# ------------------------------------------------------------
# 1. Build scales（規則與 topic1_final.py 相同：負數 → 遺漏，題目需全答）
# ------------------------------------------------------------
fig3_items = {
    "online_self_image_score": ["bs25a", "bs25b", "bs25c"],
    "offline_self_image_score": ["bs25d", "bs25e", "bs25f"],
    "self_esteem_score": ["bs53a", "bs53b", "bs53c"],
    "wellbeing_score": [f"bs58{c}" for c in "abcde"],
    "depression_score": [f"bs56{c}" for c in "abcdefghijklmn"],
}

fig3_data = pd.DataFrame(index=df.index)
for score, items in fig3_items.items():
    x = df[items].apply(pd.to_numeric, errors="coerce")
    x = x.where(x >= 0)
    fig3_data[score] = x.mean(axis=1).where(x.notna().sum(axis=1) == len(items))

fig3_bs1 = pd.to_numeric(df["bs1"], errors="coerce")
fig3_data["female"] = fig3_bs1.map({1: 1, 2: 0})
fig3_bs3 = pd.to_numeric(df["bs3"], errors="coerce")
fig3_data["family_ses"] = fig3_bs3.where(fig3_bs3.between(1, 10))
fig3_w = pd.to_numeric(df["HOUWGT"], errors="coerce")
fig3_data["HOUWGT"] = fig3_w.where(fig3_w > 0)
fig3_data["nschool_id"] = df["nschool_id"]

fig3_data = fig3_data.dropna(subset=[
    "online_self_image_score", "offline_self_image_score",
    "female", "family_ses", "HOUWGT", "nschool_id",
])

# Online − Offline gap and direction groups
fig3_data["gap"] = (
    fig3_data["online_self_image_score"]
    - fig3_data["offline_self_image_score"]
).round(4)
fig3_data["gap_group"] = np.select(
    [fig3_data["gap"] > 0, fig3_data["gap"] < 0],
    ["Online > Offline", "Offline > Online"],
    default="Equal",
)
fig3_data["gap_size"] = fig3_data["gap"].abs()
fig3_data["self_image_level"] = (
    fig3_data["online_self_image_score"]
    + fig3_data["offline_self_image_score"]
) / 2

print("\nFigure 3 group sizes:")
print(fig3_data["gap_group"].value_counts())

# ------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------
fig3_outcomes = [
    {"var": "self_esteem_score", "name": "Self-esteem",
     "ylabel": "Self-esteem (1–4)", "ylim": (1, 4), "color": "#7B3294"},
    {"var": "wellbeing_score", "name": "Well-being",
     "ylabel": "Well-being (0–5)", "ylim": (0, 5), "color": "#1B9E77"},
    {"var": "depression_score", "name": "Depressive symptoms",
     "ylabel": "Depressive symptoms (0–5)", "ylim": (0, 5), "color": "#D95F02"},
]

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

FIG3_INK = "#333333"
FIG3_MUTED = "#777777"
FIG3_GRID = "#E6E6E6"
FIG3_DOT = "#9A9A9A"
FIG3_MIN_N_DOT = 30


def fit_weighted_cluster(formula, d):
    return smf.wls(formula, data=d, weights=d["HOUWGT"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["nschool_id"]}
    )


def fmt_p(p):
    return "p < .001" if p < .001 else "p = " + f"{p:.3f}".lstrip("0")


# ------------------------------------------------------------
# 3. Plot — 三個心理結果畫在同一張圖（風格同 Figure 2D）
#    顏色 = 心理結果；線 = 模型估計值；陰影 = 95% CI；
#    小點 = 各落差值的未調整加權平均。
#    三個結果共用 0–5 的 y 軸（涵蓋 self-esteem 1–4、
#    well-being 0–5、depressive symptoms 0–5 的原始範圍）。
# ------------------------------------------------------------
for spec in fig3_specs:
    subset = fig3_data[fig3_data["gap_group"].isin(["Equal", spec["direction"]])]

    fig, ax = plt.subplots(figsize=(11, 7))
    end_labels = []   # (y at x = 3, text, colour)

    for out in fig3_outcomes:
        d = subset.dropna(subset=[out["var"]]).copy()
        d["level_z"] = zscore(d["self_image_level"])
        d["family_ses_z"] = zscore(d["family_ses"])
        d["outcome_z"] = zscore(d[out["var"]])

        rhs = "gap_size + level_z + female + family_ses_z"
        m_raw = fit_weighted_cluster(f"{out['var']} ~ {rhs}", d)   # original scale → line
        m_std = fit_weighted_cluster(f"outcome_z ~ {rhs}", d)      # standardized → β

        # Model-estimated line, covariates at weighted sample means
        grid = pd.DataFrame({
            "gap_size": np.linspace(0, 3, 61),
            "level_z": np.average(d["level_z"], weights=d["HOUWGT"]),
            "female": np.average(d["female"], weights=d["HOUWGT"]),
            "family_ses_z": np.average(d["family_ses_z"], weights=d["HOUWGT"]),
        })
        pred = m_raw.get_prediction(grid).summary_frame(alpha=0.05)

        ax.fill_between(grid["gap_size"], pred["mean_ci_lower"],
                        pred["mean_ci_upper"], color=out["color"],
                        alpha=0.15, linewidth=0)
        ax.plot(grid["gap_size"], pred["mean"], color=out["color"],
                linewidth=3)

        # Unadjusted weighted means at each gap value
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

        b = m_std.params["gap_size"]
        p = m_std.pvalues["gap_size"]
        end_labels.append((
            pred["mean"].iloc[-1],
            f"{out['name']}: β = {fmt_beta(b)}, {fmt_p(p)}",
            out["color"],
        ))

    ax.set_xlim(0, 3)
    ax.set_xticks(np.arange(0, 3.01, 0.5))
    ax.set_xticklabels(["0\n(equal)", "0.5", "1", "1.5", "2", "2.5", "3"])
    ax.set_ylim(0, 5)
    ax.set_yticks(np.arange(0, 5.01, 0.5))

    ax.set_xlabel(spec["xlabel"], fontsize=12)
    ax.set_ylabel("Model-estimated Outcome Score (original scale)", fontsize=12)
    ax.set_title(spec["title"] + "\nand Three Psychological Outcomes",
                 fontsize=14, pad=12)

    # Labels in the right margin at the end of each line,
    # sorted by height and pushed apart so they never overlap.
    min_gap = 0.25
    end_labels.sort(key=lambda t: t[0])
    label_y = [t[0] for t in end_labels]
    for i in range(1, len(label_y)):
        label_y[i] = max(label_y[i], label_y[i - 1] + min_gap)
    overflow = label_y[-1] - 4.95
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

    # Legend for marks only (outcomes are named by the end labels)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    mark_handles = [
        Line2D([0], [0], color="dimgray", lw=3, label="Model-estimated value"),
        Patch(facecolor="dimgray", alpha=0.2, label="95% CI"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="dimgray",
               markeredgecolor="white", alpha=0.6, markersize=7,
               label="Observed weighted mean"),
    ]
    ax.legend(handles=mark_handles, frameon=False, loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.25)

    n_dir = (subset["gap_group"] == spec["direction"]).sum()
    n_eq = (subset["gap_group"] == "Equal").sum()
    fig.text(
        0.01, -0.01,
        f"Note. Sample = students with equal online/offline self-image (gap = 0, n = {n_eq:,}) "
        f"plus students in the {spec['direction']} group (n = {n_dir:,}).\n"
        "Lines are model-estimated values from weighted (HOUWGT) regressions with school-clustered SEs, "
        "controlling for overall self-image level\n"
        "[(online + offline) / 2], gender and family SES (held at sample means). "
        f"Dots = unadjusted weighted means at each gap value (n \u2265 {FIG3_MIN_N_DOT}).\n"
        "Outcomes use different scales (self-esteem 1–4; well-being and depressive symptoms 0–5), "
        "so compare slopes rather than line heights.\n"
        "Higher depressive-symptom scores indicate worse mental health. "
        "β = change in the standardized outcome per 1-point gap.",
        fontsize=9, color="dimgray", ha="left", va="top",
    )

    plt.tight_layout()
    plt.savefig(spec["file"], dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
# FIGURE RQ2
# Selective Presentation × Online/Offline Self-image Groups
# 100% Stacked Bar Chart
# ============================================================

# 1. Prepare RQ2 data
rq2_plot = df[[
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score"
]].dropna().copy()

# Online - Offline
rq2_plot["gap"] = (
    rq2_plot["online_self_image_score"]
    - rq2_plot["offline_self_image_score"]
).round(6)

# 2. Classify self-image groups
rq2_plot["image_group"] = np.select(
    [
        rq2_plot["gap"] > 0,
        rq2_plot["gap"] < 0
    ],
    [
        "Online > Offline",
        "Online < Offline"
    ],
    default="Equal"
)

# 3. Classify Selective levels
rq2_plot["selective_group"] = pd.cut(
    rq2_plot["selective_presentation_score"],
    bins=[0, 2, 3, 4],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# 4. Calculate row percentages
group_order = ["Low", "Medium", "High"]
image_order = [
    "Online > Offline",
    "Equal",
    "Online < Offline"
]

rq2_pct = pd.crosstab(
    rq2_plot["selective_group"],
    rq2_plot["image_group"],
    normalize="index"
).reindex(
    index=group_order,
    columns=image_order,
    fill_value=0
) * 100

print("\nRQ2 Figure N =", len(rq2_plot))
print(rq2_pct.round(2))

# 5. Plot 100% stacked horizontal bars
colors = ["#347EB2", "#8999AA", "#D9965F"]

fig, ax = plt.subplots(figsize=(10, 5.5))

left = np.zeros(len(group_order))

for group, color in zip(image_order, colors):
    values = rq2_pct[group].to_numpy()

    bars = ax.barh(
        group_order,
        values,
        left=left,
        color=color,
        height=0.6,
        label=group,
        edgecolor="white",
        linewidth=1
    )

    # Percentage labels
    for bar, value in zip(bars, values):
        if value >= 5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold"
            )

    left += values

# 6. Axis and appearance
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_xticklabels([f"{x}%" for x in range(0, 101, 20)])

ax.invert_yaxis()  # Low → Medium → High (top to bottom)

ax.set_xlabel("Percentage of Students", fontsize=12)
ax.set_ylabel("Selective Positive Self-presentation", fontsize=12)

ax.set_title(
    "Online–Offline Self-image Profiles\n"
    "by Selective Positive Self-presentation",
    fontsize=14,
    pad=15
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=False
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()

# 7. Save
plt.savefig(
    "figure_rq2_stacked_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()