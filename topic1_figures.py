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
# FIGURE 3A–3B
# Gender moderation:
# Selective presentation → Online (3A) / Offline (3B) self-image
# Same style as Figure 1; one figure per self-image type.
#
# Model (same as topic1.py RQ3A, weighted + clustered):
#   self_image_z ~ selective_z * female + family_ses_z
#   WLS with HOUWGT, school-clustered SE (nschool_id)
#
# Lines show boys (female = 0) and girls (female = 1) with family
# SES held at its mean. β labels are the simple slopes per gender.
# ============================================================

fig3_data = df[[
    "selective_presentation_score",
    "online_self_image_score", "offline_self_image_score",
    "bs1", "bs3", "HOUWGT", "nschool_id"
]].copy()
fig3_data["female"] = (fig3_data["bs1"] == 1).astype(int)
fig3_data["family_ses"] = fig3_data["bs3"].where(fig3_data["bs3"].between(1, 10))
fig3_data = fig3_data.drop(columns=["bs1", "bs3"]).dropna()

# Standardize within the shared analytic sample (as in topic1.py RQ3A)
for col in ["selective_presentation_score", "online_self_image_score",
            "offline_self_image_score", "family_ses"]:
    fig3_data[col + "_z"] = zscore(fig3_data[col])

print("Figure 3 analytic N =", len(fig3_data))


def fmt_p(p):
    return "p < .001" if p < .001 else "p = " + f"{p:.3f}".lstrip("0")


fig3_specs = [
    {
        "var": "online_self_image_score",
        "kind": "Online",
        # Online palette (Figure 1 blue): boys darker, girls lighter
        "colors": {"Male": "#1f4e79", "Female": "#5b9bd5"},
        "file": "figure3a_gender_moderation_online.png",
    },
    {
        "var": "offline_self_image_score",
        "kind": "Offline",
        # Offline palette (Figure 1 red): boys darker, girls lighter
        "colors": {"Male": "#9e1b1b", "Female": "#e8726b"},
        "file": "figure3b_gender_moderation_offline.png",
    },
]


def draw_figure3(spec):
    y = spec["var"]
    d = fig3_data

    model = smf.wls(
        f"{y}_z ~ selective_presentation_score_z * female + family_ses_z",
        data=d,
        weights=d["HOUWGT"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["nschool_id"]})

    b = model.params
    inter = "selective_presentation_score_z:female"
    slope_m = b["selective_presentation_score_z"]
    slope_f = slope_m + b[inter]
    p_inter = model.pvalues[inter]

    print(f"Figure 3 ({spec['kind']}): male slope = {slope_m:.4f}, "
          f"female slope = {slope_f:.4f}, "
          f"interaction = {b[inter]:.4f} ({fmt_p(p_inter)})")

    # Predictions on the original 1–4 scales, family SES at mean (z = 0)
    x_raw = np.linspace(1, 4, 200)
    sel_mean, sel_sd = d["selective_presentation_score"].agg(["mean", "std"])
    y_mean, y_sd = d[y].agg(["mean", "std"])
    x_z = (x_raw - sel_mean) / sel_sd

    preds = {
        "Male": y_mean + (b["Intercept"] + slope_m * x_z) * y_sd,
        "Female": y_mean + (b["Intercept"] + b["female"] + slope_f * x_z) * y_sd,
    }
    slopes = {"Male": slope_m, "Female": slope_f}
    styles = {"Male": "-", "Female": "--"}

    # --------------------------------------------------------
    # Plot (Figure 1 style)
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    for g in ["Male", "Female"]:
        ax.plot(x_raw, preds[g], linewidth=3, linestyle=styles[g],
                color=spec["colors"][g], label=g)

    ax.set_xlim(1, 4)
    ax.set_xticks(np.arange(1, 4.01, 0.5))
    ax.set_xticklabels(["1\nLow", "1.5", "2", "2.5", "3", "3.5", "4\nHigh"])
    ax.set_ylim(1, 4)
    ax.set_yticks(np.arange(1, 4.01, 0.5))

    ax.set_xlabel("Selective Positive Self-presentation", fontsize=12)
    ax.set_ylabel(f"Model-estimated {spec['kind']} Positive Self-image",
                  fontsize=12)
    ax.set_title(f"Selective Self-presentation and {spec['kind']} "
                 f"Positive Self-image:\nBoys vs. Girls",
                 fontsize=14, pad=12)

    # Simple-slope labels: upper line labelled above, lower line below,
    # stacked above if the lower label would leave the plot.
    span = x_raw >= 3.05
    segs = {g: preds[g][span] for g in preds}
    upper = max(segs, key=lambda g: segs[g].mean())
    lower = "Female" if upper == "Male" else "Male"
    pad = 0.07 * 3
    pos = {upper: segs[upper].max() + pad, lower: segs[lower].min() - pad}
    if abs(pos[upper] - pos[lower]) < 1.3 * pad:
        pos[upper] = pos[lower] + 1.3 * pad
    if pos[lower] < 1 + 0.6 * pad:
        pos[lower] = segs[upper].max() + pad
        pos[upper] = pos[lower] + 1.3 * pad

    for g in ["Male", "Female"]:
        ax.text(3.05, pos[g], f"{g}: β = {fmt_beta(slopes[g])}",
                color=spec["colors"][g], fontsize=11, fontweight="bold",
                ha="left", va="center",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=3))

    # Interaction test, bottom right
    ax.text(3.95, 1.12,
            f"Gender × Selective presentation: "
            f"β = {fmt_beta(b[inter])}, {fmt_p(p_inter)}",
            fontsize=10, color="dimgray", ha="right", va="center")

    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.25)

    plt.tight_layout()
    plt.savefig(spec["file"], dpi=300, bbox_inches="tight")
    plt.show()


for spec in fig3_specs:
    draw_figure3(spec)