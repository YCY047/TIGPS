import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
# FIGURE 2
# Online vs Offline Positive Self-image
# and Psychological Outcomes
# ============================================================

fig2 = pd.DataFrame({
    "Outcome": [
        "Self-esteem",
        "Well-being",
        "Depressive symptoms"
    ],

    "Online_beta": [
        -0.1360,
        -0.1077,
         0.2127
    ],

    "Online_low": [
        -0.1590,
        -0.1274,
         0.1901
    ],

    "Online_high": [
        -0.1131,
        -0.0880,
         0.2352
    ],

    "Offline_beta": [
         0.4855,
         0.3842,
        -0.2683
    ],

    "Offline_low": [
         0.4589,
         0.3603,
        -0.2954
    ],

    "Offline_high": [
         0.5121,
         0.4081,
        -0.2411
    ]
})

# Reverse so Self-esteem is on top
fig2 = fig2.iloc[::-1].reset_index(drop=True)

y = np.arange(len(fig2))

fig, ax = plt.subplots(figsize=(9, 4.5))

# Online
ax.errorbar(
    fig2["Online_beta"],
    y + 0.10,
    xerr=[
        fig2["Online_beta"] - fig2["Online_low"],
        fig2["Online_high"] - fig2["Online_beta"]
    ],
    fmt="o",
    capsize=4,
    label="Online Positive Self-image"
)

# Offline
ax.errorbar(
    fig2["Offline_beta"],
    y - 0.10,
    xerr=[
        fig2["Offline_beta"] - fig2["Offline_low"],
        fig2["Offline_high"] - fig2["Offline_beta"]
    ],
    fmt="s",
    capsize=4,
    label="Offline Positive Self-image"
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_yticks(y)
ax.set_yticklabels(fig2["Outcome"])

ax.set_xlabel(
    "Standardized regression coefficient (β)"
)

ax.set_title(
    "Online and Offline Positive Self-image\n"
    "Show Distinct Psychological Associations"
)

ax.legend(
    frameon=False,
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "figure2_selfimage_psychological_outcomes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FIGURE 3
# Gender moderation:
# Selective presentation → Online / Offline self-image
# ============================================================

x = np.linspace(-2, 2, 100)

# Weighted + clustered RQ3A coefficients
# Predictions shown at family_ses_z = 0

# ONLINE
online_intercept = 0.0180
online_selective = 0.4403
online_female = -0.0587
online_interaction = 0.0164

online_male = (
    online_intercept
    + online_selective * x
)

online_female_pred = (
    online_intercept
    + online_female
    + (online_selective + online_interaction) * x
)


# OFFLINE
offline_intercept = 0.0674
offline_selective = 0.2613
offline_female = -0.1287
offline_interaction = -0.1470

offline_male = (
    offline_intercept
    + offline_selective * x
)

offline_female_pred = (
    offline_intercept
    + offline_female
    + (offline_selective + offline_interaction) * x
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

# Online
ax.plot(
    x,
    online_male,
    linewidth=2,
    label="Online — Male"
)

ax.plot(
    x,
    online_female_pred,
    linewidth=2,
    linestyle="--",
    label="Online — Female"
)

# Offline
ax.plot(
    x,
    offline_male,
    linewidth=2,
    label="Offline — Male"
)

ax.plot(
    x,
    offline_female_pred,
    linewidth=2,
    linestyle="--",
    label="Offline — Female"
)

ax.axhline(
    0,
    linewidth=0.8,
    linestyle=":"
)

ax.axvline(
    0,
    linewidth=0.8,
    linestyle=":"
)

ax.set_xlabel(
    "Selective Positive Self-presentation (SD)"
)

ax.set_ylabel(
    "Predicted Positive Self-image (SD)"
)

ax.set_title(
    "Gender Moderates the Association Between\n"
    "Selective Self-presentation and Offline Self-image"
)

ax.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "figure3_gender_moderation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()