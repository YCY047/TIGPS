import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# FIGURE 1
# RQ1 Standardized Coefficients
# ============================================================

data = pd.DataFrame({
    "Predictor": [
        "Schoolwork",
        "Extra learning",
        "Gaming",
        "Entertainment",
        "Chat / messaging",
        "Interest browsing",
        "Selective presentation"
    ],

    # Online Positive Self-image
    "Online_beta": [
        -0.0089,
         0.0088,
         0.0510,
        -0.0323,
         0.0773,
         0.0325,
         0.4255
    ],

    "Online_low": [
        -0.0415,   # approximate from SE
        -0.0214,
         0.0214,
        -0.0712,
         0.0476,
        -0.0061,
         0.4004
    ],

    "Online_high": [
         0.0237,
         0.0390,
         0.0805,
         0.0066,
         0.1071,
         0.0711,
         0.4505
    ],

    # Offline Positive Self-image
    "Offline_beta": [
         0.0025,
         0.0100,
        -0.0831,
        -0.0619,
         0.0328,
         0.0198,
         0.1961
    ],

    "Offline_low": [
        -0.0297,
        -0.0184,
        -0.1187,
        -0.1055,
         0.0001,
        -0.0175,
         0.1697
    ],

    "Offline_high": [
         0.0347,
         0.0384,
        -0.0476,
        -0.0184,
         0.0655,
         0.0571,
         0.2226
    ]
})


# Reverse order so Selective presentation appears at top
data = data.iloc[::-1].reset_index(drop=True)

y = np.arange(len(data))

fig, ax = plt.subplots(figsize=(9, 6))


# ------------------------------------------------------------
# Online coefficients
# ------------------------------------------------------------

ax.errorbar(
    data["Online_beta"],
    y + 0.12,
    xerr=[
        data["Online_beta"] - data["Online_low"],
        data["Online_high"] - data["Online_beta"]
    ],
    fmt="o",
    capsize=3,
    label="Online Positive Self-image"
)


# ------------------------------------------------------------
# Offline coefficients
# ------------------------------------------------------------

ax.errorbar(
    data["Offline_beta"],
    y - 0.12,
    xerr=[
        data["Offline_beta"] - data["Offline_low"],
        data["Offline_high"] - data["Offline_beta"]
    ],
    fmt="s",
    capsize=3,
    label="Offline Positive Self-image"
)


# Zero reference line
ax.axvline(
    x=0,
    linestyle="--",
    linewidth=1
)


# Labels
ax.set_yticks(y)
ax.set_yticklabels(data["Predictor"])

ax.set_xlabel("Standardized regression coefficient (β)")
ax.set_ylabel("")

ax.set_title(
    "Digital Use, Selective Self-presentation,\n"
    "and Positive Self-image"
)

ax.legend(frameon=False)

plt.tight_layout()

plt.savefig(
    "figure1_rq1_coefficients.png",
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


# ============================================================
# FIGURE 1
# Selective Positive Self-presentation
# and Online / Offline Positive Self-image
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Final RQ1 focused-model coefficients
# ------------------------------------------------------------

online_intercept = 0.0201
online_beta = 0.4477

offline_intercept = 0.0483
offline_beta = 0.1949


# ------------------------------------------------------------
# X values
# ------------------------------------------------------------

x = np.linspace(-2, 2, 200)

online_pred = online_intercept + online_beta * x
offline_pred = offline_intercept + offline_beta * x


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))


# Online
ax.plot(
    x,
    online_pred,
    linewidth=2.7,
    color="tab:blue",
    label="Online Positive Self-image"
)


# Offline
ax.plot(
    x,
    offline_pred,
    linewidth=2.7,
    linestyle="--",
    color="tab:red",                 # 改成紅色
    label="Offline Positive Self-image"
)


# ------------------------------------------------------------
# Reference lines
# ------------------------------------------------------------

ax.axhline(
    0,
    linestyle=":",
    linewidth=1,
    color="gray",
    alpha=0.5
)

ax.axvline(
    0,
    linestyle=":",
    linewidth=1,
    color="gray",
    alpha=0.5
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel(
    "Selective Positive Self-presentation (SD)",
    fontsize=11
)

ax.set_ylabel(
    "Predicted Positive Self-image (SD)",
    fontsize=11
)

ax.set_title(
    "Selective Self-presentation and\n"
    "Online vs. Offline Positive Self-image",
    fontsize=14,
    pad=12
)


# ------------------------------------------------------------
# X-axis
# ------------------------------------------------------------

ax.set_xticks([-2, -1, 0, 1, 2])

ax.set_xticklabels([
    "−2 SD",
    "−1 SD",
    "Mean",
    "+1 SD",
    "+2 SD"
])


# ------------------------------------------------------------
# Coefficient annotations
# ------------------------------------------------------------

# 放在右側，但不要壓在線上
x_label = 1.25

online_y = online_intercept + online_beta * x_label
offline_y = offline_intercept + offline_beta * x_label


ax.annotate(
    "Online: β = .448",
    xy=(x_label, online_y),
    xytext=(12, 15),                 # 往右、往上移
    textcoords="offset points",
    fontsize=10,
    color="tab:blue",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.8,
        pad=2
    )
)


ax.annotate(
    "Offline: β = .195",
    xy=(x_label, offline_y),
    xytext=(12, -22),                # 往右、往下移
    textcoords="offset points",
    fontsize=10,
    color="tab:red",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.8,
        pad=2
    )
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    frameon=False,
    loc="upper left",
    fontsize=10
)


# ------------------------------------------------------------
# Appearance
# ------------------------------------------------------------

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.savefig(
    "figure1_selective_self_presentation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()