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