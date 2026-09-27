import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv("TIGPSw2_s.csv", low_memory=False)

# Internet-use activities
internet_vars = [
    "as35a", "as35b", "as35c",
    "as35d", "as35e", "as35f"
]

# Social-media behavior
social_media_vars = [
    "bs23a", "bs23b", "bs23c",
    "bs23d", "bs23e", "bs23f",
    "bs23g", "bs23h", "bs23i"
]

# Self-image
self_image_vars = [
    "bs25a", "bs25b", "bs25c",
    "bs25d", "bs25e", "bs25f"
]

# General self-esteem
self_esteem_vars = [
    "bs53a", "bs53b", "bs53c"
]

# Body image
body_vars = [
    "bs62", "bs63", "bs64"
]

# Gender
background_vars = [
    "bs1",
    "bs3",
    "HOUWGT",
    "nschool_id"
]

all_vars = (
    internet_vars
    + social_media_vars
    + self_image_vars
    + self_esteem_vars
    + body_vars
    + background_vars
)

print("=== Missing Columns ===")
print([x for x in all_vars if x not in df.columns])

existing_vars = [x for x in all_vars if x in df.columns]
topic1 = df[existing_vars].copy()

print("\nShape:", topic1.shape)

print("\n=== Raw Value Counts ===")
for var in existing_vars:
    print(f"\n--- {var} ---")
    print(topic1[var].value_counts(dropna=False).sort_index())

# ==========================================
# Step 2A: Find actual internet-use variables
# ==========================================

print("\n=== Columns around bs20 / bs21 / bs22 ===")

for col in df.columns:
    if (
        col.startswith("bs20")
        or col.startswith("bs21")
        or col.startswith("bs22")
        or col.startswith("as35")
    ):
        print(col)

# ==========================================
# Step 2B: Clean questionnaire variables
# ==========================================

questionnaire_vars = (
    social_media_vars
    + self_image_vars
    + self_esteem_vars
)

topic1[questionnaire_vars] = (
    topic1[questionnaire_vars]
    .replace(-9, np.nan)
)

print("\n=== Missing after cleaning ===")
print(
    topic1[questionnaire_vars]
    .isna()
    .sum()
)


# ==========================================
# Step 2C: Test social-media constructs
# ==========================================

def cronbach_alpha(data):
    data = data.dropna()

    k = data.shape[1]
    item_variances = data.var(axis=0, ddof=1)
    total_score = data.sum(axis=1)
    total_variance = total_score.var(ddof=1)

    alpha = (k / (k - 1)) * (
        1 - item_variances.sum() / total_variance
    )

    return alpha


def item_total_correlation(data):
    data = data.dropna()

    results = {}

    for item in data.columns:
        other_items = data.drop(columns=item)
        other_total = other_items.sum(axis=1)

        results[item] = data[item].corr(other_total)

    return pd.Series(results)


selective_presentation_vars = [
    "bs23a",
    "bs23b",
    "bs23c"
]

print("\n=== Selective Self-presentation ===")

print(
    "Cronbach alpha:",
    round(
        cronbach_alpha(
            topic1[selective_presentation_vars]
        ),
        3
    )
)

print("\nItem-total correlations:")

print(
    item_total_correlation(
        topic1[selective_presentation_vars]
    )
    .sort_values()
    .round(3)
)

print("\nCorrelation matrix:")

print(
    topic1[selective_presentation_vars]
    .corr()
    .round(3)
)

online_image_vars = [
    "bs25a",
    "bs25b",
    "bs25c"
]

offline_image_vars = [
    "bs25d",
    "bs25e",
    "bs25f"
]

print("\n=== Online Self-image ===")
print(
    "Cronbach alpha:",
    round(
        cronbach_alpha(
            topic1[online_image_vars]
        ),
        3
    )
)

print(
    item_total_correlation(
        topic1[online_image_vars]
    )
    .sort_values()
    .round(3)
)

print("\n=== Offline Self-image ===")
print(
    "Cronbach alpha:",
    round(
        cronbach_alpha(
            topic1[offline_image_vars]
        ),
        3
    )
)

print(
    item_total_correlation(
        topic1[offline_image_vars]
    )
    .sort_values()
    .round(3)
)


# ============================================================
# Step 3. Create Core Scores
# ============================================================

selective_vars = [
    "bs23a", "bs23b", "bs23c"
]

online_image_vars = [
    "bs25a", "bs25b", "bs25c"
]

offline_image_vars = [
    "bs25d", "bs25e", "bs25f"
]


# 三題都必須有效
topic1["selective_presentation_score"] = (
    topic1[selective_vars]
    .mean(axis=1)
    .where(topic1[selective_vars].notna().sum(axis=1) == 3)
)

topic1["online_self_image_score"] = (
    topic1[online_image_vars]
    .mean(axis=1)
    .where(topic1[online_image_vars].notna().sum(axis=1) == 3)
)

topic1["offline_self_image_score"] = (
    topic1[offline_image_vars]
    .mean(axis=1)
    .where(topic1[offline_image_vars].notna().sum(axis=1) == 3)
)


# Online - Offline discrepancy
topic1["self_image_gap"] = (
    topic1["online_self_image_score"]
    - topic1["offline_self_image_score"]
)


print("\n=== Core Score Descriptive Statistics ===")

print(
    topic1[
        [
            "selective_presentation_score",
            "online_self_image_score",
            "offline_self_image_score",
            "self_image_gap"
        ]
    ]
    .describe()
    .round(3)
)


print("\n=== Correlation Matrix ===")

print(
    topic1[
        [
            "selective_presentation_score",
            "online_self_image_score",
            "offline_self_image_score",
            "self_image_gap"
        ]
    ]
    .corr()
    .round(3)
)

internet_vars = [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f"
]
print("\n=== Internet Use Raw Distribution ===")

for var in internet_vars:
    print(f"\n--- {var} ---")
    print(
        df[var]
        .value_counts(dropna=False)
        .sort_index()
    )

# ==========================================
# Step 4. Internet Use vs Self-image
# ==========================================

for var in internet_vars:
    topic1[var] = df[var].where(df[var] > 0)


analysis_vars = (
    internet_vars
    + [
        "selective_presentation_score",
        "online_self_image_score",
        "offline_self_image_score",
        "self_image_gap"
    ]
)

spearman_corr = (
    topic1[analysis_vars]
    .corr(method="spearman")
)

print("\n=== Internet Use vs Self-image: Spearman Correlations ===")

print(
    spearman_corr.loc[
        internet_vars,
        [
            "selective_presentation_score",
            "online_self_image_score",
            "offline_self_image_score",
            "self_image_gap"
        ]
    ]
    .round(3)
    .to_string()
)

# ============================================================
# Step 5A. Gender Descriptive Statistics
# ============================================================

topic1["female"] = (topic1["bs1"] == 1).astype(int)

gender_summary = (
    topic1
    .groupby("bs1")[
        [
            "selective_presentation_score",
            "online_self_image_score",
            "offline_self_image_score",
            "self_image_gap"
        ]
    ]
    .agg(["mean", "std", "count"])
    .round(3)
)

print("\n=== Gender Descriptive Statistics ===")
print(gender_summary.to_string())

topic1["selective_z"] = (
    topic1["selective_presentation_score"]
    - topic1["selective_presentation_score"].mean()
) / topic1["selective_presentation_score"].std()

import statsmodels.formula.api as smf

# Online self-image
model_online = smf.ols(
    "online_self_image_score ~ selective_z * female",
    data=topic1
).fit()

# Offline self-image
model_offline = smf.ols(
    "offline_self_image_score ~ selective_z * female",
    data=topic1
).fit()

# Online - Offline gap
model_gap = smf.ols(
    "self_image_gap ~ selective_z * female",
    data=topic1
).fit()


print("\n========== Online Self-image ==========")
print(
    model_online.summary2().tables[1].round(4)
)

print("\n========== Offline Self-image ==========")
print(
    model_offline.summary2().tables[1].round(4)
)

print("\n========== Self-image Gap ==========")
print(
    model_gap.summary2().tables[1].round(4)
)

# ============================================================
# Step 6A. Clean Height / Weight and Calculate BMI
# ============================================================

topic1["height_cm"] = df["bs62"].where(df["bs62"] > 0)
topic1["weight_kg"] = df["bs63"].where(df["bs63"] > 0)

topic1["bmi"] = (
    topic1["weight_kg"]
    / (topic1["height_cm"] / 100) ** 2
)

# bs64: 1 = thinnest, 9 = largest body figure
topic1["body_perception"] = df["bs64"].where(
    df["bs64"].between(1, 9)
)

print("\n=== Height ===")
print(topic1["height_cm"].describe(percentiles=[.01, .05, .95, .99]))

print("\n=== Weight ===")
print(topic1["weight_kg"].describe(percentiles=[.01, .05, .95, .99]))

print("\n=== BMI ===")
print(topic1["bmi"].describe(percentiles=[.01, .05, .95, .99]))

print("\n=== Subjective Body Perception ===")
print(
    topic1["body_perception"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\n=== Suspicious Height / Weight Cases ===")

suspicious = topic1[
    (topic1["height_cm"] < 130) |
    (topic1["height_cm"] > 200) |
    (topic1["weight_kg"] < 25) |
    (topic1["weight_kg"] > 130) |
    (topic1["bmi"] < 10) |
    (topic1["bmi"] > 50)
][
    ["height_cm", "weight_kg", "bmi", "body_perception", "bs1"]
]

print(suspicious.sort_values("bmi").to_string())
print("\nNumber of suspicious cases:", len(suspicious))


body_data = topic1[
    topic1["height_cm"].between(130, 200)
    & topic1["weight_kg"].between(25, 130)
    & topic1["bmi"].between(10, 50)
].copy()

print("\nBody analysis N =", len(body_data))

print(
    body_data[
        ["height_cm", "weight_kg", "bmi", "body_perception"]
    ].describe().round(3)
)

body_data["bmi_z"] = (
    body_data["bmi"] - body_data["bmi"].mean()
) / body_data["bmi"].std()

body_data["selective_z"] = (
    body_data["selective_presentation_score"]
    - body_data["selective_presentation_score"].mean()
) / body_data["selective_presentation_score"].std()

body_data["female"] = (body_data["bs1"] == 1).astype(int)

model_body1 = smf.ols(
    "body_perception ~ bmi_z + female",
    data=body_data
).fit()

print("\n========== Body Model 1 ==========")
print(model_body1.summary2().tables[1].round(4))
print("R-squared:", round(model_body1.rsquared, 3))

model_body2 = smf.ols(
    "body_perception ~ bmi_z + selective_z + female",
    data=body_data
).fit()

print("\n========== Body Model 2 ==========")
print(model_body2.summary2().tables[1].round(4))
print("R-squared:", round(model_body2.rsquared, 3))

model_body3 = smf.ols(
    "body_perception ~ bmi_z + selective_z * female",
    data=body_data
).fit()

print("\n========== Body Model 3 ==========")
print(model_body3.summary2().tables[1].round(4))
print("R-squared:", round(model_body3.rsquared, 3))


# ============================================================
# Step 7A. General Self-esteem
# ============================================================

self_esteem_vars = [
    "bs53a", "bs53b", "bs53c"
]

print("\n=== General Self-esteem ===")

print(
    "Cronbach alpha:",
    round(
        cronbach_alpha(
            topic1[self_esteem_vars]
        ),
        3
    )
)

print("\nItem-total correlations:")

print(
    item_total_correlation(
        topic1[self_esteem_vars]
    )
    .sort_values()
    .round(3)
)

topic1["self_esteem_score"] = (
    topic1[self_esteem_vars]
    .mean(axis=1)
    .where(topic1[self_esteem_vars].notna().sum(axis=1) == 3)
)

print("\n=== Self-esteem Descriptive ===")
print(topic1["self_esteem_score"].describe().round(3))

depression_vars = [
    "bs56a", "bs56b", "bs56c", "bs56d",
    "bs56e", "bs56f", "bs56g", "bs56h",
    "bs56i", "bs56j", "bs56k", "bs56l",
    "bs56m", "bs56n"
]

wellbeing_vars = [
    "bs58a", "bs58b", "bs58c",
    "bs58d", "bs58e"
]

# Clean special codes
for var in depression_vars + wellbeing_vars:
    topic1[var] = df[var].where(df[var] >= 0)

topic1["depression_score"] = (
    topic1[depression_vars]
    .mean(axis=1)
    .where(topic1[depression_vars].notna().sum(axis=1) == len(depression_vars))
)

topic1["wellbeing_score"] = (
    topic1[wellbeing_vars]
    .mean(axis=1)
    .where(topic1[wellbeing_vars].notna().sum(axis=1) == len(wellbeing_vars))
)

psych_vars = [
    "online_self_image_score",
    "offline_self_image_score",
    "self_image_gap",
    "self_esteem_score",
    "wellbeing_score",
    "depression_score"
]

print("\n=== Self-image × Psychological Outcomes ===")

print(
    topic1[psych_vars]
    .corr()
    .round(3)
    .to_string()
)


topic1["absolute_gap"] = topic1["self_image_gap"].abs()
print("\n=== Directional Gap vs Absolute Gap ===")

gap_corr = topic1[
    [
        "self_image_gap",
        "absolute_gap",
        "self_esteem_score",
        "wellbeing_score",
        "depression_score"
    ]
].corr()

print(gap_corr.round(3).to_string())

# ============================================================
# Step 8. Online vs Offline Self-image
# Psychological Outcomes
# ============================================================

import statsmodels.formula.api as smf

# Gender coding
# bs1: 1 = Female, 2 = Male
topic1["female"] = (topic1["bs1"] == 1).astype(int)

# ------------------------------------------------------------
# Model 1: Self-esteem
# ------------------------------------------------------------

model_selfesteem = smf.ols(
    "self_esteem_score ~ online_self_image_score + offline_self_image_score + female",
    data=topic1
).fit()

print("\n========== Self-esteem Model ==========")
print(
    model_selfesteem.summary2().tables[1].round(4)
)
print("R-squared:", round(model_selfesteem.rsquared, 3))


# ------------------------------------------------------------
# Model 2: Positive Well-being
# ------------------------------------------------------------

model_wellbeing = smf.ols(
    "wellbeing_score ~ online_self_image_score + offline_self_image_score + female",
    data=topic1
).fit()

print("\n========== Well-being Model ==========")
print(
    model_wellbeing.summary2().tables[1].round(4)
)
print("R-squared:", round(model_wellbeing.rsquared, 3))


# ------------------------------------------------------------
# Model 3: Depressive Symptoms
# ------------------------------------------------------------

model_depression = smf.ols(
    "depression_score ~ online_self_image_score + offline_self_image_score + female",
    data=topic1
).fit()

print("\n========== Depression Model ==========")
print(
    model_depression.summary2().tables[1].round(4)
)
print("R-squared:", round(model_depression.rsquared, 3))

# ============================================================
# Step 9. Hierarchical Regression
# Add Selective Self-presentation
# ============================================================

outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depression": "depression_score"
}

for name, outcome in outcomes.items():

    # Model A: Self-image + Gender
    model_a = smf.ols(
        f"{outcome} ~ online_self_image_score "
        f"+ offline_self_image_score + female",
        data=topic1
    ).fit()

    # Model B: Add Selective Self-presentation
    model_b = smf.ols(
        f"{outcome} ~ online_self_image_score "
        f"+ offline_self_image_score "
        f"+ selective_presentation_score "
        f"+ female",
        data=topic1
    ).fit()

    print(f"\n{'='*60}")
    print(name)
    print("="*60)

    print("\n--- Model A: Self-image + Gender ---")
    print(model_a.summary2().tables[1].round(4))
    print("R-squared:", round(model_a.rsquared, 4))

    print("\n--- Model B: + Selective Presentation ---")
    print(model_b.summary2().tables[1].round(4))
    print("R-squared:", round(model_b.rsquared, 4))

    print(
        "Delta R-squared:",
        round(model_b.rsquared - model_a.rsquared, 4)
    )

    # ============================================================
# Step 10. Internet-use activities vs Selective Presentation
# ============================================================

# 六種網路活動
internet_vars = [
    "bs21a",  # 學校功課
    "bs21b",  # 課外學習
    "bs21c",  # 玩遊戲
    "bs21d",  # 影音／音樂／迷因／漫畫
    "bs21e",  # 聊天／傳訊息
    "bs21f"   # 瀏覽有興趣的資訊
]

# Step 10 使用的變項
step10_vars = (
    internet_vars
    + [
        "selective_presentation_score",
        "online_self_image_score",
        "offline_self_image_score",
        "female"
    ]
)

step10 = topic1[step10_vars].copy()

# bs21a-f 的特殊遺漏碼轉成 NaN
for col in internet_vars:
    step10[col] = step10[col].where(step10[col] > 0)

print("\nStep 10 sample missingness:")
print(step10.isna().sum())

# ============================================================
# Standardization
# ============================================================

continuous_vars = (
    internet_vars
    + [
        "selective_presentation_score",
        "online_self_image_score",
        "offline_self_image_score"
    ]
)

for col in continuous_vars:
    step10[col + "_z"] = (
        step10[col] - step10[col].mean()
    ) / step10[col].std()

print("\nStandardized variables created.")

# ============================================================
# Model 10A: Online Self-image
# ============================================================

model_online = smf.ols(
    """
    online_self_image_score_z ~
    bs21a_z +
    bs21b_z +
    bs21c_z +
    bs21d_z +
    bs21e_z +
    bs21f_z +
    selective_presentation_score_z +
    female
    """,
    data=step10
).fit()

print("\n" + "="*60)
print("Step 10A: Predicting Online Self-image")
print("="*60)

print(model_online.summary2().tables[1].round(4))
print("N =", int(model_online.nobs))
print("R-squared =", round(model_online.rsquared, 4))
print("Adjusted R-squared =", round(model_online.rsquared_adj, 4))

# ============================================================
# Model 10B: Offline Self-image
# ============================================================

model_offline = smf.ols(
    """
    offline_self_image_score_z ~
    bs21a_z +
    bs21b_z +
    bs21c_z +
    bs21d_z +
    bs21e_z +
    bs21f_z +
    selective_presentation_score_z +
    female
    """,
    data=step10
).fit()

print("\n" + "="*60)
print("Step 10B: Predicting Offline Self-image")
print("="*60)

print(model_offline.summary2().tables[1].round(4))
print("N =", int(model_offline.nobs))
print("R-squared =", round(model_offline.rsquared, 4))
print("Adjusted R-squared =", round(model_offline.rsquared_adj, 4))

# ============================================================
# Step 11. Gender moderation:
# Online / Offline Self-image × Gender → Psychological outcomes
# ============================================================

psych_outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depression": "depression_score"
}

for name, outcome in psych_outcomes.items():

    # 每個 outcome 建立自己的 complete-case sample
    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female"
    ]

    d = topic1[cols].dropna().copy()

    # Standardize Online / Offline self-image
    d["online_z"] = (
        d["online_self_image_score"]
        - d["online_self_image_score"].mean()
    ) / d["online_self_image_score"].std()

    d["offline_z"] = (
        d["offline_self_image_score"]
        - d["offline_self_image_score"].mean()
    ) / d["offline_self_image_score"].std()

    # --------------------------------------------------------
    # Model A: Main effects only
    # --------------------------------------------------------
    model_main = smf.ols(
        f"{outcome} ~ online_z + offline_z + female",
        data=d
    ).fit()

    # --------------------------------------------------------
    # Model B: Add gender interactions
    # --------------------------------------------------------
    model_interaction = smf.ols(
        f"{outcome} ~ online_z * female + offline_z * female",
        data=d
    ).fit()

    print("\n" + "="*70)
    print(name)
    print("="*70)

    print("\n--- Main-effects model ---")
    print(model_main.summary2().tables[1].round(4))
    print("R-squared =", round(model_main.rsquared, 4))

    print("\n--- Gender-interaction model ---")
    print(model_interaction.summary2().tables[1].round(4))
    print("R-squared =", round(model_interaction.rsquared, 4))

    print(
        "Delta R-squared =",
        round(model_interaction.rsquared - model_main.rsquared, 4)
    )

    # ============================================================
# Step 12. Online vs Offline Self-image coefficient comparison
# ============================================================

psych_outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depression": "depression_score"
}

for name, outcome in psych_outcomes.items():

    # 建立 complete-case sample
    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female"
    ]

    d = topic1[cols].dropna().copy()

    # Standardize Online / Offline Self-image
    # 讓兩個 coefficient 在相同尺度下比較
    d["online_z"] = (
        d["online_self_image_score"]
        - d["online_self_image_score"].mean()
    ) / d["online_self_image_score"].std()

    d["offline_z"] = (
        d["offline_self_image_score"]
        - d["offline_self_image_score"].mean()
    ) / d["offline_self_image_score"].std()

    # 基本模型：控制 Gender
    model = smf.ols(
        f"{outcome} ~ online_z + offline_z + female",
        data=d
    ).fit()

    # H0: beta_online = beta_offline
    test = model.t_test("online_z - offline_z = 0")

    print("\n" + "="*70)
    print(name)
    print("="*70)

    print("\nRegression coefficients:")
    print(model.summary2().tables[1].round(4))

    print("\nOnline vs Offline coefficient equality test:")
    print(test)

    print("\nN =", int(model.nobs))
    print("R-squared =", round(model.rsquared, 4))

    # ============================================================
# FORMAL ANALYSIS 3
# Primary Regression Models
# ============================================================

# ------------------------------------------------------------
# 3.1 RQ1:
# Digital behavior -> Online / Offline Positive Self-image
# ------------------------------------------------------------

rq1_vars = [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f",
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score"
]

rq1 = topic1[rq1_vars].copy()

# 特殊值處理
for col in ["bs21a", "bs21b", "bs21c", "bs21d", "bs21e", "bs21f"]:
    rq1[col] = rq1[col].where(rq1[col] > 0)

# 同一 complete-case sample
rq1 = rq1.dropna()

# Standardize predictors + outcomes
standardize_vars = rq1_vars

for col in standardize_vars:
    rq1[col + "_z"] = (
        rq1[col] - rq1[col].mean()
    ) / rq1[col].std()

# Online model
rq1_online = smf.ols(
    """
    online_self_image_score_z ~
    bs21a_z + bs21b_z + bs21c_z +
    bs21d_z + bs21e_z + bs21f_z +
    selective_presentation_score_z
    """,
    data=rq1
).fit()

# Offline model
rq1_offline = smf.ols(
    """
    offline_self_image_score_z ~
    bs21a_z + bs21b_z + bs21c_z +
    bs21d_z + bs21e_z + bs21f_z +
    selective_presentation_score_z
    """,
    data=rq1
).fit()

print("\n" + "="*70)
print("FORMAL RQ1A: Online Positive Self-image")
print("="*70)
print(rq1_online.summary2().tables[1].round(4))
print("N =", int(rq1_online.nobs))
print("R² =", round(rq1_online.rsquared, 4))
print("Adjusted R² =", round(rq1_online.rsquared_adj, 4))

print("\n" + "="*70)
print("FORMAL RQ1B: Offline Positive Self-image")
print("="*70)
print(rq1_offline.summary2().tables[1].round(4))
print("N =", int(rq1_offline.nobs))
print("R² =", round(rq1_offline.rsquared, 4))
print("Adjusted R² =", round(rq1_offline.rsquared_adj, 4))

# ------------------------------------------------------------
# 3.2 RQ2:
# Online / Offline Self-image -> Psychological outcomes
# ------------------------------------------------------------

psych_outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depression": "depression_score"
}

rq2_results = {}

for name, outcome in psych_outcomes.items():

    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score"
    ]

    d = topic1[cols].dropna().copy()

    # Standardize all variables
    for col in cols:
        d[col + "_z"] = (
            d[col] - d[col].mean()
        ) / d[col].std()

    model = smf.ols(
        f"""
        {outcome}_z ~
        online_self_image_score_z +
        offline_self_image_score_z
        """,
        data=d
    ).fit()

    rq2_results[name] = model

    # Direct coefficient equality test
    contrast = model.t_test(
        "online_self_image_score_z - "
        "offline_self_image_score_z = 0"
    )

    print("\n" + "="*70)
    print(f"FORMAL RQ2: {name}")
    print("="*70)

    print(model.summary2().tables[1].round(4))

    print("\nOnline vs Offline coefficient test:")
    print(contrast)

    print("N =", int(model.nobs))
    print("R² =", round(model.rsquared, 4))
    print("Adjusted R² =", round(model.rsquared_adj, 4))

    # ------------------------------------------------------------
# 3.3 RQ3A:
# Selective Presentation × Gender -> Self-image
# ------------------------------------------------------------

rq3a_vars = [
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "female"
]

rq3a = topic1[rq3a_vars].dropna().copy()

for col in [
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score"
]:
    rq3a[col + "_z"] = (
        rq3a[col] - rq3a[col].mean()
    ) / rq3a[col].std()

rq3_online = smf.ols(
    """
    online_self_image_score_z ~
    selective_presentation_score_z * female
    """,
    data=rq3a
).fit()

rq3_offline = smf.ols(
    """
    offline_self_image_score_z ~
    selective_presentation_score_z * female
    """,
    data=rq3a
).fit()

print("\n" + "="*70)
print("FORMAL RQ3A: Selective × Gender -> Online Self-image")
print("="*70)
print(rq3_online.summary2().tables[1].round(4))

print("\n" + "="*70)
print("FORMAL RQ3A: Selective × Gender -> Offline Self-image")
print("="*70)
print(rq3_offline.summary2().tables[1].round(4))

# ------------------------------------------------------------
# 3.3 RQ3B:
# Online / Offline Self-image × Gender -> Psychology
# ------------------------------------------------------------

for name, outcome in psych_outcomes.items():

    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female"
    ]

    d = topic1[cols].dropna().copy()

    for col in [
        outcome,
        "online_self_image_score",
        "offline_self_image_score"
    ]:
        d[col + "_z"] = (
            d[col] - d[col].mean()
        ) / d[col].std()

    model = smf.ols(
        f"""
        {outcome}_z ~
        online_self_image_score_z * female +
        offline_self_image_score_z * female
        """,
        data=d
    ).fit()

    print("\n" + "="*70)
    print(f"FORMAL RQ3B: Gender moderation - {name}")
    print("="*70)

    print(model.summary2().tables[1].round(4))
    print("N =", int(model.nobs))
    print("R² =", round(model.rsquared, 4))

    # ============================================================
# FORMAL ANALYSIS 4
# Adjusted models with background controls
# Controls:
#   female = 1 female, 0 male
#   bs3 = subjective family socioeconomic status (1-10)
# ============================================================

# Clean subjective family SES
topic1["family_ses"] = topic1["bs3"].where(
    topic1["bs3"].between(1, 10)
)

# Standardize SES
topic1["family_ses_z"] = (
    topic1["family_ses"] - topic1["family_ses"].mean()
) / topic1["family_ses"].std()

print("\nControl variable missingness:")
print(
    topic1[
        ["female", "family_ses"]
    ].isna().sum()
)

print("\nFamily SES distribution:")
print(topic1["family_ses"].describe())

# ------------------------------------------------------------
# 4B. Adjusted RQ1
# ------------------------------------------------------------

rq1_control_vars = [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f",
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "female",
    "family_ses"
]

d = topic1[rq1_control_vars].copy()

# Clean internet-use variables
for col in ["bs21a", "bs21b", "bs21c",
            "bs21d", "bs21e", "bs21f"]:
    d[col] = d[col].where(d[col] > 0)

d = d.dropna()

# Standardize continuous/ordinal variables within same sample
z_vars = [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f",
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "family_ses"
]

for col in z_vars:
    d[col + "_z"] = (
        d[col] - d[col].mean()
    ) / d[col].std()

predictors = """
bs21a_z + bs21b_z + bs21c_z +
bs21d_z + bs21e_z + bs21f_z +
selective_presentation_score_z
"""

for outcome in [
    "online_self_image_score_z",
    "offline_self_image_score_z"
]:

    m0 = smf.ols(
        f"{outcome} ~ {predictors}",
        data=d
    ).fit()

    m1 = smf.ols(
        f"""
        {outcome} ~ {predictors}
        + female + family_ses_z
        """,
        data=d
    ).fit()

    print("\n" + "="*70)
    print("Adjusted RQ1:", outcome)
    print("="*70)

    print("\n--- Model 0 ---")
    print(m0.summary2().tables[1].round(4))

    print("\n--- Model 1: + Gender + Family SES ---")
    print(m1.summary2().tables[1].round(4))

    print("\nN =", int(m1.nobs))
    print("R² M0 =", round(m0.rsquared, 4))
    print("R² M1 =", round(m1.rsquared, 4))
    print("Delta R² =", round(m1.rsquared - m0.rsquared, 4))

    # ------------------------------------------------------------
# 4C. Adjusted RQ2
# ------------------------------------------------------------

psych_outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depression": "depression_score"
}

for name, outcome in psych_outcomes.items():

    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female",
        "family_ses"
    ]

    d = topic1[cols].dropna().copy()

    for col in [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "family_ses"
    ]:
        d[col + "_z"] = (
            d[col] - d[col].mean()
        ) / d[col].std()

    # Same sample
    m0 = smf.ols(
        f"""
        {outcome}_z ~
        online_self_image_score_z +
        offline_self_image_score_z
        """,
        data=d
    ).fit()

    m1 = smf.ols(
        f"""
        {outcome}_z ~
        online_self_image_score_z +
        offline_self_image_score_z +
        female +
        family_ses_z
        """,
        data=d
    ).fit()

    contrast = m1.t_test(
        "online_self_image_score_z - "
        "offline_self_image_score_z = 0"
    )

    print("\n" + "="*70)
    print("Adjusted RQ2:", name)
    print("="*70)

    print("\n--- Model 0 ---")
    print(m0.summary2().tables[1].round(4))

    print("\n--- Model 1: + Gender + Family SES ---")
    print(m1.summary2().tables[1].round(4))

    print("\nOnline vs Offline coefficient test:")
    print(contrast)

    print("\nN =", int(m1.nobs))
    print("R² M0 =", round(m0.rsquared, 4))
    print("R² M1 =", round(m1.rsquared, 4))
    print("Delta R² =", round(m1.rsquared - m0.rsquared, 4))

    # ============================================================
# 4D. Adjusted RQ3
# Gender moderation + Family SES
# ============================================================

# ------------------------------------------------------------
# RQ3A:
# Selective Presentation × Gender -> Self-image
# controlling Family SES
# ------------------------------------------------------------

rq3a_vars = [
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "female",
    "family_ses"
]

d = topic1[rq3a_vars].dropna().copy()

for col in [
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "family_ses"
]:
    d[col + "_z"] = (
        d[col] - d[col].mean()
    ) / d[col].std()

for outcome in [
    "online_self_image_score_z",
    "offline_self_image_score_z"
]:

    model = smf.ols(
        f"""
        {outcome} ~
        selective_presentation_score_z * female +
        family_ses_z
        """,
        data=d
    ).fit()

    print("\n" + "="*70)
    print("Adjusted RQ3A:", outcome)
    print("="*70)

    print(model.summary2().tables[1].round(4))
    print("N =", int(model.nobs))
    print("R² =", round(model.rsquared, 4))


# ------------------------------------------------------------
# RQ3B:
# Online / Offline Self-image × Gender -> Psychology
# controlling Family SES
# ------------------------------------------------------------

psych_outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depression": "depression_score"
}

for name, outcome in psych_outcomes.items():

    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female",
        "family_ses"
    ]

    d = topic1[cols].dropna().copy()

    for col in [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "family_ses"
    ]:
        d[col + "_z"] = (
            d[col] - d[col].mean()
        ) / d[col].std()

    model = smf.ols(
        f"""
        {outcome}_z ~
        online_self_image_score_z * female +
        offline_self_image_score_z * female +
        family_ses_z
        """,
        data=d
    ).fit()

    print("\n" + "="*70)
    print("Adjusted RQ3B:", name)
    print("="*70)

    print(model.summary2().tables[1].round(4))
    print("N =", int(model.nobs))
    print("R² =", round(model.rsquared, 4))

    # ============================================================
# FORMAL ANALYSIS 5A
# Survey design / weight audit
# ============================================================

survey_vars = [
    "JKZONE",
    "JKREP",
    "SCHWGT",
    "CLAWGT",
    "STUWGT",
    "TOTWGT",
    "HOUWGT"
]

print("\n" + "="*70)
print("SURVEY DESIGN VARIABLE AUDIT")
print("="*70)

for col in survey_vars:

    print(f"\n--- {col} ---")

    if col not in df.columns:
        print("NOT FOUND")
        continue

    x = df[col]

    print("dtype:", x.dtype)
    print("N valid:", x.notna().sum())
    print("N missing:", x.isna().sum())
    print("N unique:", x.nunique(dropna=True))

    if pd.api.types.is_numeric_dtype(x):
        print(x.describe())

    print("First values:")
    print(x.value_counts(dropna=False).head(15))

    design_id_vars = [
    "nschool_id",
    "nclass",
    "school_class"
]

print("\n" + "="*70)
print("CLUSTER / ID VARIABLE AUDIT")
print("="*70)

for col in design_id_vars:

    print(f"\n--- {col} ---")

    if col not in df.columns:
        print("NOT FOUND")
        continue

    print("N unique:", df[col].nunique(dropna=True))
    print("N missing:", df[col].isna().sum())
    print(df[col].value_counts(dropna=False).head(10))

    # Convert survey variables to numeric
topic1["HOUWGT"] = pd.to_numeric(
    topic1["HOUWGT"], errors="coerce"
)

topic1["nschool_id"] = pd.to_numeric(
    topic1["nschool_id"], errors="coerce"
)

print(topic1["HOUWGT"].describe())
print("Sum of HOUWGT =", topic1["HOUWGT"].sum())

# ============================================================
# FORMAL ANALYSIS 5C
# HOUWGT-weighted regression
# + school-cluster robust standard errors
# ============================================================

def zscore(x):
    return (x - x.mean()) / x.std()


# ============================================================
# RQ1
# Digital use + selective presentation -> self-image
# Controls: gender + family SES
# ============================================================

rq1_vars = [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f",
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "female",
    "family_ses",
    "HOUWGT",
    "nschool_id"
]

d = topic1[rq1_vars].copy()

# Internet-use valid responses
for col in [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f"
]:
    d[col] = d[col].where(d[col] > 0)

d = d.dropna()

# Standardize within analytic sample
for col in [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f",
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "family_ses"
]:
    d[col + "_z"] = zscore(d[col])


rq1_formula_base = """
    {outcome} ~
    bs21a_z + bs21b_z + bs21c_z +
    bs21d_z + bs21e_z + bs21f_z +
    selective_presentation_score_z +
    female + family_ses_z
"""

for outcome in [
    "online_self_image_score_z",
    "offline_self_image_score_z"
]:

    model = smf.wls(
        rq1_formula_base.format(outcome=outcome),
        data=d,
        weights=d["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["nschool_id"]}
    )

    print("\n" + "="*70)
    print("WEIGHTED + CLUSTERED RQ1:", outcome)
    print("="*70)

    print(model.summary2().tables[1].round(4))
    print("N =", int(model.nobs))
    print("R² =", round(model.rsquared, 4))


# ============================================================
# RQ2
# Online + Offline self-image -> psychological outcomes
# Controls: gender + family SES
# ============================================================

psych_outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depression": "depression_score"
}

for name, outcome in psych_outcomes.items():

    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female",
        "family_ses",
        "HOUWGT",
        "nschool_id"
    ]

    d = topic1[cols].dropna().copy()

    for col in [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "family_ses"
    ]:
        d[col + "_z"] = zscore(d[col])

    model = smf.wls(
        f"""
        {outcome}_z ~
        online_self_image_score_z +
        offline_self_image_score_z +
        female +
        family_ses_z
        """,
        data=d,
        weights=d["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["nschool_id"]}
    )

    print("\n" + "="*70)
    print("WEIGHTED + CLUSTERED RQ2:", name)
    print("="*70)

    print(model.summary2().tables[1].round(4))

    # Online vs Offline coefficient test
    test = model.t_test(
        "online_self_image_score_z - "
        "offline_self_image_score_z = 0"
    )

    print("\nOnline vs Offline coefficient test:")
    print(test)

    print("N =", int(model.nobs))
    print("R² =", round(model.rsquared, 4))


# ============================================================
# RQ3A
# Selective presentation × Gender -> self-image
# + Family SES
# ============================================================

cols = [
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "female",
    "family_ses",
    "HOUWGT",
    "nschool_id"
]

d = topic1[cols].dropna().copy()

for col in [
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "family_ses"
]:
    d[col + "_z"] = zscore(d[col])

for outcome in [
    "online_self_image_score_z",
    "offline_self_image_score_z"
]:

    model = smf.wls(
        f"""
        {outcome} ~
        selective_presentation_score_z * female +
        family_ses_z
        """,
        data=d,
        weights=d["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["nschool_id"]}
    )

    print("\n" + "="*70)
    print("WEIGHTED + CLUSTERED RQ3A:", outcome)
    print("="*70)

    print(model.summary2().tables[1].round(4))
    print("N =", int(model.nobs))
    print("R² =", round(model.rsquared, 4))


# ============================================================
# RQ3B
# Online / Offline self-image × Gender
# -> psychological outcomes
# + Family SES
# ============================================================

for name, outcome in psych_outcomes.items():

    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female",
        "family_ses",
        "HOUWGT",
        "nschool_id"
    ]

    d = topic1[cols].dropna().copy()

    for col in [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "family_ses"
    ]:
        d[col + "_z"] = zscore(d[col])

    model = smf.wls(
        f"""
        {outcome}_z ~
        online_self_image_score_z * female +
        offline_self_image_score_z * female +
        family_ses_z
        """,
        data=d,
        weights=d["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["nschool_id"]}
    )

    print("\n" + "="*70)
    print("WEIGHTED + CLUSTERED RQ3B:", name)
    print("="*70)

    print(model.summary2().tables[1].round(4))
    print("N =", int(model.nobs))
    print("R² =", round(model.rsquared, 4))

    # ============================================================
# FORMAL ANALYSIS 6A
# Robustness: remove bs25b from Online Positive Self-image
# ============================================================

topic1["online_self_image_reduced"] = (
    topic1[["bs25a", "bs25c"]].mean(axis=1)
)

print("\nReduced Online Self-image:")
print(topic1["online_self_image_reduced"].describe())

print("\nCorrelation with original scale:")
print(
    topic1[
        ["online_self_image_score", "online_self_image_reduced"]
    ].corr()
)
rq1_robust_vars = [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f",
    "selective_presentation_score",
    "online_self_image_reduced",
    "female",
    "family_ses",
    "HOUWGT",
    "nschool_id"
]

d = topic1[rq1_robust_vars].copy()

for col in [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f"
]:
    d[col] = d[col].where(d[col] > 0)

d = d.dropna()

for col in [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f",
    "selective_presentation_score",
    "online_self_image_reduced",
    "family_ses"
]:
    d[col + "_z"] = (
        d[col] - d[col].mean()
    ) / d[col].std()

model = smf.wls(
    """
    online_self_image_reduced_z ~
    bs21a_z + bs21b_z + bs21c_z +
    bs21d_z + bs21e_z + bs21f_z +
    selective_presentation_score_z +
    female + family_ses_z
    """,
    data=d,
    weights=d["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={"groups": d["nschool_id"]}
)

print("\n" + "="*70)
print("ROBUSTNESS 6A: RQ1 WITHOUT bs25b")
print("="*70)

print(model.summary2().tables[1].round(4))
print("N =", int(model.nobs))
print("R² =", round(model.rsquared, 4))

# ============================================================
# FORMAL ANALYSIS 6B
# Robustness: remove bs23c from Selective Presentation
# ============================================================

topic1["selective_presentation_reduced"] = (
    topic1[["bs23a", "bs23b"]].mean(axis=1)
)

print("\nReduced Selective Presentation:")
print(topic1["selective_presentation_reduced"].describe())

print("\nCorrelation with original scale:")
print(
    topic1[
        [
            "selective_presentation_score",
            "selective_presentation_reduced"
        ]
    ].corr()
)

rq1_robust_vars = [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f",
    "selective_presentation_reduced",
    "online_self_image_score",
    "female",
    "family_ses",
    "HOUWGT",
    "nschool_id"
]

d = topic1[rq1_robust_vars].copy()

for col in [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f"
]:
    d[col] = d[col].where(d[col] > 0)

d = d.dropna()

for col in [
    "bs21a", "bs21b", "bs21c",
    "bs21d", "bs21e", "bs21f",
    "selective_presentation_reduced",
    "online_self_image_score",
    "family_ses"
]:
    d[col + "_z"] = (
        d[col] - d[col].mean()
    ) / d[col].std()

model = smf.wls(
    """
    online_self_image_score_z ~
    bs21a_z + bs21b_z + bs21c_z +
    bs21d_z + bs21e_z + bs21f_z +
    selective_presentation_reduced_z +
    female + family_ses_z
    """,
    data=d,
    weights=d["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={"groups": d["nschool_id"]}
)

print("\n" + "="*70)
print("ROBUSTNESS 6B: RQ1 WITHOUT bs23c")
print("="*70)

print(model.summary2().tables[1].round(4))
print("N =", int(model.nobs))
print("R² =", round(model.rsquared, 4))

# ============================================================
# FORMAL ANALYSIS 6C
# Robustness / Suppression check for RQ2
# ============================================================

psych_outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depression": "depression_score"
}

for name, outcome in psych_outcomes.items():

    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female",
        "family_ses",
        "HOUWGT",
        "nschool_id"
    ]

    d = topic1[cols].dropna().copy()

    for col in [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "family_ses"
    ]:
        d[col + "_z"] = (
            d[col] - d[col].mean()
        ) / d[col].std()

    # Model 1: Online only
    m_online = smf.wls(
        f"""
        {outcome}_z ~
        online_self_image_score_z +
        female + family_ses_z
        """,
        data=d,
        weights=d["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["nschool_id"]}
    )

    # Model 2: Offline only
    m_offline = smf.wls(
        f"""
        {outcome}_z ~
        offline_self_image_score_z +
        female + family_ses_z
        """,
        data=d,
        weights=d["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["nschool_id"]}
    )

    # Model 3: Online + Offline
    m_both = smf.wls(
        f"""
        {outcome}_z ~
        online_self_image_score_z +
        offline_self_image_score_z +
        female + family_ses_z
        """,
        data=d,
        weights=d["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["nschool_id"]}
    )

    print("\n" + "="*70)
    print("ROBUSTNESS 6C:", name)
    print("="*70)

    print("\nONLINE ONLY")
    print(
        m_online.summary2().tables[1]
        .loc[["online_self_image_score_z"]]
        .round(4)
    )
    print("R² =", round(m_online.rsquared, 4))

    print("\nOFFLINE ONLY")
    print(
        m_offline.summary2().tables[1]
        .loc[["offline_self_image_score_z"]]
        .round(4)
    )
    print("R² =", round(m_offline.rsquared, 4))

    print("\nONLINE + OFFLINE")
    print(
        m_both.summary2().tables[1]
        .loc[
            [
                "online_self_image_score_z",
                "offline_self_image_score_z"
            ]
        ]
        .round(4)
    )
    print("R² =", round(m_both.rsquared, 4))

    # ============================================================
# FORMAL ANALYSIS 6D
# Reparameterization:
# Overall self-image level + Online–Offline discrepancy
# ============================================================

psych_outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depression": "depression_score"
}

for name, outcome in psych_outcomes.items():

    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female",
        "family_ses",
        "HOUWGT",
        "nschool_id"
    ]

    d = topic1[cols].dropna().copy()

    # Standardize Online and Offline first
    d["online_z"] = (
        d["online_self_image_score"]
        - d["online_self_image_score"].mean()
    ) / d["online_self_image_score"].std()

    d["offline_z"] = (
        d["offline_self_image_score"]
        - d["offline_self_image_score"].mean()
    ) / d["offline_self_image_score"].std()

    # Overall level and directional discrepancy
    d["self_image_mean"] = (
        d["online_z"] + d["offline_z"]
    ) / 2

    d["online_offline_diff"] = (
        d["online_z"] - d["offline_z"]
    )

    # Standardize outcome and SES
    d["outcome_z"] = (
        d[outcome] - d[outcome].mean()
    ) / d[outcome].std()

    d["family_ses_z"] = (
        d["family_ses"] - d["family_ses"].mean()
    ) / d["family_ses"].std()

    model = smf.wls(
        """
        outcome_z ~
        self_image_mean +
        online_offline_diff +
        female +
        family_ses_z
        """,
        data=d,
        weights=d["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["nschool_id"]}
    )

    print("\n" + "="*70)
    print("ROBUSTNESS 6D:", name)
    print("="*70)

    print(
        model.summary2().tables[1]
        .loc[
            ["self_image_mean", "online_offline_diff"]
        ]
        .round(4)
    )

    print("N =", int(model.nobs))
    print("R² =", round(model.rsquared, 4))

# ============================================================
# STEP 7A — RQ3A FINAL SIMPLE SLOPES
# Weighted regression + school-clustered SE
# ============================================================

import statsmodels.formula.api as smf

# 1. Prepare data
d = topic1[
    [
        "selective_presentation_score",
        "online_self_image_score",
        "offline_self_image_score",
        "female",
        "family_ses",
        "HOUWGT",
        "nschool_id"
    ]
].dropna().copy()


# 2. Standardize continuous variables
for col in [
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "family_ses"
]:
    d[col + "_z"] = (
        d[col] - d[col].mean()
    ) / d[col].std()


# 3. Online self-image model
rq3a_online_final = smf.wls(
    """
    online_self_image_score_z ~
    selective_presentation_score_z * female
    + family_ses_z
    """,
    data=d,
    weights=d["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={"groups": d["nschool_id"]}
)


# 4. Offline self-image model
rq3a_offline_final = smf.wls(
    """
    offline_self_image_score_z ~
    selective_presentation_score_z * female
    + family_ses_z
    """,
    data=d,
    weights=d["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={"groups": d["nschool_id"]}
)


# 5. Function for simple slopes
def print_simple_slopes(model):

    predictor = "selective_presentation_score_z"
    interaction = "selective_presentation_score_z:female"

    # Male = female 0
    male = model.t_test(
        f"{predictor} = 0"
    )

    # Female = female 1
    female = model.t_test(
        f"{predictor} + {interaction} = 0"
    )

    print("\nMale simple slope:")
    print(male)

    print("\nFemale simple slope:")
    print(female)


# 6. Print results
print("\n" + "=" * 70)
print("STEP 7A — ONLINE SELF-IMAGE")
print("=" * 70)

print(rq3a_online_final.summary2().tables[1].round(4))
print_simple_slopes(rq3a_online_final)


print("\n" + "=" * 70)
print("STEP 7A — OFFLINE SELF-IMAGE")
print("=" * 70)

print(rq3a_offline_final.summary2().tables[1].round(4))
print_simple_slopes(rq3a_offline_final)

# ============================================================
# STEP 7B — RQ3B SIMPLE SLOPES
# Weighted regression + school-clustered SE
# ============================================================

psych_outcomes = {
    "SELF-ESTEEM": "self_esteem_score",
    "WELL-BEING": "wellbeing_score",
    "DEPRESSION": "depression_score"
}

for outcome_name, outcome in psych_outcomes.items():

    cols = [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "female",
        "family_ses",
        "HOUWGT",
        "nschool_id"
    ]

    d = topic1[cols].dropna().copy()

    # Standardize continuous variables
    for col in [
        outcome,
        "online_self_image_score",
        "offline_self_image_score",
        "family_ses"
    ]:
        d[col + "_z"] = (
            d[col] - d[col].mean()
        ) / d[col].std()

    # Model:
    # Online + Offline + both gender interactions
    model = smf.wls(
        f"""
        {outcome}_z ~
        online_self_image_score_z * female +
        offline_self_image_score_z * female +
        family_ses_z
        """,
        data=d,
        weights=d["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["nschool_id"]}
    )

    print("\n" + "=" * 70)
    print(f"STEP 7B — {outcome_name}")
    print("=" * 70)

    # -------------------------
    # ONLINE simple slopes
    # -------------------------

    online_male = model.t_test(
        "online_self_image_score_z = 0"
    )

    online_female = model.t_test(
        """
        online_self_image_score_z +
        online_self_image_score_z:female = 0
        """
    )

    print("\nONLINE — Male:")
    print(online_male)

    print("\nONLINE — Female:")
    print(online_female)

    # -------------------------
    # OFFLINE simple slopes
    # -------------------------

    offline_male = model.t_test(
        "offline_self_image_score_z = 0"
    )

    offline_female = model.t_test(
        """
        offline_self_image_score_z +
        offline_self_image_score_z:female = 0
        """
    )

    print("\nOFFLINE — Male:")
    print(offline_male)

    print("\nOFFLINE — Female:")
    print(offline_female)

    print("\nR² =", round(model.rsquared, 4))