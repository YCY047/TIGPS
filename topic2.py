# ============================================================
# Step 0. Load Data & Check Variables
# ============================================================

import pandas as pd
import numpy as np

# 0-1. 讀取資料
df = pd.read_csv("TIGPSw2_s.csv")

print("Dataset shape:", df.shape)
print(df.head())


# ------------------------------------------------------------
# 0-2. 定義 Prototype 需要的變數
# ------------------------------------------------------------

# Offline Social Connectedness
offline_vars = [
    "bs29a", "bs29b", "bs29c", "bs29d", "bs29e",  # 摯友支持
    "bs12a", "bs12b", "bs12c",                    # 班級同儕互助
    "bs11a", "bs11b"                               # 師生關係
]

# Online Social Support Seeking
online_vars = [
    "bs22a",   # 上網找人聊天、尋求安慰
    "bs22b"    # 上網找人討論問題、尋求解決方法
]

# Psychological Well-being
wellbeing_vars = [
    "bs50",    # 生活滿意度
    "bs51",    # 快樂感
    "bs52",    # 健康自評

    "bs53a", "bs53b", "bs53c",  # 自尊

    # 憂鬱情緒
    "bs56a", "bs56b", "bs56c", "bs56d",
    "bs56e", "bs56f", "bs56g", "bs56h",
    "bs56i", "bs56j", "bs56k", "bs56l",
    "bs56m", "bs56n",

    # 情緒狀態
    "bs58a", "bs58b", "bs58c", "bs58d", "bs58e"
]

# Basic demographic variables
background_vars = [
    "bs1",     # 性別
    "bs3"      # 主觀社經地位
]

# Cluster / survey design variables
survey_vars = [
    "school_class",
    "nschool_id",
    "nclass",
    "STUWGT",
    "JKZONE",
    "JKREP"
]


# ------------------------------------------------------------
# 0-3. 確認所有需要的欄位是否真的存在
# ------------------------------------------------------------

all_vars = (
    offline_vars
    + online_vars
    + wellbeing_vars
    + background_vars
    + survey_vars
)

print("\n--- Variable Check ---")

for var in all_vars:
    print(f"{var}: {var in df.columns}")


# 找出不存在的變數
missing_columns = [var for var in all_vars if var not in df.columns]

print("\nMissing columns:", missing_columns)


# ------------------------------------------------------------
# 0-4. 建立 Prototype Dataset
# ------------------------------------------------------------

# 只保留實際存在的欄位，避免不存在的欄位造成 KeyError
existing_vars = [var for var in all_vars if var in df.columns]

proto = df[existing_vars].copy()

print("\nPrototype shape:", proto.shape)
print(proto.head())


# ------------------------------------------------------------
# 0-5. 查看 Offline Connectedness 題目的原始分布
# ------------------------------------------------------------

print("\n========== Offline Connectedness ==========")

for var in offline_vars:
    if var in proto.columns:
        print(f"\n--- {var} ---")
        print(
            proto[var]
            .value_counts(dropna=False)
            .sort_index()
        )


# ------------------------------------------------------------
# 0-6. 查看 Online Social Support Seeking 題目的原始分布
# ------------------------------------------------------------

print("\n========== Online Social Support Seeking ==========")

for var in online_vars:
    if var in proto.columns:
        print(f"\n--- {var} ---")
        print(
            proto[var]
            .value_counts(dropna=False)
            .sort_index()
        )

# ============================================================
# Step 1. Data Cleaning
# ============================================================

# 1-1. Replace special missing codes with NaN

core_vars = offline_vars + online_vars

proto[core_vars] = proto[core_vars].replace(-9, np.nan)

print("Missing values after cleaning:")
print(proto[core_vars].isna().sum())

# Number of valid responses for each student

proto["offline_valid_n"] = proto[offline_vars].notna().sum(axis=1)
proto["online_valid_n"] = proto[online_vars].notna().sum(axis=1)

print("\nOffline valid response counts:")
print(proto["offline_valid_n"].value_counts().sort_index())

print("\nOnline valid response counts:")
print(proto["online_valid_n"].value_counts().sort_index())

# ============================================================
# Step 2. Reliability Analysis
# ============================================================

def cronbach_alpha(data):
    """
    Calculate Cronbach's alpha.
    Rows = participants
    Columns = items
    """

    # 只使用所有題目都有回答的學生
    data = data.dropna()

    k = data.shape[1]

    item_variances = data.var(axis=0, ddof=1)
    total_score = data.sum(axis=1)
    total_variance = total_score.var(ddof=1)

    alpha = (k / (k - 1)) * (
        1 - item_variances.sum() / total_variance
    )

    return alpha
#計算這幾題是否傾向一起變動?

offline_alpha = cronbach_alpha(proto[offline_vars])

print("\nCronbach's alpha")
print("Offline Connectedness:", round(offline_alpha, 3))

friend_vars = [
    "bs29a", "bs29b", "bs29c", "bs29d", "bs29e"
]

classmate_vars = [
    "bs12a", "bs12b", "bs12c"
]

teacher_vars = [
    "bs11a", "bs11b"
]

print("\nSubscale reliability")

print(
    "Friend support:",
    round(cronbach_alpha(proto[friend_vars]), 3)
)

print(
    "Classmate support:",
    round(cronbach_alpha(proto[classmate_vars]), 3)
)

print(
    "Teacher relationship:",
    round(cronbach_alpha(proto[teacher_vars]), 3)
)

online_alpha = cronbach_alpha(proto[online_vars])

print(
    "\nOnline Social Support Seeking:",
    round(online_alpha, 3)
)

def item_total_correlation(data):

    data = data.dropna()

    results = {}

    for item in data.columns:

        # 除了目前這一題之外的其他題目
        other_items = data.drop(columns=item)

        # 其他題目的總分
        other_total = other_items.sum(axis=1)

        # 目前題目與其他題目總分的相關
        correlation = data[item].corr(other_total)

        results[item] = correlation

    return pd.Series(results)


offline_item_total = item_total_correlation(
    proto[offline_vars]
)

print("\nOffline corrected item-total correlations:")
print(
    offline_item_total
    .sort_values()
    .round(3)
)

offline_corr = proto[offline_vars].corr()

print("\nOffline item correlation matrix:")
print(offline_corr.round(2))

# ============================================================
# Step 2.5 Revised Offline Peer Connectedness
# ============================================================

# Friend support
# bs29b excluded because it measures friends' problematic behavior
friend_vars = [
    "bs29a",
    "bs29c",
    "bs29d",
    "bs29e"
]

# Classmate support
classmate_vars = [
    "bs12a",
    "bs12b",
    "bs12c"
]

# Offline Peer Connectedness
peer_vars = friend_vars + classmate_vars


# ------------------------------------------------------------
# Reliability
# ------------------------------------------------------------

print("\n=== Revised Reliability ===")

print(
    "Friend Support:",
    round(cronbach_alpha(proto[friend_vars]), 3)
)

print(
    "Classmate Support:",
    round(cronbach_alpha(proto[classmate_vars]), 3)
)

print(
    "Offline Peer Connectedness:",
    round(cronbach_alpha(proto[peer_vars]), 3)
)

print(
    "Online Social Support Seeking:",
    round(cronbach_alpha(proto[online_vars]), 3)
)


# ------------------------------------------------------------
# Corrected item-total correlations
# ------------------------------------------------------------

peer_item_total = item_total_correlation(proto[peer_vars])

print("\nOffline Peer Connectedness item-total correlations:")
print(
    peer_item_total
    .sort_values()
    .round(3)
)

# ============================================================
# Step 3. Create Core Scores
# ============================================================

# 至少回答 6 / 7 題
offline_valid = proto[peer_vars].notna().sum(axis=1)

proto["offline_peer_score"] = (
    proto[peer_vars]
    .mean(axis=1)
    .where(offline_valid >= 6)
)

# Online 必須兩題都有回答
online_valid = proto[online_vars].notna().sum(axis=1)

proto["online_support_score"] = (
    proto[online_vars]
    .mean(axis=1)
    .where(online_valid == 2)
)


# Descriptive statistics
print("\n=== Core Score Descriptive Statistics ===")

print(
    proto[
        ["offline_peer_score", "online_support_score"]
    ]
    .describe()
    .round(3)
)


# Correlation
print("\n=== Correlation ===")

print(
    proto[
        ["offline_peer_score", "online_support_score"]
    ]
    .corr()
    .round(3)
)

# ============================================================
# Step 4. Create Psychological Well-being Outcomes
# ============================================================

cesd_vars = [
    "bs56a", "bs56b", "bs56c", "bs56d",
    "bs56e", "bs56f", "bs56g", "bs56h",
    "bs56i", "bs56j", "bs56k", "bs56l",
    "bs56m", "bs56n"
]

who5_vars = [
    "bs58a", "bs58b", "bs58c", "bs58d", "bs58e"
]

# 將 -9 轉成 missing
proto[cesd_vars + who5_vars] = (
    proto[cesd_vars + who5_vars]
    .replace(-9, np.nan)
)

# CESD-R：要求至少回答 12 / 14 題
cesd_valid = proto[cesd_vars].notna().sum(axis=1)

proto["depression_score"] = (
    proto[cesd_vars]
    .mean(axis=1)
    .where(cesd_valid >= 12)
)

# WHO-5：要求至少回答 4 / 5 題
who5_valid = proto[who5_vars].notna().sum(axis=1)

proto["wellbeing_score"] = (
    proto[who5_vars]
    .mean(axis=1)
    .where(who5_valid >= 4)
)

print("\n=== Psychological Outcomes ===")

print(
    proto[
        ["depression_score", "wellbeing_score"]
    ]
    .describe()
    .round(3)
)

# ============================================================
# Step 5. Core Regression Models
# ============================================================

import statsmodels.formula.api as smf

# Standardize X variables
proto["offline_z"] = (
    proto["offline_peer_score"]
    - proto["offline_peer_score"].mean()
) / proto["offline_peer_score"].std()

proto["online_z"] = (
    proto["online_support_score"]
    - proto["online_support_score"].mean()
) / proto["online_support_score"].std()


# ------------------------------------------------------------
# Model 1: Depressive symptoms
# ------------------------------------------------------------

model_dep = smf.ols(
    "depression_score ~ offline_z * online_z",
    data=proto
).fit()

print("\n========== Depression Model ==========")
print(model_dep.summary())


# ------------------------------------------------------------
# Model 2: Positive well-being
# ------------------------------------------------------------

model_wb = smf.ols(
    "wellbeing_score ~ offline_z * online_z",
    data=proto
).fit()

print("\n========== Well-being Model ==========")
print(model_wb.summary())


# ============================================================
# Step 6. Four Social Support Profiles
# ============================================================

offline_median = proto["offline_peer_score"].median()
online_median = proto["online_support_score"].median()

print("Offline median:", offline_median)
print("Online median:", online_median)


proto["offline_high"] = (
    proto["offline_peer_score"] >= offline_median
)

proto["online_high"] = (
    proto["online_support_score"] >= online_median
)


def create_profile(row):

    if pd.isna(row["offline_peer_score"]) or pd.isna(row["online_support_score"]):
        return np.nan

    if row["offline_high"] and row["online_high"]:
        return "High Offline / High Online"

    elif row["offline_high"] and not row["online_high"]:
        return "High Offline / Low Online"

    elif not row["offline_high"] and row["online_high"]:
        return "Low Offline / High Online"

    else:
        return "Low Offline / Low Online"


proto["support_profile"] = proto.apply(
    create_profile,
    axis=1
)


# ------------------------------------------------------------
# Profile distribution
# ------------------------------------------------------------

print("\n=== Profile Distribution ===")

print(
    proto["support_profile"]
    .value_counts()
)

print("\n=== Profile Percentage ===")

print(
    (
        proto["support_profile"]
        .value_counts(normalize=True)
        * 100
    ).round(1)
)


# ------------------------------------------------------------
# Mental health by profile
# ------------------------------------------------------------

profile_results = (
    proto
    .groupby("support_profile")
    [["depression_score", "wellbeing_score"]]
    .agg(["mean", "std", "count"])
    .round(3)
)

print("\n=== Mental Health by Profile ===")

print(profile_results)

print("\n=== Profile Mean Scores ===")

print(
    proto.groupby("support_profile")[
        ["depression_score", "wellbeing_score"]
    ]
    .mean()
    .round(3)
    .to_string()
)
