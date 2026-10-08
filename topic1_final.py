import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

#1. load Data
df= pd.read_csv(
    "TIGPSw2_s.csv",
    low_memory=False)
print("Raw data shape:",df.shape)

#2. Define Variables
#selective positive self-presentation
selective_vars = [
    "bs23a", "bs23b", "bs23c"
]

# Online Positive Self-image
online_image_vars = [
    "bs25a", "bs25b", "bs25c"
]

# Offline Positive Self-image
offline_image_vars = [
    "bs25d", "bs25e", "bs25f"
]

# General Self-esteem
self_esteem_vars = [
    "bs53a", "bs53b", "bs53c"
]

# Depressive symptoms
depression_vars = [
    "bs56a", "bs56b", "bs56c", "bs56d",
    "bs56e", "bs56f", "bs56g", "bs56h",
    "bs56i", "bs56j", "bs56k", "bs56l",
    "bs56m", "bs56n"
]

# Well-being
wellbeing_vars = [
    "bs58a", "bs58b", "bs58c",
    "bs58d", "bs58e"
]

# Background / survey-design variables
background_vars = [
    "bs1",          # Gender
    "bs3",          # Subjective family SES
    "HOUWGT",       # Student house weight
    "nschool_id"    # School ID
]

#3. Check Variables and Create Analysis Data
analysis_vars = (
    selective_vars
    + online_image_vars
    + offline_image_vars
    + self_esteem_vars
    + depression_vars
    + wellbeing_vars
    + background_vars
)

# Check whether any required variables are missing
missing_vars = [
    var for var in analysis_vars
    if var not in df.columns
]

print("Missing variables:", missing_vars)

# Create analysis dataframe
topic1 = df[analysis_vars].copy()

print("Analysis data shape:", topic1.shape)
#將分散的六組量表+background variables 合併成一組清單

#4. Data cleaning
questionnaire_vars = (
    selective_vars
    + online_image_vars
    + offline_image_vars
    + self_esteem_vars
    + depression_vars
    + wellbeing_vars
)

for col in questionnaire_vars:
    topic1[col]=pd.to_numeric(
        topic1[col],
        errors="coerce"
    )
    topic1[col]=topic1[col].where(
        topic1[col]>=0 #負數->NaN;0以上先保留 
        #depression variable 有合法的0
    )
print(
        "Negative questionaire values remaining:",
        (topic1[questionnaire_vars]<0).sum().sum()
    )

#5. Clean Background and survey variables
#轉換數值
for col in background_vars:
    topic1[col]=pd.to_numeric(
        topic1[col],
        errors="coerce"
    )
 # Gender: 1 = Female, 2 = Male
topic1["bs1"] = topic1["bs1"].where(
    topic1["bs1"].isin([1, 2])
)   
# Subjective family SES: valid range = 1–10
topic1["bs3"] = topic1["bs3"].where(
    topic1["bs3"].between(1, 10)
)
# Survey weight must be positive
topic1["HOUWGT"] = topic1["HOUWGT"].where(
    topic1["HOUWGT"] > 0
)
print("\nBackground variables after cleaning:")
print(
    topic1[background_vars]
    .isna()
    .sum()
)

#6. recoding gender
topic1["female"]=topic1["bs1"].map({1:1,2:0}) #female=1 male=0; male 當作reference group
print("\nGender coding:")
print(
    topic1[["bs1","female"]]
    .value_counts()
    .sort_index()
)

#建立Selective positive self-presentation 量表
topic1["selective_presentation_score"]=(
    topic1[selective_vars]
    .mean(axis=1)
    .where(
        topic1[selective_vars].notna().sum(axis=1)==3 #對每一個學生橫向算三題平均;且要三題都有有效答案才保留平均數
    )
)
print("\nSelective Positive Self-presentation:")
print(
    topic1["selective_presentation_score"]
    .describe()
    .round(3) #顯示到小數點第三位
)

#8.建立Online Positive Self-image
topic1["online_self_image_score"]=(
    topic1[online_image_vars]
    .mean(axis=1)
    .where(
        topic1[online_image_vars].notna().sum(axis=1)==3 #對每一個學生橫向算三題平均;且要三題都有有效答案才保留平均數
    )
)
print("\nOnline Positive Self-image:")
print(
    topic1["online_self_image_score"]
    .describe()
    .round(3) #顯示到小數點第三位
)

#9. 建立Offline Positive Self-image
topic1["offline_self_image_score"]=(
    topic1[offline_image_vars]
    .mean(axis=1)
    .where(
        topic1[offline_image_vars].notna().sum(axis=1)==3
    )
)
print("\nOffline Positive Self-image:")
print(
    topic1["offline_self_image_score"]
    .describe()
    .round(3)
)

#11. Reliability Analysis, 檢測三題是不是一致的在測量同個概念(Internal consistency)
def cronbach_alpha(data): 
    data = data.dropna() #只留三題都有回答的人

    k = data.shape[1]
    item_variances = data.var(axis=0, ddof=1)
    total_score = data.sum(axis=1)
    total_variance = total_score.var(ddof=1)

    alpha = (
        k / (k - 1)
        * (1 - item_variances.sum() / total_variance)
    )

    return alpha
selective_alpha=cronbach_alpha(
    topic1[selective_vars]
)
print(
    "\nSelective Cronbach's alpha:",
    round(selective_alpha, 3)
)
online_alpha = cronbach_alpha(
    topic1[online_image_vars]
)

offline_alpha = cronbach_alpha(
    topic1[offline_image_vars]
)

print(
    "Online Self-image Cronbach's alpha:",
    round(online_alpha, 3)
)

print(
    "Offline Self-image Cronbach's alpha:",
    round(offline_alpha, 3)
)
#R1 result: Selective Positive Self-presentation 與 Online、Offline Positive Self-image 都呈正向關聯，但與 Online Positive Self-image 的關聯顯著更強

#12. 建構psychological outcome 量表
topic1["self_esteem_score"]=(
    topic1[self_esteem_vars]
    .mean(axis=1)
    .where(
        topic1[self_esteem_vars].notna().sum(axis=1)==3 #對每一個學生橫向算三題平均;且要三題都有有效答案才保留平均數
    )
)
print("\nGeneral Self-esteem:")
print(
    topic1["self_esteem_score"]
    .describe()
    .round(3) #顯示到小數點第三位
)
self_esteem_alpha = cronbach_alpha(
    topic1[self_esteem_vars]
)

print(
    "Self-esteem Cronbach's alpha:",
    round(self_esteem_alpha, 3)
)
#13. 建立Well-being
topic1["wellbeing_score"]=(
    topic1[wellbeing_vars]
    .mean(axis=1)
    .where(
        topic1[wellbeing_vars].notna().sum(axis=1)==5 #wellbeing有五題
    )
)
print("\nWell-being:")
print(
    topic1["wellbeing_score"]
    .describe()
    .round(3) #顯示到小數點第三位
)
wellbeing_alpha = cronbach_alpha(
    topic1[wellbeing_vars]
)

print(
    "Well-being Cronbach's alpha:",
    round(wellbeing_alpha, 3)
)
#14. 建立 Depressive symptoms scale
topic1["depression_score"] = (
    topic1[depression_vars]
    .mean(axis=1)
    .where(
        topic1[depression_vars].notna().sum(axis=1) == 14
    )
)
print("\nDepressive Symptoms:")
print(
    topic1["depression_score"]
    .describe()
    .round(3)
)
depression_alpha = cronbach_alpha(
    topic1[depression_vars]
)
print(
    "Depressive Symptoms Cronbach's alpha:",
    round(depression_alpha, 3)
)
#15 prepare RQ1 analytic sample
rq1_vars = [
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "female",
    "bs3",
    "HOUWGT",
    "nschool_id"
]
rq1 = (
    topic1[rq1_vars]
    .dropna() #找出要得7個變數再drop
    .copy()
)
print("\nRQ1 analytic sample:")
print("N =", len(rq1))

#16. RQ1樣本內做標準化
standardize_vars = [
    "selective_presentation_score",
    "online_self_image_score",
    "offline_self_image_score",
    "bs3"
]

for col in standardize_vars:
    rq1[col + "_z"] = (
        rq1[col] - rq1[col].mean()
    ) / rq1[col].std()

print("\nStandardized RQ1 variables:")
print(
    rq1[
        [col + "_z" for col in standardize_vars]
    ]
    .describe()
    .round(3)
)

#17. RQ1 model1: selective->online self-image
model_online=smf.wls( #weighted least squares
    formula=(
        "online_self_image_score_z~"
        "selective_presentation_score_z+"
        "female+bs3_z"
    ),
    data=rq1,
    weights=rq1["HOUWGT"]#不同學生照HOUWGT有不同分析權重
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups":rq1["nschool_id"] #同一所學校的學生不一定彼此獨立；以學校為cluster做調整
    }
)
print("\nRQ1 Model 1: Online Positive Self-image")
print(model_online.summary())

#RQ1 model2: 
model_offline=smf.wls(
    formula=(
        "offline_self_image_score_z~"
        "selective_presentation_score_z+"
        "female+bs3_z"
    ),
    data=rq1,
    weights=rq1["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups":rq1["nschool_id"]
    }
)
print("\nRQ1 Model 2: Offline Positive Self-image")
print(model_offline.summary())

#19A: reshape RQ1 for slope comparison
rq1_long = pd.concat([
    rq1.assign(
        self_image_z=rq1["offline_self_image_score_z"],
        online_type=0
    ),
    rq1.assign(
        self_image_z=rq1["online_self_image_score_z"],
        online_type=1
    )
], ignore_index=True)

print("\nRQ1 long-format sample:")
print("Rows =", len(rq1_long))
print(
    rq1_long["online_type"]
    .value_counts()
    .sort_index()
)

#19B: 檢測Online 與offline 斜率差異
model_slope_difference = smf.wls(
    formula=(
        "self_image_z ~ "
        "online_type * selective_presentation_score_z + "
        "online_type * female + "
        "online_type * bs3_z"
    ),
    data=rq1_long,
    weights=rq1_long["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": rq1_long["nschool_id"]
    }
)
print("\nRQ1 Online vs. Offline Slope Comparison")
print(model_slope_difference.summary())

#20. 建立Online-Offline Self-image gap
#Gap=Online Positive Self image-Offline Positive Self image
topic1["self_image_gap"]=(
    topic1["online_self_image_score"]-topic1["offline_self_image_score"]
)
print("\nOnline-Offline Self-image Gap:")
print(
    topic1["self_image_gap"]
    .describe()
    .round(3)
)

#21. 建立RQ2 analytic examples
def prepare_rq2_sample(outcome):
    variables=[
        "online_self_image_score",
        "offline_self_image_score",
        outcome,
        "female",
        "bs3",
        "HOUWGT",
        "nschool_id"
    ]
    sample=(
        topic1[variables]
        .dropna()
        .copy()
    )
    return sample
rq2_self_esteem=prepare_rq2_sample(
    "self_esteem_score"
)
print(
    "\nRQ2 Self-esteem analytic sample:",
    len(rq2_self_esteem)
)
#22.standardize RQ2 self-esteem variables
rq2_self_esteem_vars = [
    "online_self_image_score",
    "offline_self_image_score",
    "self_esteem_score",
    "bs3"
]

for col in rq2_self_esteem_vars:
    rq2_self_esteem[col + "_z"] = (
        rq2_self_esteem[col]
        - rq2_self_esteem[col].mean()
    ) / rq2_self_esteem[col].std()

print("\nStandardized RQ2 Self-esteem variables:")
print(
    rq2_self_esteem[
        [col + "_z" for col in rq2_self_esteem_vars]
    ]
    .describe()
    .round(3)
)
#RQ2 model1:general self-esteem
model_self_esteem = smf.wls(
    formula=(
        "self_esteem_score_z ~ " 
        "online_self_image_score_z +" #將online和offline同時放入同一個regression
        "offline_self_image_score_z + "
        "female + bs3_z"
    ),
    data=rq2_self_esteem,
    weights=rq2_self_esteem["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": rq2_self_esteem["nschool_id"]
    }
)
print("\nRQ2 Model 1: General Self-esteem")
print(model_self_esteem.summary())

#24. well being sample
rq2_wellbeing = prepare_rq2_sample(
    "wellbeing_score"
)
print(
    "\nRQ2 Well-being analytic sample:",
    len(rq2_wellbeing)
)

rq2_wellbeing_vars = [
    "online_self_image_score",
    "offline_self_image_score",
    "wellbeing_score",
    "bs3"
]

for col in rq2_wellbeing_vars:
    rq2_wellbeing[col + "_z"] = (
        rq2_wellbeing[col]
        - rq2_wellbeing[col].mean()
    ) / rq2_wellbeing[col].std()

print("\nStandardized RQ2 Well-being variables:")
print(
    rq2_wellbeing[
        [col + "_z" for col in rq2_wellbeing_vars]
    ]
    .describe()
    .round(3)
)

#RQ2 model2: Well-being
model_wellbeing = smf.wls(
    formula=(
        "wellbeing_score_z ~ "
        "online_self_image_score_z + "
        "offline_self_image_score_z + "
        "female + bs3_z"
    ),
    data=rq2_wellbeing,
    weights=rq2_wellbeing["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": rq2_wellbeing["nschool_id"]
    }
)
print("\nRQ2 Model 2: Well-being")
print(model_wellbeing.summary())

#25. Prepare RQ2 depression samplw
rq2_depression = prepare_rq2_sample(
    "depression_score"
)
print(
    "\nRQ2 Depression analytic sample:",
    len(rq2_depression)
)
#標準化Depression model
rq2_depression_vars = [
    "online_self_image_score",
    "offline_self_image_score",
    "depression_score",
    "bs3"
]

for col in rq2_depression_vars:
    rq2_depression[col + "_z"] = (
        rq2_depression[col]
        - rq2_depression[col].mean()
    ) / rq2_depression[col].std()

print("\nStandardized RQ2 Depression variables:")
print(
    rq2_depression[
        [col + "_z" for col in rq2_depression_vars]
    ]
    .describe()
    .round(3)
)
#RQ2 model3: Depressive symptoms
model_depression = smf.wls(
    formula=(
        "depression_score_z ~ "
        "online_self_image_score_z + "
        "offline_self_image_score_z + "
        "female + bs3_z"
    ),
    data=rq2_depression,
    weights=rq2_depression["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": rq2_depression["nschool_id"]
    }
)

print("\nRQ2 Model 3: Depressive Symptoms")
print(model_depression.summary())

#26. RQ2 online vs offline coefficieint contrasts
contrast_self_esteem=model_self_esteem.t_test(
    "online_self_image_score_z -"
    "offline_self_image_score_z=0"
)
print("\nSelf-esteem Online-Offline")
print(contrast_self_esteem)
contrast_wellbeing = model_wellbeing.t_test(
    "online_self_image_score_z - "
    "offline_self_image_score_z = 0"
)

contrast_depression = model_depression.t_test(
    "online_self_image_score_z - "
    "offline_self_image_score_z = 0"
)

print("\nWell-being: Online - Offline")
print(contrast_wellbeing)

print("\nDepressive Symptoms: Online - Offline")
print(contrast_depression)

#27. RQ2 穩健性: separart self-image models

model_self_esteem_online_only = smf.wls(
    formula=(
        "self_esteem_score_z ~ "
        "online_self_image_score_z + "
        "female + bs3_z"
    ),
    data=rq2_self_esteem,
    weights=rq2_self_esteem["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": rq2_self_esteem["nschool_id"]
    }
)

print("\nSelf-esteem: Online-only model")
print(model_self_esteem_online_only.summary())
#online only beta=+.040=> online 本身與self-esteem 不是負向關聯，負係數是在控制offline self image後才出現
def fit_single_predictor_model(data, outcome, predictor):
    model = smf.wls(
        formula=f"{outcome} ~ {predictor} + female + bs3_z",
        data=data,
        weights=data["HOUWGT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": data["nschool_id"]
        }
    )

    return model


# Self-esteem
model_self_esteem_offline_only = fit_single_predictor_model(
    rq2_self_esteem,
    "self_esteem_score_z",
    "offline_self_image_score_z"
)

# Well-being
model_wellbeing_online_only = fit_single_predictor_model(
    rq2_wellbeing,
    "wellbeing_score_z",
    "online_self_image_score_z"
)

model_wellbeing_offline_only = fit_single_predictor_model(
    rq2_wellbeing,
    "wellbeing_score_z",
    "offline_self_image_score_z"
)

# Depression
model_depression_online_only = fit_single_predictor_model(
    rq2_depression,
    "depression_score_z",
    "online_self_image_score_z"
)

model_depression_offline_only = fit_single_predictor_model(
    rq2_depression,
    "depression_score_z",
    "offline_self_image_score_z"
)
print("\nRQ2 Robustness — Single-predictor models")

print(
    "Self-esteem | Online:",
    round(model_self_esteem_online_only.params["online_self_image_score_z"], 4),
    "| Offline:",
    round(model_self_esteem_offline_only.params["offline_self_image_score_z"], 4)
)

print(
    "Well-being  | Online:",
    round(model_wellbeing_online_only.params["online_self_image_score_z"], 4),
    "| Offline:",
    round(model_wellbeing_offline_only.params["offline_self_image_score_z"], 4)
)

print(
    "Depression  | Online:",
    round(model_depression_online_only.params["online_self_image_score_z"], 4),
    "| Offline:",
    round(model_depression_offline_only.params["offline_self_image_score_z"], 4)
)
#28. RQ2 Mean-Difference Reparameterization
rq2_self_esteem["self_image_mean"] = (
    rq2_self_esteem["online_self_image_score_z"]
    + rq2_self_esteem["offline_self_image_score_z"]
) / 2

rq2_self_esteem["self_image_difference"] = (
    rq2_self_esteem["online_self_image_score_z"]
    - rq2_self_esteem["offline_self_image_score_z"]
)

model_self_esteem_mean_diff = smf.wls(
    formula=(
        "self_esteem_score_z ~ "
        "self_image_mean + "
        "self_image_difference + "
        "female + bs3_z"
    ),
    data=rq2_self_esteem,
    weights=rq2_self_esteem["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": rq2_self_esteem["nschool_id"]
    }
)

print("\nSelf-esteem Mean-Difference Model")
print(
    model_self_esteem_mean_diff.params[
        ["self_image_mean", "self_image_difference"]
    ].round(4)
)
for data in [rq2_wellbeing, rq2_depression]:
    data["self_image_mean"] = (
        data["online_self_image_score_z"]
        + data["offline_self_image_score_z"]
    ) / 2

    data["self_image_difference"] = (
        data["online_self_image_score_z"]
        - data["offline_self_image_score_z"]
    )


model_wellbeing_mean_diff = smf.wls(
    formula=(
        "wellbeing_score_z ~ "
        "self_image_mean + self_image_difference + "
        "female + bs3_z"
    ),
    data=rq2_wellbeing,
    weights=rq2_wellbeing["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={"groups": rq2_wellbeing["nschool_id"]}
)


model_depression_mean_diff = smf.wls(
    formula=(
        "depression_score_z ~ "
        "self_image_mean + self_image_difference + "
        "female + bs3_z"
    ),
    data=rq2_depression,
    weights=rq2_depression["HOUWGT"]
).fit(
    cov_type="cluster",
    cov_kwds={"groups": rq2_depression["nschool_id"]}
)


print("\nWell-being Mean-Difference Model")
print(
    model_wellbeing_mean_diff.params[
        ["self_image_mean", "self_image_difference"]
    ].round(4)
)

print("\nDepression Mean-Difference Model")
print(
    model_depression_mean_diff.params[
        ["self_image_mean", "self_image_difference"]
    ].round(4)
)
#在相同整體 self-image 水準下，self-image 越呈現 Online > Offline 的相對配置，與較低 Self-esteem、較低 Well-being、較高 Depressive symptoms 有關。

# ============================================================
# RQ3（修改版）：Online／Offline 自我形象「哪一邊比較高」與心理狀態
# ------------------------------------------------------------
# 取代原本的 RQ3 性別調節分析。
#
# 依 Online − Offline 原始分數（1–4 量尺）把學生分成三組：
#   Online > Offline：線上自我形象高於線下
#   Equal           ：兩者相同（作為比較基準）
#   Offline > Online：線下自我形象高於線上
#
# RQ3A｜三組之間的心理狀態是否不同？
#   outcome_z ~ 落差組別 (ref = Equal) + 整體自我形象水準 + female + bs3
#
# RQ3B｜在同一個方向上，落差越大，心理狀態是否越好／越差？
#   (1) Online > Offline：樣本 = Equal + Online > Offline
#   (2) Offline > Online：樣本 = Equal + Offline > Online
#   outcome_z ~ 落差大小（原始分數，0–3）+ 整體自我形象水準 + female + bs3
#
# 控制「整體自我形象水準」= (Online + Offline) / 2，
# 讓係數代表「在同樣的整體自我形象下，偏向某一邊」的關聯，
# 而不是單純反映「線下自我形象高的人心理比較好」。
#
# 全部採 HOUWGT 加權＋學校層級 cluster-robust SE（與 RQ1、RQ2 相同）。
# ============================================================

rq3_outcomes = {
    "Self-esteem": "self_esteem_score",
    "Well-being": "wellbeing_score",
    "Depressive symptoms": "depression_score",
}

rq3_base = topic1[[
    "online_self_image_score",
    "offline_self_image_score",
    "self_esteem_score",
    "wellbeing_score",
    "depression_score",
    "female",
    "bs3",
    "HOUWGT",
    "nschool_id",
]].dropna(subset=[
    "online_self_image_score",
    "offline_self_image_score",
    "female", "bs3", "HOUWGT", "nschool_id",
]).copy()

# 落差（原始量尺；三題平均相減，四捨五入避免浮點誤差）
rq3_base["gap"] = (
    rq3_base["online_self_image_score"]
    - rq3_base["offline_self_image_score"]
).round(4)

rq3_base["gap_group"] = np.select(
    [rq3_base["gap"] > 0, rq3_base["gap"] < 0],
    ["Online > Offline", "Offline > Online"],
    default="Equal",
)
rq3_base["gap_size"] = rq3_base["gap"].abs()

# 整體自我形象水準
rq3_base["self_image_level"] = (
    rq3_base["online_self_image_score"]
    + rq3_base["offline_self_image_score"]
) / 2

print("\n" + "=" * 60)
print("RQ3 (revised): Online–Offline self-image direction")
print("=" * 60)
print("\nGroup sizes:")
print(
    rq3_base["gap_group"]
    .value_counts()
    .reindex(["Online > Offline", "Equal", "Offline > Online"])
)


def zscore(x):
    return (x - x.mean()) / x.std()


def fit_weighted_cluster(formula, data):
    return smf.wls(
        formula=formula,
        data=data,
        weights=data["HOUWGT"],
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["nschool_id"]},
    )


def prepare_rq3_sample(data, outcome):
    """Complete cases for one outcome; standardize within that sample."""
    sample = data.dropna(subset=[outcome]).copy()
    sample["outcome_z"] = zscore(sample[outcome])
    sample["level_z"] = zscore(sample["self_image_level"])
    sample["bs3_z"] = zscore(sample["bs3"])
    return sample


# ------------------------------------------------------------
# RQ3A：三組比較（Equal 為基準）
# ------------------------------------------------------------
print("\n--- RQ3A: Group differences vs. Equal (standardized outcome) ---")

group_term = 'C(gap_group, Treatment(reference="Equal"))'
rq3a_rows = []

for label, outcome in rq3_outcomes.items():
    sample = prepare_rq3_sample(rq3_base, outcome)
    model = fit_weighted_cluster(
        f"outcome_z ~ {group_term} + level_z + female + bs3_z",
        sample,
    )
    for grp in ["Online > Offline", "Offline > Online"]:
        term = f"{group_term}[T.{grp}]"
        ci = model.conf_int().loc[term]
        rq3a_rows.append({
            "Outcome": label,
            "Group (vs. Equal)": grp,
            "b": model.params[term],
            "CI_low": ci[0],
            "CI_high": ci[1],
            "p": model.pvalues[term],
            "N": int(model.nobs),
        })

rq3a_table = pd.DataFrame(rq3a_rows)
print(rq3a_table.round(4).to_string(index=False))

# ------------------------------------------------------------
# RQ3B：同方向內，落差大小 → 心理狀態
# ------------------------------------------------------------
rq3_directions = {
    "Online > Offline": ["Equal", "Online > Offline"],
    "Offline > Online": ["Equal", "Offline > Online"],
}


def run_rq3b(include_equal=True, control_level=True):
    rows = []
    for direction, groups in rq3_directions.items():
        keep = groups if include_equal else [direction]
        subset = rq3_base[rq3_base["gap_group"].isin(keep)]
        for label, outcome in rq3_outcomes.items():
            sample = prepare_rq3_sample(subset, outcome)
            rhs = "gap_size + female + bs3_z"
            if control_level:
                rhs += " + level_z"
            model = fit_weighted_cluster(f"outcome_z ~ {rhs}", sample)
            ci = model.conf_int().loc["gap_size"]
            rows.append({
                "Direction": direction,
                "Outcome": label,
                "b_per_1pt_gap": model.params["gap_size"],
                "CI_low": ci[0],
                "CI_high": ci[1],
                "p": model.pvalues["gap_size"],
                "N": int(model.nobs),
            })
    return pd.DataFrame(rows)


print("\n--- RQ3B (main): gap size → outcome_z, Equal group as 0 anchor ---")
print("    controls: overall self-image level, female, SES")
rq3b_main = run_rq3b(include_equal=True, control_level=True)
print(rq3b_main.round(4).to_string(index=False))

# 敏感度 1：只用該方向的學生（不含 Equal）
print("\n--- RQ3B sensitivity 1: within-direction only (Equal excluded) ---")
print(run_rq3b(include_equal=False, control_level=True).round(4).to_string(index=False))

# 敏感度 2：不控制整體自我形象水準
print("\n--- RQ3B sensitivity 2: without controlling overall self-image level ---")
print(run_rq3b(include_equal=True, control_level=False).round(4).to_string(index=False))

# ------------------------------------------------------------
# 描述統計：各組加權平均（原始量尺）
# ------------------------------------------------------------
print("\n--- Weighted descriptive means by group (original scales) ---")


def weighted_mean(x, w):
    ok = x.notna()
    return np.average(x[ok], weights=w[ok])


desc = (
    rq3_base
    .groupby("gap_group")
    .apply(lambda g: pd.Series({
        label: weighted_mean(g[outcome], g["HOUWGT"])
        for label, outcome in rq3_outcomes.items()
    } | {
        "Online": weighted_mean(g["online_self_image_score"], g["HOUWGT"]),
        "Offline": weighted_mean(g["offline_self_image_score"], g["HOUWGT"]),
        "n": len(g),
    }), include_groups=False)
    .reindex(["Online > Offline", "Equal", "Offline > Online"])
)
print(desc.round(3).to_string())