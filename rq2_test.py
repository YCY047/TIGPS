import pandas as pd
import numpy as np

df=pd.read_csv("TIGPSw2_s.csv",
    low_memory=False)
items = ["bs25j", "bs25k", "bs25l"] #檢查bs25j-l能否組成線上/線下不一致感量表

# 複製需要的欄位
discrepancy = df[items].copy()

# 排除 TIGPS 特殊遺漏碼及無效值
discrepancy = discrepancy.where(
    discrepancy.isin([1, 2, 3, 4])
)

print("=== Descriptive Statistics ===")
print(discrepancy.describe())

print("\n=== Correlation Matrix ===")
print(discrepancy.corr())

# 僅使用三題皆有效的受訪者
complete = discrepancy.dropna()

def cronbach_alpha(data):
    k = data.shape[1]
    item_variances = data.var(axis=0, ddof=1)
    total_variance = data.sum(axis=1).var(ddof=1)

    return (k / (k - 1)) * (
        1 - item_variances.sum() / total_variance
    )

print("\nValid N:", len(complete))
print("Cronbach's alpha:", round(cronbach_alpha(complete), 3))


# 主觀自我不一致感量表
df["perceived_discrepancy"] = discrepancy.mean(
    axis=1, skipna=False
)
# Online / Offline 正向自我形象
online_items = ["bs25a", "bs25b", "bs25c"]
offline_items = ["bs25d", "bs25e", "bs25f"]

self_image = df[online_items + offline_items].copy()
self_image = self_image.where(
    self_image.isin([1, 2, 3, 4])
)

df["online_image"] = self_image[online_items].mean(
    axis=1, skipna=False
)

df["offline_image"] = self_image[offline_items].mean(
    axis=1, skipna=False
)
# 有方向的差異
df["image_difference"] = (
    df["online_image"] - df["offline_image"]
)

# 兩種指標之間的相關
print(
    df[["image_difference", "perceived_discrepancy"]]
    .corr()
)

# 描述統計
print(
    df[["image_difference", "perceived_discrepancy"]]
    .describe()
)

# Selective Positive Self-presentation
selective_items = ["bs23a", "bs23b", "bs23c"]

selective_data = df[selective_items].copy()
selective_data = selective_data.where(
    selective_data.isin([1, 2, 3, 4])
)

df["selective"] = selective_data.mean(
    axis=1, skipna=False
)

# 三個核心變數的相關矩陣
variables = [
    "selective",
    "image_difference",
    "perceived_discrepancy"
]

print(df[variables].corr())

print("\nValid N:")
print(df[variables].dropna().shape[0])

import numpy as np
import pandas as pd

data = df[
    ["selective", "image_difference"]
].dropna().copy()

# 依照 Online - Offline 分成三組
data["image_group"] = np.select(
    [
        data["image_difference"] > 1e-9,
        data["image_difference"] < -1e-9
    ],
    [
        "Online > Offline",
        "Online < Offline"
    ],
    default="Online = Offline"
)

# 各組的 Selective 平均分數
summary = data.groupby("image_group").agg(
    n=("selective", "size"),
    selective_mean=("selective", "mean"),
    selective_sd=("selective", "std"),
    difference_mean=("image_difference", "mean")
)

summary["percentage"] = (
    summary["n"] / len(data) * 100
)

summary = summary.reindex([
    "Online > Offline",
    "Online = Offline",
    "Online < Offline"
])

print(summary.round(3))

# 使用前面已經建立好的變數
data = df[["selective", "image_difference"]].dropna().copy()

# Selective 分成三組
data["selective_group"] = pd.cut(
    data["selective"],
    bins=[0, 2, 3, 4],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# 線上與線下自我形象分成三組
data["image_group"] = np.select(
    [
        data["image_difference"] > 1e-9,
        data["image_difference"] < -1e-9
    ],
    [
        "Online > Offline",
        "Online < Offline"
    ],
    default="Online = Offline"
)

# 交叉表：人數
counts = pd.crosstab(
    data["selective_group"],
    data["image_group"]
)

# 交叉表：各 Selective 組內的百分比
percentages = pd.crosstab(
    data["selective_group"],
    data["image_group"],
    normalize="index"
) * 100

print("=== Counts ===")
print(counts)

print("\n=== Row Percentages (%) ===")
print(percentages.round(2))

import numpy as np
import pandas as pd
import statsmodels.api as sm

cols = [
    "selective",
    "image_difference",
    "bs1",
    "bs3",
    "nschool_id"
]

model_data = df[cols].copy()

# 清理控制變數
model_data["bs1"] = pd.to_numeric(
    model_data["bs1"], errors="coerce"
)
model_data["bs3"] = pd.to_numeric(
    model_data["bs3"], errors="coerce"
)

model_data["bs1"] = model_data["bs1"].where(
    model_data["bs1"].isin([1, 2])
)
model_data["bs3"] = model_data["bs3"].where(
    model_data["bs3"].between(1, 10)
)

model_data = model_data.dropna()

# 0 = Online = Offline
# 1 = Online > Offline
# 2 = Online < Offline
model_data["image_group"] = np.select(
    [
        model_data["image_difference"] > 1e-9,
        model_data["image_difference"] < -1e-9
    ],
    [1, 2],
    default=0
)

# 女生 = 1，男生 = 0
model_data["female"] = (
    model_data["bs1"] == 1
).astype(int)

X = sm.add_constant(
    model_data[["selective", "female", "bs3"]]
)

y = model_data["image_group"]

model = sm.MNLogit(y, X)

result = model.fit(
    method="newton",
    maxiter=200,
    disp=False,
    cov_type="cluster",
    cov_kwds={
        "groups": model_data["nschool_id"]
    }
)

print("Valid N:", len(model_data))
print(result.summary())

# Relative Risk Ratios (RRR)
print("\n=== Relative Risk Ratios ===")
print(np.exp(result.params))

# 改以 Online < Offline 作為參照組
# 0 = Online < Offline
# 1 = Online = Offline
# 2 = Online > Offline

model_data["image_group_new"] = (
    model_data["image_group"].map({
        2: 0,
        0: 1,
        1: 2
    })
)

model2 = sm.MNLogit(
    model_data["image_group_new"], X
)

result2 = model2.fit(
    method="newton",
    maxiter=200,
    disp=False,
    cov_type="cluster",
    cov_kwds={
        "groups": model_data["nschool_id"]
    }
)

print(result2.summary())
print("\nRRR:")
print(np.exp(result2.params))