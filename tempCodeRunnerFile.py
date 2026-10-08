#RQ2D 權重穩健性
# 1. 準備加權分析所需的變數
rq2_weighted_vars = [
    "selective_presentation_score",
    "self_image_gap",
    "female",
    "bs3",
    "HOUWGT",
    "nschool_id"
]

rq2_weighted_data = (
    topic1[rq2_weighted_vars]
    .dropna()
    .copy()
)

# 2. 建立與原本 MNLogit 相同的三分類
# 0 = Online < Offline (reference)
# 1 = Online = Offline
# 2 = Online > Offline

rq2_weighted_data["image_group"] = np.select(
    [
        rq2_weighted_data["self_image_gap"] < -rq2_gap_tolerance,
        rq2_weighted_data["self_image_gap"] > rq2_gap_tolerance
    ],
    [0, 2],
    default=1
).astype(int)

# 3. 檢查樣本與權重
print("\n=== RQ2D: Weighted Sample Check ===")
print("Original MNLogit N:", len(rq2_model_data))
print("Weighted sample N:", len(rq2_weighted_data))

print("\nWeight descriptives:")
print(rq2_weighted_data["HOUWGT"].describe().round(3))

print("\nImage group counts:")
print(
    rq2_weighted_data["image_group"]
    .value_counts()
    .sort_index()
)

print(
    "\nNumber of school clusters:",
    rq2_weighted_data["nschool_id"].nunique()
)