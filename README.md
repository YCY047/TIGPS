# Topic 1 分析流程

## Step 1｜資料整理與變項確認
- 讀取 TIGPS Wave 2 學生資料。
- 確認研究所需題項與實際欄位名稱。
- 處理問卷特殊遺漏值與無效回答。
- 確認數位使用、自我呈現、自我形象、心理結果、性別與家庭社會階層等變項。

---

## Step 2｜量表建構與信度檢驗
建立主要研究構念，並以 Cronbach's α、item-total correlation 與題目相關檢查量表品質。

主要構念包括：
- Selective Positive Self-presentation（bs23a–c）
- Online Positive Self-image（bs25a–c）
- Offline Positive Self-image（bs25d–f）
- Self-esteem（bs53a–c）
- Well-being
- Depressive symptoms（bs56a–n）

通過檢驗後，以題目平均分數建立各量表分數。

---

## Step 3｜探索性分析
初步了解資料結構與變項之間的關係，包括：
- 描述統計
- 相關分析
- 不同數位使用活動與自我呈現／自我形象的關聯
- 性別差異
- Online–Offline self-image gap
- 心理結果之間的關係

此階段主要用於確認後續正式研究模型，不作為最終推論模型。

---

## Step 4｜正式 RQ1：數位行為與自我形象
檢驗不同數位使用活動與 Selective Positive Self-presentation 是否與：

- Online Positive Self-image
- Offline Positive Self-image

相關。

所有主要連續變項標準化，使不同 predictor 的迴歸係數可以直接比較。

---

## Step 5｜正式 RQ2：線上／線下自我形象與心理適應
將 Online 與 Offline Positive Self-image 同時納入模型，檢驗其與：

- Self-esteem
- Well-being
- Depressive symptoms

之間的條件關聯。

另外直接檢定 Online 與 Offline 兩個迴歸係數是否存在差異。

---

## Step 6｜正式 RQ3：性別調節效果
檢驗 Gender 是否調節兩類關係：

### RQ3A
Selective Positive Self-presentation  
→ Online / Offline Positive Self-image

### RQ3B
Online / Offline Positive Self-image  
→ Self-esteem / Well-being / Depressive symptoms

---

## Step 7｜加入背景控制變項
在 RQ1–RQ3 模型中加入：

- Gender
- Subjective Family SES（bs3）

檢查主要結果在控制基本背景差異後是否維持。

---

## Step 8｜抽樣權重與群聚調整
檢查 TIGPS 的抽樣與權重變項，正式模型採用：

- HOUWGT 加權
- School-level cluster-robust standard errors（nschool_id）

重新估計 RQ1–RQ3，作為主要正式結果。

---

## Step 9｜Robustness / Sensitivity Analysis
進行多項敏感度分析，確認主要結果是否依賴特定量表或模型設定。

### 9A｜移除 bs25b
從 Online Positive Self-image 移除可能與自我呈現概念重疊的 bs25b，重新估計 RQ1。

### 9B｜移除 bs23c
從 Selective Positive Self-presentation 移除較極端的 bs23c，重新估計 RQ1。

### 9C｜Suppression Check
分別比較：
- Online only
- Offline only
- Online + Offline

確認 RQ2 的 Online coefficient 應解讀為控制 Offline self-image 後的 conditional association，而非單純的 Online self-image 效果。

### 9D｜Online–Offline Reparameterization
將 Online / Offline self-image 重新表示為：

- Overall Self-image Level
- Online–Offline Directional Difference

用來區分「整體自我形象水準」與「自我形象相對偏向 Online 或 Offline」。

---

## Step 10｜Interaction Simple Slopes
針對 RQ3 的 Gender interaction 計算男女各自的 simple slopes 與 95% CI，使調節效果更容易解釋。

---

## Step 11｜結果視覺化
使用 `topic1_figures.py` 產生三張主要結果圖：

1. **Figure 1：** Digital-use activities / Selective presentation → Online / Offline self-image
2. **Figure 2：** Online / Offline self-image → Psychological outcomes
3. **Figure 3：** Gender moderation of Selective presentation → Self-image