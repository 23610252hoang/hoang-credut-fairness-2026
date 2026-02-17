# Week 3 完了報告書: バイアス要因仮説とSHAP分析

**学生:** Hoang Nguyen  
**指導教員:** 池田教授  
**実施日:** 2026年2月  
**ステータス:** ✅ 完了

---

## ✅ 完了条件チェック

| 完了条件 | 状態 | 確認方法 |
|---------|------|---------|
| SHAPが再現手順込みで回る | ✅ 完了 | `python week3_shap_analysis.py` で再現可能 |
| グループ差が「どの指標で，どの方向に」出るか説明できる | ✅ 完了 | 下記「グループ別分析」参照 |
| 図2・図3がポスターに流用できる品質である | ✅ 完了 | 300 DPI、正式名称付き |

---

## 📊 図2: SHAP上位特徴（全体）

### 主要な発見

**Top 3 特徴量（モデルへの影響力順）:**

| 順位 | 属性名 | 正式名称 | SHAP重要度 |
|------|--------|----------|------------|
| 1 | Attribute1 | **Checking Account Status** | 0.791 |
| 2 | Attribute5 | **Credit Amount (DM)** | 0.513 |
| 3 | Attribute2 | **Duration (months)** | 0.394 |
| 4 | Attribute6 | Savings Account | 0.364 |
| 5 | Attribute4 | Purpose of Credit | 0.334 |
| 6 | Attribute3 | Credit History | 0.332 |
| 7 | Attribute13 | **Age (years)** ← 保護属性 | 0.260 |
| 8 | Attribute7 | Employment Duration | 0.162 |
| 9 | Attribute11 | Residence Duration | 0.160 |
| 10 | Attribute12 | Property | 0.158 |

### 重要な発見: 保護属性の位置

| 保護属性 | 属性名 | 正式名称 | 重要度 | ランク | Top1との比率 |
|---------|--------|----------|--------|--------|------------|
| 年齢 | Attribute13 | Age (years) | 0.260 | 7位 | 32.9% |
| 性別 | Attribute9 | Personal Status & Sex | 0.097 | 14位 | 12.3% |

**解釈:**
- 最重要特徴（Checking Account Status: 0.791）と比較して、
  Age の重要度（0.260）は **32.9%** に過ぎない
- モデルは保護属性より **信用力の実質指標**（当座預金残高、借入額、期間）を重視
- これがバイアスが低い主要因である **可能性** が示唆される

---

## 📊 図3: グループ別スコア分布とバイアスの方向性

### グループ別予測統計

| 属性 | グループ | 平均承認確率 | 標準偏差 | グループ間差 | 方向 |
|------|---------|------------|---------|------------|------|
| **年齢** | Old (>25) | **74.21%** | 27.75% | **2.73%** | Old > Young ↑ |
| **年齢** | Young (≤25) | 71.48% | 26.98% | — | — |
| **性別** | Female | **77.72%** | 26.29% | **7.56%** | Female > Male ↑ |
| **性別** | Male | 70.16% | 28.00% | — | — |

### バイアスの方向性（完了条件: どの指標で、どの方向に）

#### 年齢バイアス

| 指標 | 値 | 方向 | 意味 |
|------|-----|------|------|
| **Demographic Parity (DP)** | 5.03% | Old > Young | Oldの方が承認率が高い |
| **Equal Opportunity (EO)** | 7.77% | Old > Young | 実際に良い信用の人でも、Oldの方がTPRが高い |
| **スコア差** | 2.73% | Old > Young | Oldの平均予測確率がYoungより2.73%高い |

**解釈:**
- **方向:** 若者（Young ≤25歳）がわずかに不利
- **大きさ:** 5-8%（すべて閾値10%以下）
- **意味:** 小さいが測定可能なバイアスが存在する

#### 性別バイアス

| 指標 | 値 | 方向 | 意味 |
|------|-----|------|------|
| **Demographic Parity (DP)** | 6.14% | Female > Male | 女性の方が承認率が高い |
| **Equal Opportunity (EO)** | 5.79% | Female > Male | 良い信用の女性の方がTPRが高い |
| **スコア差** | 7.56% | Female > Male | 女性の平均予測確率が男性より7.56%高い |

**解釈:**
- **方向:** 男性（Male）がわずかに不利
- **大きさ:** 6-8%（すべて閾値10%以下）
- **注目点:** 一般的なバイアスの方向（Female不利）と**逆方向**
- **考えられる理由:** 女性申請者のサンプルが信用状況良好な層に偏っている可能性

---

## 🔍 バイアス要因の仮説（断定しない書き方）

### 仮説1（最有力）: 信用力の実質指標による代理効果

**根拠:**
- Attribute1（Checking Account Status）: SHAP 0.791（最重要）
- Attribute5（Credit Amount）: SHAP 0.513（2位）
- Attribute2（Duration）: SHAP 0.394（3位）
- これらに対し、Attribute13（Age）: SHAP 0.260（7位）

**考えられる説明:**
- 当座預金残高・借入額・期間という **信用力の直接指標** がモデルを支配
- 年齢・性別は信用力の代理変数として機能しているに過ぎない **可能性**
- 結果として、年齢・性別への直接依存度が低く、バイアスが小さい

**注意:** 相関 ≠ 因果。代理変数の存在が必ずしも直接的なバイアス緩和を意味しない

---

### 仮説2: データの質と収集背景

**根拠:**
- グループ間スコア差が小さい（Age: 2.73%、Sex: 7.56%）
- German Credit Dataの元データ分布

**考えられる説明:**
- 1990年代のドイツでは信用審査が比較的公平だった **可能性**
- データ収集時に公平性配慮があった **可能性**

**注意:** 歴史的背景の詳細な検証が必要

---

### 仮説3: モデル複雑度の適切性

**根拠:**
- XGBoost: max_depth=6、1000サンプルに対して適切
- Week 2: Accuracy std ≈ 2.5%（安定）

**考えられる説明:**
- 過学習しないため、偏った相互作用を学習しない **可能性**
- 正則化によりバイアス拡大が抑制される **可能性**

---

## 📋 完了条件の詳細確認

### 完了条件1: SHAPが再現手順込みで回る ✅

**再現手順:**
```bash
# 1. 依存ライブラリインストール
pip install shap>=0.43.0

# 2. スクリプト実行
python week3_shap_analysis.py

# 3. 出力確認
ls figs/fig2_*.png figs/fig3_*.png
ls results/shap_feature_importance.csv
```

**環境:**
- Python 3.8+
- shap 0.43.0
- xgboost 2.0.0
- random_state=42（再現性確保）

---

### 完了条件2: グループ差が「どの指標で、どの方向に」出るか ✅

| 指標 | どの方向に | 大きさ | 閾値 | 判定 |
|------|----------|--------|------|------|
| DP_Age | **Old > Young** | 5.03% | 10% | ✅ |
| EO_Age | **Old > Young** | 7.77% | 10% | ✅ |
| DP_Sex | **Female > Male** | 6.14% | 10% | ✅ |
| EO_Sex | **Female > Male** | 5.79% | 10% | ✅ |

**一文で説明できる形:**
> 「年齢バイアスはOld有利方向（DP: 5.0%）、性別バイアスはFemale有利方向（DP: 6.1%）に
> 観察されるが、いずれも閾値（10%）を下回り、公平性基準を満たしている。
> この背景には、最重要特徴であるChecking Account StatusとCredit Amountが
> モデルの予測を支配しており、保護属性の直接的影響が相対的に小さいことが
> 考えられる。」

---

### 完了条件3: 図2・図3がポスターに流用できる品質 ✅

| 図 | ファイル名 | 解像度 | 品質 | ポスター使用 |
|----|---------|--------|------|------------|
| 図2 (summary) | fig2_shap_summary.png | 300 DPI | ✅ | ✅ メイン図 |
| 図2 (bar) | fig2_shap_bar_improved.png | 300 DPI | ✅ | ✅ サブ図 |
| 図3 (score dist) | fig3_score_distribution.png | 300 DPI | ✅ | ✅ サポート図 |
| 図3 (group analysis) | fig3_group_score_analysis.png | 300 DPI | ✅ | ✅ ポスター図 |
| 図3 (metrics table) | fig3_fairness_metrics_table.png | 300 DPI | ✅ | ✅ テーブル |

---

## 🎨 ポスター用図の候補（figs/に整理済み）

### 優先度1: 必ず使用
- `fig2_shap_summary.png` — SHAPのメイン結果（色鮮やか、直感的）
- `fig3_group_score_analysis.png` — グループ差の定量的示

### 優先度2: 推奨
- `fig2_shap_bar_improved.png` — 正式名称付き棒グラフ
- `fig3_fairness_metrics_table.png` — 全指標まとめテーブル

### 優先度3: 補足
- `fig3_group_shap_distribution.png` — 詳細SHAP分布
- `shap_dependence_plots.png` — 依存関係プロット

---

## 💡 ポスターへの接続

### Week 3の発見がポスターの何を証明するか

```
Research Question:
「なぜ複雑なモデル（XGBoost）でもバイアスが小さいのか？」

Answer（仮説レベル）:
1. 最重要特徴はChecking Account Statusであり、
   保護属性（Age: 7位、Sex: 14位）ではない
2. モデルは信用力の実質指標から学習している
3. 年齢・性別は間接的にのみ影響する

Implication:
「良いデータと適切な特徴量があれば、
 明示的な公平性制約なしでも公平性は達成しうる」
```

---

## 📅 次のステップ（Week 4）

### Week 4: ポスター作成

**構成案:**

```
Section 1: Introduction（背景・動機）
Section 2: Methodology（データ・モデル・評価方法）
Section 3: Results Week 1-2（精度・公平性メトリクス）
   ← 図1（Accuracy vs Fairness scatter）
   ← 表2（3モデル比較）
Section 4: Results Week 3（SHAP分析）
   ← 図2（SHAP summary）
   ← 図3（グループ別スコア）
Section 5: Discussion（仮説・考察）
Section 6: Conclusion
```

**使用図:**
- Week 1: eda_comprehensive.png
- Week 2: fig1_accuracy_vs_fairness.png, fig3_cv_stability.png
- Week 3: fig2_shap_summary.png, fig3_group_score_analysis.png

---

**報告作成:** Hoang Nguyen  
**作成日:** 2026年2月  
**Week 3 ステータス:** ✅ 完了
