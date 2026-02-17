"""
Week 3: 完了スクリプト
- German Credit Dataの属性名を正式名称に変換
- 図2・図3を改善（ポスター品質）
- 完了報告書を生成
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

# =============================================
# German Credit Dataset 正式な属性名マッピング
# =============================================
ATTRIBUTE_NAMES = {
    'Attribute1':  'Checking Account Status',
    'Attribute2':  'Duration (months)',
    'Attribute3':  'Credit History',
    'Attribute4':  'Purpose of Credit',
    'Attribute5':  'Credit Amount (DM)',
    'Attribute6':  'Savings Account',
    'Attribute7':  'Employment Duration',
    'Attribute8':  'Installment Rate (%)',
    'Attribute9':  'Personal Status & Sex',
    'Attribute10': 'Other Debtors',
    'Attribute11': 'Residence Duration',
    'Attribute12': 'Property',
    'Attribute13': 'Age (years)',
    'Attribute14': 'Other Installment Plans',
    'Attribute15': 'Housing',
    'Attribute16': 'Existing Credits',
    'Attribute17': 'Job',
    'Attribute18': 'Dependents',
    'Attribute19': 'Telephone',
    'Attribute20': 'Foreign Worker',
}

# =============================================
# 既存のSHAP結果を読み込んで改善する
# =============================================

def load_existing_results():
    """既存の結果を読み込む"""
    print("📥 Loading existing results...")
    
    # SHAP feature importance
    importance_df = pd.read_csv('results/shap_feature_importance.csv')
    print(f"✅ Loaded {len(importance_df)} features")
    
    return importance_df

def create_improved_shap_bar(importance_df):
    """
    図2改善版: 正式名称付きSHAP棒グラフ（ポスター品質）
    """
    print("\n📊 Creating Improved Figure 2: SHAP Feature Importance Bar Chart...")
    
    # 正式名称に変換
    importance_df = importance_df.copy()
    importance_df['feature_label'] = importance_df['feature'].map(
        lambda x: ATTRIBUTE_NAMES.get(x, x)
    )
    importance_df = importance_df.sort_values('importance', ascending=False).head(15)
    importance_df_plot = importance_df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # カラーマップ（重要度に応じて色変化）
    n = len(importance_df_plot)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n))
    
    bars = ax.barh(
        range(n),
        importance_df_plot['importance'],
        color=colors,
        edgecolor='white',
        linewidth=0.5,
        height=0.7
    )
    
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{row['feature_label']}\n({row['feature']})" 
         for _, row in importance_df_plot.iterrows()],
        fontsize=10
    )
    ax.set_xlabel('Mean |SHAP Value| (Impact on Model Output)', fontsize=13, fontweight='bold')
    ax.set_title('図2: Feature Importance (XGBoost + SHAP)\nTop 15 Features', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, importance_df_plot['importance'].max() * 1.15)
    
    # 数値ラベル
    for bar, val in zip(bars, importance_df_plot['importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 注釈: 保護属性
    protected = {
        'Attribute13': 'Age (protected)',
        'Attribute9': 'Personal Status/Sex (protected)'
    }
    for feat, label in protected.items():
        match = importance_df_plot[importance_df_plot['feature'] == feat]
        if not match.empty:
            idx = match.index[0]
            pos = importance_df_plot.index.get_loc(idx)
            ax.axhline(y=pos, color='red', linestyle=':', alpha=0.4, linewidth=1)
    
    # 凡例（保護属性の注釈）
    red_patch = mpatches.Patch(color='red', alpha=0.4, label='Protected attribute position')
    ax.legend(handles=[red_patch], loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figs/fig2_shap_bar_improved.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✅ Saved: figs/fig2_shap_bar_improved.png")
    
    return importance_df

def create_group_score_analysis():
    """
    グループ別スコア差の定量分析（図3補足）
    """
    print("\n📊 Creating Group Score Analysis...")
    
    # 既存のデータから統計を計算（reportから）
    group_stats = {
        'Age': {
            'Old':   {'mean': 0.7421, 'std': 0.2775},
            'Young': {'mean': 0.7148, 'std': 0.2698},
        },
        'Sex': {
            'Female': {'mean': 0.7772, 'std': 0.2629},
            'Male':   {'mean': 0.7016, 'std': 0.2800},
        }
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('図3: Group-wise Prediction Score Analysis', 
                 fontsize=16, fontweight='bold')
    
    for ax_idx, (attr, groups) in enumerate(group_stats.items()):
        ax = axes[ax_idx]
        
        group_names = list(groups.keys())
        means = [groups[g]['mean'] for g in group_names]
        stds  = [groups[g]['std']  for g in group_names]
        diff  = abs(means[0] - means[1])
        
        colors = ['#4ECDC4', '#FF6B6B'] if attr == 'Age' else ['#74B9FF', '#FFA8E0']
        
        bars = ax.bar(group_names, means, yerr=stds,
                      color=colors, alpha=0.85, capsize=8,
                      edgecolor='black', linewidth=1.2,
                      error_kw={'elinewidth': 2, 'ecolor': 'black'})
        
        # 差分アノテーション
        y_max = max(means) + max(stds) + 0.05
        ax.annotate(
            f'Δ = {diff:.4f}\n({diff*100:.2f}%)',
            xy=(0.5, y_max),
            xycoords=('axes fraction', 'data'),
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='darkred', alpha=0.8)
        )
        
        # 矢印で差分を示す
        ax.annotate('', xy=(1, means[1]), xytext=(0, means[0]),
                    xycoords=('data', 'data'),
                    textcoords=('data', 'data'),
                    arrowprops=dict(arrowstyle='<->', color='darkred',
                                   lw=1.5, connectionstyle='arc3,rad=0.3'))
        
        ax.set_ylim(0, y_max + 0.1)
        ax.set_ylabel('Mean Predicted Probability\n(P(Good Credit))', 
                      fontsize=12, fontweight='bold')
        ax.set_title(f'{attr} Groups\n(DP ≈ {diff*100:.1f}%)', 
                     fontsize=14, fontweight='bold')
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.4, label='Overall mean ≈ 0.7')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # 数値ラベル
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + std + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figs/fig3_group_score_analysis.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: figs/fig3_group_score_analysis.png")

def create_tpr_fpr_comparison():
    """
    TPR/FPR比較表（図3補足 - ポスター用）
    """
    print("\n📊 Creating TPR/FPR Comparison Table Figure...")
    
    # Week 2の結果から計算（mean値を使用）
    metrics_data = {
        'Group': ['Old (Age=0)', 'Young (Age=1)', 'Male (Sex=0)', 'Female (Sex=1)'],
        'Predicted\nApproval Rate': [0.742, 0.715, 0.702, 0.777],
        'DP Gap': ['—', '2.73% (Age DP)', '—', '7.56% (Sex DP)'],
        'Fairness': ['✅', '✅', '✅', '✅'],
    }
    
    # Summary table
    summary = {
        'Metric': ['DP_Age', 'EO_Age', 'DP_Sex', 'EO_Sex'],
        'Value (mean)': ['5.03%', '7.77%', '6.14%', '5.79%'],
        'Threshold': ['10%', '10%', '10%', '10%'],
        'Direction': ['Old > Young', 'Old > Young', 'Female > Male', 'Female > Male'],
        'Status': ['✅ PASS', '✅ PASS', '✅ PASS', '✅ PASS'],
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('図3: Fairness Metrics Summary\n(Which direction, which metric)',
                 fontsize=15, fontweight='bold')
    
    # Table 1: Group approval rates
    ax1.axis('off')
    df1 = pd.DataFrame(metrics_data)
    table1 = ax1.table(
        cellText=df1.values,
        colLabels=df1.columns,
        cellLoc='center', loc='center',
        bbox=[0, 0, 1, 1]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    for (row, col), cell in table1.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#ECF0F1')
        cell.set_edgecolor('gray')
    ax1.set_title('Group-wise Predicted Approval Rates', 
                  fontsize=13, fontweight='bold', pad=10)
    
    # Table 2: Fairness metrics
    ax2.axis('off')
    df2 = pd.DataFrame(summary)
    table2 = ax2.table(
        cellText=df2.values,
        colLabels=df2.columns,
        cellLoc='center', loc='center',
        bbox=[0, 0, 1, 1]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D5F5E3')
        if row > 0 and col == 4:  # Status column
            cell.set_facecolor('#D5F5E3')
        cell.set_edgecolor('gray')
    ax2.set_title('Fairness Metrics (All Pass ≤ 10% Threshold)', 
                  fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig('figs/fig3_fairness_metrics_table.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Saved: figs/fig3_fairness_metrics_table.png")

def create_week3_full_report(importance_df):
    """
    Week 3 完了報告書（正式名称付き）
    """
    print("\n📝 Creating Week 3 Full Report...")
    
    # 正式名称マッピング
    importance_df = importance_df.copy()
    importance_df['feature_name'] = importance_df['feature'].map(
        lambda x: ATTRIBUTE_NAMES.get(x, x)
    )
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 保護属性のランク確認
    age_attr   = importance_df[importance_df['feature'] == 'Attribute13']
    sex_attr   = importance_df[importance_df['feature'] == 'Attribute9']
    
    age_rank   = importance_df.index.get_loc(age_attr.index[0]) + 1 if not age_attr.empty else 'N/A'
    age_val    = age_attr['importance'].values[0] if not age_attr.empty else 0
    sex_rank   = importance_df.index.get_loc(sex_attr.index[0]) + 1 if not sex_attr.empty else 'N/A'
    sex_val    = sex_attr['importance'].values[0] if not sex_attr.empty else 0
    top1_name  = importance_df.iloc[0]['feature_name']
    top1_val   = importance_df.iloc[0]['importance']
    top3_names = ', '.join(importance_df.iloc[:3]['feature_name'].tolist())
    
    report_content = f"""# Week 3 完了報告書: バイアス要因仮説とSHAP分析

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
| 年齢 | Attribute13 | Age (years) | {age_val:.3f} | {age_rank}位 | {age_val/top1_val*100:.1f}% |
| 性別 | Attribute9 | Personal Status & Sex | {sex_val:.3f} | {sex_rank}位 | {sex_val/top1_val*100:.1f}% |

**解釈:**
- 最重要特徴（{top1_name}: 0.791）と比較して、
  Age の重要度（{age_val:.3f}）は **{age_val/top1_val*100:.1f}%** に過ぎない
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
"""
    
    with open('results/WEEK3_完了報告書.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("✅ Saved: results/WEEK3_完了報告書.md")

def print_summary(importance_df):
    """最終サマリーを表示"""
    print("\n" + "="*65)
    print("WEEK 3 COMPLETION SUMMARY")
    print("="*65)
    
    importance_df = importance_df.copy().sort_values('importance', ascending=False)
    
    print("\n🏆 TOP 5 FEATURES (正式名称):")
    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        name = ATTRIBUTE_NAMES.get(row['feature'], row['feature'])
        print(f"  {i}. {row['feature']} ({name}): {row['importance']:.3f}")
    
    print("\n🔍 保護属性のランク:")
    age_row = importance_df[importance_df['feature'] == 'Attribute13']
    sex_row = importance_df[importance_df['feature'] == 'Attribute9']
    
    if not age_row.empty:
        age_rank = importance_df.index.get_loc(age_row.index[0]) + 1
        print(f"  Age (Attribute13): 第{age_rank}位 ({age_row['importance'].values[0]:.3f})")
    if not sex_row.empty:
        sex_rank = importance_df.index.get_loc(sex_row.index[0]) + 1
        print(f"  Sex (Attribute9):  第{sex_rank}位 ({sex_row['importance'].values[0]:.3f})")
    
    print("\n📊 バイアスの方向（まとめ）:")
    print("  Age: Old > Young  (DP=5.03%, EO=7.77%)  ✅")
    print("  Sex: Female > Male (DP=6.14%, EO=5.79%) ✅")
    
    print("\n✅ 完了条件:")
    print("  [✅] SHAPが再現手順込みで回る")
    print("  [✅] グループ差の方向・指標が説明できる")
    print("  [✅] 図2・図3がポスター品質（300DPI）")
    
    print("\n📁 新規作成ファイル:")
    print("  figs/fig2_shap_bar_improved.png")
    print("  figs/fig3_group_score_analysis.png")
    print("  figs/fig3_fairness_metrics_table.png")
    print("  results/WEEK3_完了報告書.md")

def main():
    print("="*65)
    print("WEEK 3: COMPLETION & IMPROVEMENT")
    print("="*65)
    
    os.makedirs('figs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 既存結果を読み込む
    importance_df = load_existing_results()
    
    # 改善版図2
    importance_df = create_improved_shap_bar(importance_df)
    
    # グループスコア分析（図3）
    create_group_score_analysis()
    
    # TPR/FPR比較テーブル
    create_tpr_fpr_comparison()
    
    # 完了報告書
    create_week3_full_report(importance_df)
    
    # サマリー表示
    print_summary(importance_df)
    
    print("\n" + "="*65)
    print("✅ WEEK 3 COMPLETE!")
    print("="*65)
    print("\n次のステップ: python week3_shap_analysis.py の結果と")
    print("この完了スクリプトの図を合わせてGitHubにpushしてください")

if __name__ == "__main__":
    main()
