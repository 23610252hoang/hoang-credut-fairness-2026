"""
Week 2: Visualization - Model Comparison
目的: 精度と公平性のトレードオフを可視化（図1）
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Style設定
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_comparison_plots():
    """
    モデル比較の図を作成
    """
    print("="*60)
    print("CREATING COMPARISON PLOTS")
    print("="*60)
    
    # データ読み込み
    print("\n📥 Loading results...")
    summary_df = pd.read_csv('results/exp_v1_summary.csv')
    all_folds_df = pd.read_csv('results/exp_v1_all_folds.csv')
    
    print(f"✅ Loaded summary for {len(summary_df)} models")
    print(f"✅ Loaded {len(all_folds_df)} fold results")
    
    # 出力ディレクトリ
    os.makedirs('figs', exist_ok=True)
    
    # ==========================================
    # 図1: Accuracy vs Fairness Scatter Plot
    # ==========================================
    print("\n📊 Creating Figure 1: Accuracy vs Fairness...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison: Accuracy vs Fairness Trade-off', 
                 fontsize=16, fontweight='bold')
    
    # モデルの色とマーカー
    model_styles = {
        'Logistic Regression': {'color': '#2ecc71', 'marker': 'o', 'label': 'LR'},
        'Random Forest': {'color': '#3498db', 'marker': 's', 'label': 'RF'},
        'XGBoost': {'color': '#e74c3c', 'marker': '^', 'label': 'XGB'}
    }
    
    # Left: Accuracy vs DP_Age
    ax1 = axes[0]
    for model in summary_df['model']:
        row = summary_df[summary_df['model'] == model].iloc[0]
        style = model_styles[model]
        
        ax1.scatter(
            row['dp_age_mean'], 
            row['accuracy_mean'],
            s=200,
            color=style['color'],
            marker=style['marker'],
            label=style['label'],
            alpha=0.7,
            edgecolors='black',
            linewidths=2
        )
        
        # エラーバー
        ax1.errorbar(
            row['dp_age_mean'],
            row['accuracy_mean'],
            xerr=row['dp_age_std'],
            yerr=row['accuracy_std'],
            fmt='none',
            ecolor=style['color'],
            alpha=0.3,
            capsize=5
        )
    
    # 公平性閾値線
    ax1.axvline(x=0.10, color='red', linestyle='--', linewidth=2, 
                label='Fairness Threshold (10%)', alpha=0.7)
    
    ax1.set_xlabel('Demographic Parity Gap (Age)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Age Bias', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Accuracy vs DP_Sex
    ax2 = axes[1]
    for model in summary_df['model']:
        row = summary_df[summary_df['model'] == model].iloc[0]
        style = model_styles[model]
        
        ax2.scatter(
            row['dp_sex_mean'],
            row['accuracy_mean'],
            s=200,
            color=style['color'],
            marker=style['marker'],
            label=style['label'],
            alpha=0.7,
            edgecolors='black',
            linewidths=2
        )
        
        # エラーバー
        ax2.errorbar(
            row['dp_sex_mean'],
            row['accuracy_mean'],
            xerr=row['dp_sex_std'],
            yerr=row['accuracy_std'],
            fmt='none',
            ecolor=style['color'],
            alpha=0.3,
            capsize=5
        )
    
    ax2.axvline(x=0.10, color='red', linestyle='--', linewidth=2,
                label='Fairness Threshold (10%)', alpha=0.7)
    
    ax2.set_xlabel('Demographic Parity Gap (Sex)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Sex Bias', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/fig1_accuracy_vs_fairness.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figs/fig1_accuracy_vs_fairness.png")
    
    # ==========================================
    # 図2: モデル別性能比較 (Bar Chart)
    # ==========================================
    print("\n📊 Creating Figure 2: Model Performance Comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy
    ax1 = axes[0, 0]
    models = summary_df['model'].values
    acc_means = summary_df['accuracy_mean'].values
    acc_stds = summary_df['accuracy_std'].values
    colors = [model_styles[m]['color'] for m in models]
    
    bars = ax1.bar(range(len(models)), acc_means, yerr=acc_stds, 
                   color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([model_styles[m]['label'] for m in models], fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, acc_means, acc_stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                 f'{mean:.3f}±{std:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # AUC
    ax2 = axes[0, 1]
    auc_means = summary_df['auc_mean'].values
    auc_stds = summary_df['auc_std'].values
    
    bars = ax2.bar(range(len(models)), auc_means, yerr=auc_stds,
                   color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([model_styles[m]['label'] for m in models], fontsize=11)
    ax2.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax2.set_title('AUC Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, auc_means, auc_stds)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                 f'{mean:.3f}±{std:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # DP_Age
    ax3 = axes[1, 0]
    dp_age_means = summary_df['dp_age_mean'].values
    dp_age_stds = summary_df['dp_age_std'].values
    
    bars = ax3.bar(range(len(models)), dp_age_means, yerr=dp_age_stds,
                   color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=0.10, color='red', linestyle='--', linewidth=2, label='Threshold', alpha=0.7)
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([model_styles[m]['label'] for m in models], fontsize=11)
    ax3.set_ylabel('DP Gap (Age)', fontsize=12, fontweight='bold')
    ax3.set_title('Age Bias Comparison', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, dp_age_means, dp_age_stds)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                 f'{mean:.3f}±{std:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # DP_Sex
    ax4 = axes[1, 1]
    dp_sex_means = summary_df['dp_sex_mean'].values
    dp_sex_stds = summary_df['dp_sex_std'].values
    
    bars = ax4.bar(range(len(models)), dp_sex_means, yerr=dp_sex_stds,
                   color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=0.10, color='red', linestyle='--', linewidth=2, label='Threshold', alpha=0.7)
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels([model_styles[m]['label'] for m in models], fontsize=11)
    ax4.set_ylabel('DP Gap (Sex)', fontsize=12, fontweight='bold')
    ax4.set_title('Sex Bias Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, dp_sex_means, dp_sex_stds)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                 f'{mean:.3f}±{std:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figs/fig2_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figs/fig2_model_comparison.png")
    
    # ==========================================
    # 図3: Fold別の分布 (Box Plot)
    # ==========================================
    print("\n📊 Creating Figure 3: Fold-wise Distribution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Validation Stability Analysis', fontsize=16, fontweight='bold')
    
    # Accuracy box plot
    ax1 = axes[0, 0]
    all_folds_df.boxplot(column='accuracy', by='model', ax=ax1, patch_artist=True)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Distribution (5 folds)', fontsize=14, fontweight='bold')
    plt.suptitle('')
    
    # AUC box plot
    ax2 = axes[0, 1]
    all_folds_df.boxplot(column='auc', by='model', ax=ax2, patch_artist=True)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax2.set_title('AUC Distribution (5 folds)', fontsize=14, fontweight='bold')
    plt.suptitle('')
    
    # DP_Age box plot
    ax3 = axes[1, 0]
    all_folds_df.boxplot(column='dp_age', by='model', ax=ax3, patch_artist=True)
    ax3.axhline(y=0.10, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('DP Gap (Age)', fontsize=12, fontweight='bold')
    ax3.set_title('Age Bias Distribution (5 folds)', fontsize=14, fontweight='bold')
    plt.suptitle('')
    
    # DP_Sex box plot
    ax4 = axes[1, 1]
    all_folds_df.boxplot(column='dp_sex', by='model', ax=ax4, patch_artist=True)
    ax4.axhline(y=0.10, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax4.set_ylabel('DP Gap (Sex)', fontsize=12, fontweight='bold')
    ax4.set_title('Sex Bias Distribution (5 folds)', fontsize=14, fontweight='bold')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('figs/fig3_cv_stability.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: figs/fig3_cv_stability.png")
    
    print("\n" + "="*60)
    print("✅ ALL PLOTS CREATED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    create_comparison_plots()
