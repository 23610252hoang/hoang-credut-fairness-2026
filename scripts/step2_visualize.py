"""
STEP 2: Visualization - Age & Sex Analysis
Mục tiêu: Tạo biểu đồ cho báo cáo
"""

import os
import pandas as pd
import matplotlib
# Use a non-interactive backend to avoid Tk/Tcl issues on systems without tkinter
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def main():
    print("="*60)
    print("STEP 2: VISUALIZATION")
    print("="*60)
    
    # Đọc dữ liệu
    df = pd.read_csv('data/german_credit_processed.csv')
    print(f"✅ Đã đọc dữ liệu: {len(df)} dòng")
    # Ensure basic demographic groups exist or create them when possible
    if 'Age' in df.columns and 'age_group' not in df.columns:
      df['age_group'] = df['Age'].apply(lambda x: 'Young' if x <= 25 else 'Old')
    if 'Personal_status' in df.columns and 'sex_group' not in df.columns:
      df['sex_group'] = df['Personal_status'].apply(
        lambda x: 'Male' if 'male' in str(x).lower() else 'Female'
      )

    has_age = 'Age' in df.columns
    has_age_group = 'age_group' in df.columns
    has_sex_group = 'sex_group' in df.columns
    has_target = 'target' in df.columns
    
    # Tạo figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('German Credit Data - Age & Sex Analysis', 
                 fontsize=20, fontweight='bold')
    
    # 1. Target Distribution
    ax1 = plt.subplot(3, 3, 1)
    target_counts = df['target'].value_counts()
    colors_target = ['#e74c3c', '#2ecc71']  # Red=Bad, Green=Good
    bars = ax1.bar(['Bad (0)', 'Good (1)'], target_counts.values, 
                   color=colors_target, edgecolor='black', linewidth=1.5)
    ax1.set_title('Target Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}\n({height/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=11)
    
    # 2. Age Distribution
    ax2 = plt.subplot(3, 3, 2)
    if has_age:
      ax2.hist(df['Age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
      ax2.axvline(df['Age'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {df["Age"].mean():.1f}')
      ax2.axvline(25, color='orange', linestyle='--', 
            linewidth=2, label='Threshold: 25')
      ax2.set_title('Age Distribution', fontsize=14, fontweight='bold')
      ax2.set_xlabel('Age', fontsize=12)
      ax2.set_ylabel('Count', fontsize=12)
      ax2.legend()
      ax2.grid(axis='y', alpha=0.3)
    else:
      ax2.text(0.5, 0.5, 'Column missing: Age', ha='center', va='center', fontsize=12)
    
    # 3. Age Groups
    ax3 = plt.subplot(3, 3, 3)
    if has_age_group:
      age_counts = df['age_group'].value_counts()
      colors_age = ['#f39c12', '#9b59b6']
      bars = ax3.bar(age_counts.index, age_counts.values,
               color=colors_age, edgecolor='black', linewidth=1.5)
      ax3.set_title('Age Groups', fontsize=14, fontweight='bold')
      ax3.set_ylabel('Count', fontsize=12)
      ax3.grid(axis='y', alpha=0.3)
      for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=11)
    else:
      ax3.text(0.5, 0.5, 'Column missing: age_group', ha='center', va='center', fontsize=12)
    
    # 4. Sex Groups
    ax4 = plt.subplot(3, 3, 4)
    if has_sex_group:
      sex_counts = df['sex_group'].value_counts()
      colors_sex = ['#3498db', '#e91e63']
      bars = ax4.bar(sex_counts.index, sex_counts.values,
               color=colors_sex, edgecolor='black', linewidth=1.5)
      ax4.set_title('Sex Groups', fontsize=14, fontweight='bold')
      ax4.set_ylabel('Count', fontsize=12)
      ax4.grid(axis='y', alpha=0.3)
      for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(df)*100:.1f}%)',
              ha='center', va='bottom', fontsize=11)
    else:
      ax4.text(0.5, 0.5, 'Column missing: sex_group', ha='center', va='center', fontsize=12)
    
    # 5. Approval Rate by Age
    ax5 = plt.subplot(3, 3, 5)
    age_target = None
    if has_age_group and has_target:
      age_target = pd.crosstab(df['age_group'], df['target'], normalize='index') * 100
      age_target.plot(kind='bar', ax=ax5, color=colors_target,
              edgecolor='black', linewidth=1.5)
      ax5.set_title('Approval Rate by Age Group', fontsize=14, fontweight='bold')
      ax5.set_xlabel('Age Group', fontsize=12)
      ax5.set_ylabel('Percentage (%)', fontsize=12)
      ax5.legend(['Bad (0)', 'Good (1)'], title='Credit')
      ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
      ax5.grid(axis='y', alpha=0.3)
      for container in ax5.containers:
        ax5.bar_label(container, fmt='%.1f%%')
    else:
      ax5.text(0.5, 0.5, 'Need age_group and target', ha='center', va='center', fontsize=12)
    
    # 6. Approval Rate by Sex
    ax6 = plt.subplot(3, 3, 6)
    sex_target = None
    if has_sex_group and has_target:
      sex_target = pd.crosstab(df['sex_group'], df['target'], normalize='index') * 100
      sex_target.plot(kind='bar', ax=ax6, color=colors_target,
              edgecolor='black', linewidth=1.5)
      ax6.set_title('Approval Rate by Sex Group', fontsize=14, fontweight='bold')
      ax6.set_xlabel('Sex Group', fontsize=12)
      ax6.set_ylabel('Percentage (%)', fontsize=12)
      ax6.legend(['Bad (0)', 'Good (1)'], title='Credit')
      ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)
      ax6.grid(axis='y', alpha=0.3)
      for container in ax6.containers:
        ax6.bar_label(container, fmt='%.1f%%')
    else:
      ax6.text(0.5, 0.5, 'Need sex_group and target', ha='center', va='center', fontsize=12)
    
    # 7. Age by Target (Histogram)
    ax7 = plt.subplot(3, 3, 7)
    if has_age and has_target:
      for target_val, color, label in [(0, '#e74c3c', 'Bad'), 
                         (1, '#2ecc71', 'Good')]:
        subset = df[df['target'] == target_val]['Age']
        ax7.hist(subset, bins=20, alpha=0.6, color=color, 
             label=label, edgecolor='black')
      ax7.set_title('Age Distribution by Target', fontsize=14, fontweight='bold')
      ax7.set_xlabel('Age', fontsize=12)
      ax7.set_ylabel('Count', fontsize=12)
      ax7.legend()
      ax7.grid(axis='y', alpha=0.3)
    else:
      ax7.text(0.5, 0.5, 'Need Age and target', ha='center', va='center', fontsize=12)
    
    # 8. Age Boxplot by Target
    ax8 = plt.subplot(3, 3, 8)
    if has_age and has_target:
      df.boxplot(column='Age', by='target', ax=ax8, patch_artist=True)
      ax8.set_title('Age Boxplot by Target', fontsize=14, fontweight='bold')
      ax8.set_xlabel('Target', fontsize=12)
      ax8.set_ylabel('Age', fontsize=12)
      ax8.set_xticklabels(['Bad (0)', 'Good (1)'])
      plt.suptitle('')
    else:
      ax8.text(0.5, 0.5, 'Need Age and target', ha='center', va='center', fontsize=12)
    
    # 9. Summary Text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Build safe summary values
    total_samples = len(df)
    if has_age:
        age_mean = df['Age'].mean()
        age_median = df['Age'].median()
        age_min = df['Age'].min()
        age_max = df['Age'].max()
    else:
        age_mean = age_median = age_min = age_max = None

    if has_age_group:
        age_young_count = (df['age_group'] == 'Young').sum()
        age_old_count = (df['age_group'] == 'Old').sum()
    else:
        age_young_count = age_old_count = 0

    if has_sex_group:
        sex_male_count = (df['sex_group'] == 'Male').sum()
        sex_female_count = (df['sex_group'] == 'Female').sum()
    else:
        sex_male_count = sex_female_count = 0

    if has_target:
        target_good = (df['target'] == 1).sum()
        target_bad = (df['target'] == 0).sum()
    else:
        target_good = target_bad = 0

    summary_text = f"""
SUMMARY STATISTICS

Total Samples: {total_samples:,}

AGE:
  Mean: {f'{age_mean:.1f}' if age_mean is not None else 'N/A'} years
  Median: {f'{age_median:.1f}' if age_median is not None else 'N/A'} years
  Range: {f'{age_min}-{age_max}' if age_min is not None else 'N/A'}
  
  Young (≤25): {age_young_count} ({age_young_count/total_samples*100:.1f}% if total_samples else 0)
  Old (>25): {age_old_count} ({age_old_count/total_samples*100:.1f}% if total_samples else 0)

SEX:
  Male: {sex_male_count} ({sex_male_count/total_samples*100:.1f}% if total_samples else 0)
  Female: {sex_female_count} ({sex_female_count/total_samples*100:.1f}% if total_samples else 0)

TARGET:
  Good: {target_good} ({target_good/total_samples*100:.1f}% if total_samples else 0)
  Bad: {target_bad} ({target_bad/total_samples*100:.1f}% if total_samples else 0)
"""
    
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Ensure output directory exists and save
    os.makedirs('figs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figs/eda_comprehensive.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu: figs/eda_comprehensive.png")
    
    # Print statistics
    print("\n" + "="*60)
    print("APPROVAL RATE ANALYSIS")
    print("="*60)
    print("\nBy Age Group:")
    print(age_target)
    print("\nBy Sex Group:")
    print(sex_target)
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH STEP 2!")
    print("="*60)

if __name__ == "__main__":
    main()