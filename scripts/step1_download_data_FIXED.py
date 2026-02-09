"""
STEP 1: Download German Credit Data (FIXED VERSION)
Mục tiêu: Tải dữ liệu và xử lý đúng protected attributes
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import os

def main():
    print("="*60)
    print("STEP 1: TẢI GERMAN CREDIT DATA (FIXED)")
    print("="*60)
    
    # Tạo thư mục
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figs', exist_ok=True)
    
    # Tải dữ liệu
    print("\n📥 Đang tải dữ liệu từ UCI Repository...")
    german_credit = fetch_ucirepo(id=144)
    
    X = german_credit.data.features
    y = german_credit.data.targets
    
    print(f"✅ Tải thành công!")
    print(f"   Features: {X.shape}")
    print(f"   Target: {y.shape}")
    
    # Gộp thành 1 DataFrame
    df = pd.concat([X, y], axis=1)
    
    # Hiển thị thông tin
    print("\n" + "="*60)
    print("THÔNG TIN DỮ LIỆU")
    print("="*60)
    print(f"Số dòng: {len(df)}")
    print(f"Số cột: {len(df.columns)}")
    
    print("\nTên các cột:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # ============================================
    # FIX 1: Xử lý Age đúng
    # ============================================
    print("\n" + "="*60)
    print("PHÂN TÍCH AGE")
    print("="*60)
    
    if 'Age' in df.columns:
        print(f"Min: {df['Age'].min()}")
        print(f"Max: {df['Age'].max()}")
        print(f"Mean: {df['Age'].mean():.1f}")
        print(f"Median: {df['Age'].median():.1f}")
        
        # Tạo age groups (categorical)
        df['age_group'] = df['Age'].apply(lambda x: 'Young' if x <= 25 else 'Old')
        
        # ✅ FIX: Tạo age_binary (0/1) để tính fairness
        df['age_binary'] = (df['Age'] <= 25).astype(int)  # 1=Young, 0=Old
        
        print("\nAge Groups:")
        print(df['age_group'].value_counts())
        print("\nAge Binary (for fairness metrics):")
        print(f"  Young (1): {(df['age_binary']==1).sum()}")
        print(f"  Old (0):   {(df['age_binary']==0).sum()}")
    else:
        print("⚠️  Column 'Age' not found!")
    
    # ============================================
    # FIX 2: Xử lý Sex đúng từ Personal_status
    # ============================================
    print("\n" + "="*60)
    print("PHÂN TÍCH SEX")
    print("="*60)
    
    if 'Personal_status' in df.columns:
        print("Personal_status values:")
        print(df['Personal_status'].value_counts())
        
        # Mapping dựa trên định nghĩa German Credit Data
        # A91: male divorced/separated
        # A92: female divorced/separated/married
        # A93: male single
        # A94: male married/widowed
        # A95: female single
        
        def extract_sex(status):
            status_lower = str(status).lower()
            if 'a92' in status_lower or 'a95' in status_lower:
                return 'Female'
            elif 'a91' in status_lower or 'a93' in status_lower or 'a94' in status_lower:
                return 'Male'
            elif 'female' in status_lower:
                return 'Female'
            elif 'male' in status_lower:
                return 'Male'
            else:
                # Fallback: check text content
                if 'female' in status_lower:
                    return 'Female'
                else:
                    return 'Male'
        
        df['sex_group'] = df['Personal_status'].apply(extract_sex)
        
        # ✅ FIX: Tạo sex_binary (0/1) để tính fairness
        df['sex_binary'] = (df['sex_group'] == 'Female').astype(int)  # 1=Female, 0=Male
        
        print("\nSex Groups:")
        print(df['sex_group'].value_counts())
        print("\nSex Binary (for fairness metrics):")
        print(f"  Female (1): {(df['sex_binary']==1).sum()}")
        print(f"  Male (0):   {(df['sex_binary']==0).sum()}")
    else:
        print("⚠️  Column 'Personal_status' not found!")
    
    # ============================================
    # FIX 3: Xử lý Target đúng
    # ============================================
    print("\n" + "="*60)
    print("PHÂN TÍCH TARGET")
    print("="*60)
    
    # Tìm cột target
    target_col = None
    if 'class' in df.columns:
        target_col = 'class'
    elif 'Risk' in df.columns:
        target_col = 'Risk'
    else:
        target_col = df.columns[-1]
    
    print(f"Target column: {target_col}")
    print(f"Unique values: {df[target_col].unique()}")
    print(f"Value counts:\n{df[target_col].value_counts()}")
    
    # Chuyển về 0/1 (1=Good credit, 0=Bad credit)
    if df[target_col].dtype == 'object':
        # Nếu là string 'good'/'bad'
        df['target'] = df[target_col].map({'good': 1, 'bad': 0})
        if df['target'].isnull().any():
            df['target'] = (df[target_col] == '1').astype(int)
    else:
        # Nếu là số 1/2 hoặc 0/1
        if df[target_col].min() == 1:
            df['target'] = (df[target_col] == 1).astype(int)  # 1→1, 2→0
        else:
            df['target'] = df[target_col].astype(int)
    
    print("\n✅ Đã tạo cột 'target': 1=Good credit, 0=Bad credit")
    print(df['target'].value_counts())
    print(f"Good credit rate: {df['target'].mean():.2%}")
    
    # ============================================
    # FIX 4: Kiểm tra bias trong raw data
    # ============================================
    print("\n" + "="*60)
    print("⚠️  KIỂM TRA BIAS TRONG DỮ LIỆU GỐC")
    print("="*60)
    
    if 'age_group' in df.columns and 'target' in df.columns:
        print("\nGood credit rate by Age:")
        for age in ['Young', 'Old']:
            mask = df['age_group'] == age
            rate = df[mask]['target'].mean()
            count = mask.sum()
            print(f"  {age}: {rate:.3f} ({rate:.1%}) - n={count}")
        
        age_gap = abs(df[df['age_group']=='Young']['target'].mean() - 
                     df[df['age_group']=='Old']['target'].mean())
        print(f"  ⚠️  Age gap: {age_gap:.3f} ({age_gap:.1%})")
    
    if 'sex_group' in df.columns and 'target' in df.columns:
        print("\nGood credit rate by Sex:")
        for sex in ['Male', 'Female']:
            mask = df['sex_group'] == sex
            rate = df[mask]['target'].mean()
            count = mask.sum()
            print(f"  {sex}: {rate:.3f} ({rate:.1%}) - n={count}")
        
        sex_gap = abs(df[df['sex_group']=='Male']['target'].mean() - 
                     df[df['sex_group']=='Female']['target'].mean())
        print(f"  ⚠️  Sex gap: {sex_gap:.3f} ({sex_gap:.1%})")
    
    # Kiểm tra missing values
    print("\n" + "="*60)
    print("MISSING VALUES")
    print("="*60)
    
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ Không có missing values!")
    else:
        print("⚠️  Có missing values:")
        print(missing[missing > 0])
    
    # ============================================
    # Lưu dữ liệu
    # ============================================
    output_file = 'data/german_credit_processed.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH!")
    print("="*60)
    print(f"💾 Đã lưu: {output_file}")
    print(f"📊 Tổng số dòng: {len(df)}")
    print(f"📊 Tổng số cột: {len(df.columns)}")
    
    # Tạo summary
    summary = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'age_young': int((df['age_binary']==1).sum()) if 'age_binary' in df.columns else 0,
        'age_old': int((df['age_binary']==0).sum()) if 'age_binary' in df.columns else 0,
        'sex_male': int((df['sex_binary']==0).sum()) if 'sex_binary' in df.columns else 0,
        'sex_female': int((df['sex_binary']==1).sum()) if 'sex_binary' in df.columns else 0,
        'target_good': int((df['target']==1).sum()) if 'target' in df.columns else 0,
        'target_bad': int((df['target']==0).sum()) if 'target' in df.columns else 0,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('results/data_summary.csv', index=False)
    print("💾 Đã lưu: results/data_summary.csv")
    
    print("\n✅ Các cột quan trọng đã tạo:")
    print("  - age_binary: 0=Old, 1=Young (for fairness)")
    print("  - sex_binary: 0=Male, 1=Female (for fairness)")
    print("  - target: 0=Bad credit, 1=Good credit")

if __name__ == "__main__":
    main()
