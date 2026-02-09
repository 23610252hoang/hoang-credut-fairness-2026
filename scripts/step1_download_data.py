"""
STEP 1: Download German Credit Data
Mục tiêu: Tải dữ liệu và kiểm tra cơ bản
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

def main():
    print("="*60)
    print("STEP 1: TẢI GERMAN CREDIT DATA")
    print("="*60)
    
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
    
    # Kiểm tra Age
    print("\n" + "="*60)
    print("PHÂN TÍCH AGE")
    print("="*60)
    
    if 'Age' in df.columns:
        print(f"Min: {df['Age'].min()}")
        print(f"Max: {df['Age'].max()}")
        print(f"Mean: {df['Age'].mean():.1f}")
        print(f"Median: {df['Age'].median():.1f}")
        
        # Tạo age groups
        df['age_group'] = df['Age'].apply(lambda x: 'Young' if x <= 25 else 'Old')
        
        print("\nAge Groups:")
        print(df['age_group'].value_counts())
        print("\nPhần trăm:")
        print(df['age_group'].value_counts(normalize=True) * 100)
    
    # Kiểm tra Sex
    print("\n" + "="*60)
    print("PHÂN TÍCH SEX")
    print("="*60)
    
    if 'Personal_status' in df.columns:
        print("Personal_status values:")
        print(df['Personal_status'].value_counts())
        
        # Tạo sex groups
        df['sex_group'] = df['Personal_status'].apply(
            lambda x: 'Male' if 'male' in str(x).lower() else 'Female'
        )
        
        print("\nSex Groups:")
        print(df['sex_group'].value_counts())
        print("\nPhần trăm:")
        print(df['sex_group'].value_counts(normalize=True) * 100)
    
    # Kiểm tra Target
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
    print(df[target_col].value_counts())
    
    # Chuyển về 0/1
    if df[target_col].dtype == 'object' or df[target_col].min() != 0:
        df['target'] = (df[target_col] == 1).astype(int)
        print("\n✅ Đã tạo cột 'target': 1=Good, 0=Bad")
        print(df['target'].value_counts())
    else:
        df['target'] = df[target_col]
    
    # Kiểm tra missing values
    print("\n" + "="*60)
    print("MISSING VALUES")
    print("="*60)
    
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ Không có missing values!")
    else:
        print("⚠️ Có missing values:")
        print(missing[missing > 0])
    
    # Lưu dữ liệu
    df.to_csv('data/german_credit_processed.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH!")
    print("="*60)
    print("💾 Đã lưu: data/german_credit_processed.csv")
    print(f"📊 Tổng số dòng: {len(df)}")
    print(f"📊 Tổng số cột: {len(df.columns)}")
    
    # Tạo summary
    # Build summary safely: demographic group columns may not exist
    age_young = int(df['age_group'].eq('Young').sum()) if 'age_group' in df.columns else 0
    age_old = int(df['age_group'].eq('Old').sum()) if 'age_group' in df.columns else 0
    sex_male = int(df['sex_group'].eq('Male').sum()) if 'sex_group' in df.columns else 0
    sex_female = int(df['sex_group'].eq('Female').sum()) if 'sex_group' in df.columns else 0

    summary = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'age_young': age_young,
        'age_old': age_old,
        'sex_male': sex_male,
        'sex_female': sex_female,
        'target_good': int(df['target'].eq(1).sum()) if 'target' in df.columns else 0,
        'target_bad': int(df['target'].eq(0).sum()) if 'target' in df.columns else 0,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('results/data_summary.csv', index=False)
    print("💾 Đã lưu: results/data_summary.csv")

if __name__ == "__main__":
    main()
    