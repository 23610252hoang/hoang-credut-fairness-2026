"""
STEP 3: Baseline Model (Logistic Regression)
Mục tiêu: Chạy model đầu tiên, đánh giá accuracy + fairness
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def main():
    print("="*60)
    print("STEP 3: BASELINE MODEL (LOGISTIC REGRESSION)")
    print("="*60)
    
    # Đọc dữ liệu
    df = pd.read_csv('data/german_credit_processed.csv')
    print(f"✅ Đã đọc dữ liệu: {len(df)} dòng")
    # Ensure demographic group columns exist or create fallbacks
    if 'age_group' not in df.columns:
        if 'Age' in df.columns:
            df['age_group'] = df['Age'].apply(lambda x: 'Young' if x <= 25 else 'Old')
        else:
            df['age_group'] = 'Unknown'
    if 'sex_group' not in df.columns:
        if 'Personal_status' in df.columns:
            df['sex_group'] = df['Personal_status'].apply(
                lambda x: 'Male' if 'male' in str(x).lower() else 'Female'
            )
        else:
            df['sex_group'] = 'Unknown'
    
    # Chuẩn bị features
    print("\n📊 Chuẩn bị dữ liệu...")
    
    # Lấy features số
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Loại bỏ target và group columns
    exclude = ['class', 'target', 'age_group', 'sex_group', 'Risk']
    features = [col for col in numeric_cols if col not in exclude]
    
    print(f"Sử dụng {len(features)} features:")
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")
    
    X = df[features].values
    y = df['target'].values
    
    # Sensitive attributes (map to numeric, fallback to 0 for unknown)
    age_group = df['age_group'].map({'Young': 0, 'Old': 1}).fillna(0).astype(int).values
    sex_group = df['sex_group'].map({'Female': 0, 'Male': 1}).fillna(0).astype(int).values
    
    # Chia train/test
    print("\n📊 Chia train/test...")
    X_train, X_test, y_train, y_test, age_train, age_test, sex_train, sex_test = train_test_split(
        X, y, age_group, sex_group,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Chuẩn hóa
    print("\n📊 Chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n🤖 Training Logistic Regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("✅ Training hoàn thành!")
    
    # Dự đoán
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Đánh giá Accuracy
    print("\n" + "="*60)
    print("KẾT QUẢ ACCURACY")
    print("="*60)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"AUC-ROC:  {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Bad (0)', 'Good (1)']))
    
    # Đánh giá Fairness - Age
    print("\n" + "="*60)
    print("KẾT QUẢ FAIRNESS - AGE")
    print("="*60)
    
    dp_age = demographic_parity_difference(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=age_test
    )
    
    eo_age = equalized_odds_difference(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=age_test
    )
    
    print(f"Demographic Parity: {dp_age:.4f}")
    print(f"  (0 = hoàn toàn công bằng)")
    print(f"Equalized Odds:     {eo_age:.4f}")
    
    print("\nApproval Rate by Age Group:")
    for age_val, name in [(0, 'Young'), (1, 'Old')]:
        mask = (age_test == age_val)
        rate = y_pred[mask].mean()
        count = mask.sum()
        print(f"  {name}: {rate:.2%} ({count} samples)")
    
    # Đánh giá Fairness - Sex
    print("\n" + "="*60)
    print("KẾT QUẢ FAIRNESS - SEX")
    print("="*60)
    
    dp_sex = demographic_parity_difference(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sex_test
    )
    
    eo_sex = equalized_odds_difference(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sex_test
    )
    
    print(f"Demographic Parity: {dp_sex:.4f}")
    print(f"Equalized Odds:     {eo_sex:.4f}")
    
    print("\nApproval Rate by Sex Group:")
    for sex_val, name in [(0, 'Female'), (1, 'Male')]:
        mask = (sex_test == sex_val)
        rate = y_pred[mask].mean()
        count = mask.sum()
        print(f"  {name}: {rate:.2%} ({count} samples)")
    
    # Lưu kết quả
    results = pd.DataFrame({
        'Model': ['Logistic Regression'],
        'Accuracy': [acc],
        'AUC': [auc],
        'DP_Age': [dp_age],
        'EO_Age': [eo_age],
        'DP_Sex': [dp_sex],
        'EO_Sex': [eo_sex]
    })
    
    results.to_csv('results/baseline_results.csv', index=False)
    # Ensure results directory exists (file saved above) – create if needed
    import os
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH STEP 3!")
    print("="*60)
    print("💾 Kết quả đã lưu: results/baseline_results.csv")
    print("\n📊 SUMMARY:")
    print(results.to_string(index=False))
    
    # Nhận xét
    print("\n🎯 NHẬN XÉT")
    print("Xem file results/baseline_results.csv để biết chi tiết.")

if __name__ == "__main__":
    main()