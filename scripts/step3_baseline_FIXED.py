"""
STEP 3: Baseline Model (FIXED VERSION)
Mục tiêu: Train model với TOÀN BỘ features (có encode categorical) + tính fairness đúng
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    Tính các fairness metrics theo cách manual (không dùng fairlearn)
    để hiểu rõ từng metric
    """
    
    # Demographic Parity (Statistical Parity)
    # DP = |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|
    group_0_rate = y_pred[sensitive_attr == 0].mean()
    group_1_rate = y_pred[sensitive_attr == 1].mean()
    dp_diff = abs(group_0_rate - group_1_rate)
    
    # Equal Opportunity
    # EO = |TPR_0 - TPR_1|
    # TPR = True Positive Rate = TP / (TP + FN)
    
    # Group 0 TPR
    mask_0_positive = (sensitive_attr == 0) & (y_true == 1)
    if mask_0_positive.sum() > 0:
        tpr_0 = ((y_pred == 1) & mask_0_positive).sum() / mask_0_positive.sum()
    else:
        tpr_0 = 0
    
    # Group 1 TPR
    mask_1_positive = (sensitive_attr == 1) & (y_true == 1)
    if mask_1_positive.sum() > 0:
        tpr_1 = ((y_pred == 1) & mask_1_positive).sum() / mask_1_positive.sum()
    else:
        tpr_1 = 0
    
    eo_diff = abs(tpr_0 - tpr_1)
    
    return {
        'dp_diff': dp_diff,
        'eo_diff': eo_diff,
        'group_0_rate': group_0_rate,
        'group_1_rate': group_1_rate,
        'tpr_0': tpr_0,
        'tpr_1': tpr_1
    }

def main():
    print("="*60)
    print("STEP 3: BASELINE MODEL (FIXED VERSION)")
    print("="*60)
    
    # Đọc dữ liệu
    df = pd.read_csv('data/german_credit_processed.csv')
    print(f"✅ Đã đọc dữ liệu: {len(df)} dòng, {len(df.columns)} cột")
    
    # Kiểm tra các cột cần thiết
    required_cols = ['target', 'age_binary', 'sex_binary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n⚠️  CẢNH BÁO: Thiếu các cột: {missing_cols}")
        print("Vui lòng chạy lại step1_download_data_FIXED.py")
        return
    
    print("✅ Tất cả các cột cần thiết đều có")
    
    # ============================================
    # FIX: XỬ LÝ TOÀN BỘ FEATURES (không chỉ numeric)
    # ============================================
    print("\n" + "="*60)
    print("📊 CHUẨN BỊ FEATURES")
    print("="*60)
    
    # Loại bỏ các cột không dùng để train
    exclude_cols = ['target', 'age_binary', 'sex_binary', 'age_group', 'sex_group']
    
    # Lấy tất cả features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Tổng số features: {len(feature_cols)}")
    
    # Phân loại features
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    print(f"  - Numeric features: {len(numeric_features)}")
    print(f"  - Categorical features: {len(categorical_features)}")
    
    # ✅ FIX: Encode categorical features
    print("\n🔧 Encoding categorical features...")
    
    df_features = df[feature_cols].copy()
    
    if len(categorical_features) > 0:
        for col in categorical_features:
            le = LabelEncoder()
            df_features[col + '_encoded'] = le.fit_transform(df_features[col].astype(str))
            print(f"  ✓ Encoded {col}: {df_features[col].nunique()} unique values")
        
        # Thay thế categorical bằng encoded
        df_features = df_features.drop(columns=categorical_features)
        
        # Rename encoded columns
        df_features.columns = [col.replace('_encoded', '') for col in df_features.columns]
    
    X = df_features.values
    y = df['target'].values
    age_binary = df['age_binary'].values
    sex_binary = df['sex_binary'].values
    
    print(f"\n✅ Final feature matrix shape: {X.shape}")
    
    # ============================================
    # Chia train/test
    # ============================================
    print("\n" + "="*60)
    print("📊 CHIA TRAIN/TEST")
    print("="*60)
    
    X_train, X_test, y_train, y_test, age_train, age_test, sex_train, sex_test = train_test_split(
        X, y, age_binary, sex_binary,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    print(f"Features:  {X_train.shape[1]}")
    
    # Chuẩn hóa
    print("\n🔧 Chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Đã chuẩn hóa xong!")
    
    # ============================================
    # Train model
    # ============================================
    print("\n" + "="*60)
    print("🤖 TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    print("✅ Training hoàn thành!")
    
    # Dự đoán
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # ============================================
    # Đánh giá Accuracy
    # ============================================
    print("\n" + "="*60)
    print("📈 KẾT QUẢ ACCURACY")
    print("="*60)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_proba_test)
    
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test AUC-ROC:      {test_auc:.4f}")
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred_test, 
                                target_names=['Bad (0)', 'Good (1)']))
    
    # ============================================
    # Đánh giá Fairness - Age
    # ============================================
    print("\n" + "="*60)
    print("⚖️  FAIRNESS ANALYSIS - AGE")
    print("="*60)
    
    age_metrics = calculate_fairness_metrics(y_test, y_pred_test, age_test)
    
    print(f"Demographic Parity (DP):")
    print(f"  Group 0 (Old)   positive rate: {age_metrics['group_0_rate']:.4f}")
    print(f"  Group 1 (Young) positive rate: {age_metrics['group_1_rate']:.4f}")
    print(f"  → DP difference: {age_metrics['dp_diff']:.4f} {'✅ Fair' if age_metrics['dp_diff'] <= 0.10 else '❌ Unfair'}")
    
    print(f"\nEqual Opportunity (EO):")
    print(f"  Group 0 (Old)   TPR: {age_metrics['tpr_0']:.4f}")
    print(f"  Group 1 (Young) TPR: {age_metrics['tpr_1']:.4f}")
    print(f"  → EO difference: {age_metrics['eo_diff']:.4f} {'✅ Fair' if age_metrics['eo_diff'] <= 0.10 else '❌ Unfair'}")
    
    print(f"\nApproval Rate by Age Group:")
    for age_val, name in [(0, 'Old'), (1, 'Young')]:
        mask = (age_test == age_val)
        rate = y_pred_test[mask].mean()
        count = mask.sum()
        print(f"  {name:6s}: {rate:.2%} ({count} samples)")
    
    # ============================================
    # Đánh giá Fairness - Sex
    # ============================================
    print("\n" + "="*60)
    print("⚖️  FAIRNESS ANALYSIS - SEX")
    print("="*60)
    
    sex_metrics = calculate_fairness_metrics(y_test, y_pred_test, sex_test)
    
    print(f"Demographic Parity (DP):")
    print(f"  Group 0 (Male)   positive rate: {sex_metrics['group_0_rate']:.4f}")
    print(f"  Group 1 (Female) positive rate: {sex_metrics['group_1_rate']:.4f}")
    print(f"  → DP difference: {sex_metrics['dp_diff']:.4f} {'✅ Fair' if sex_metrics['dp_diff'] <= 0.10 else '❌ Unfair'}")
    
    print(f"\nEqual Opportunity (EO):")
    print(f"  Group 0 (Male)   TPR: {sex_metrics['tpr_0']:.4f}")
    print(f"  Group 1 (Female) TPR: {sex_metrics['tpr_1']:.4f}")
    print(f"  → EO difference: {sex_metrics['eo_diff']:.4f} {'✅ Fair' if sex_metrics['eo_diff'] <= 0.10 else '❌ Unfair'}")
    
    print(f"\nApproval Rate by Sex Group:")
    for sex_val, name in [(0, 'Male'), (1, 'Female')]:
        mask = (sex_test == sex_val)
        rate = y_pred_test[mask].mean()
        count = mask.sum()
        print(f"  {name:6s}: {rate:.2%} ({count} samples)")
    
    # ============================================
    # Lưu kết quả
    # ============================================
    results = pd.DataFrame({
        'Model': ['Logistic Regression'],
        'Train_Accuracy': [train_acc],
        'Test_Accuracy': [test_acc],
        'AUC': [test_auc],
        'DP_Age': [age_metrics['dp_diff']],
        'EO_Age': [age_metrics['eo_diff']],
        'DP_Sex': [sex_metrics['dp_diff']],
        'EO_Sex': [sex_metrics['eo_diff']]
    })
    
    results.to_csv('results/baseline_results.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH STEP 3!")
    print("="*60)
    print("💾 Kết quả đã lưu: results/baseline_results.csv")
    print("\n📊 SUMMARY:")
    print(results.to_string(index=False))
    
    # Nhận xét
    print("\n" + "="*60)
    print("🎯 NHẬN XÉT")
    print("="*60)
    
    print("\n1. Accuracy & AUC:")
    print(f"   - Model đạt {test_acc:.1%} accuracy và {test_auc:.3f} AUC")
    print(f"   - {'Tốt' if test_auc > 0.75 else 'Cần cải thiện'}")
    
    print("\n2. Age Bias:")
    if age_metrics['dp_diff'] > 0.10:
        print(f"   ⚠️  Có age bias nghiêm trọng (DP = {age_metrics['dp_diff']:.3f})")
        print(f"   - Người trẻ bị thiệt: chỉ {age_metrics['group_1_rate']:.1%} approval rate")
        print(f"   - So với người lớn tuổi: {age_metrics['group_0_rate']:.1%}")
    else:
        print(f"   ✅ Age bias chấp nhận được (DP = {age_metrics['dp_diff']:.3f})")
    
    print("\n3. Sex Bias:")
    if sex_metrics['dp_diff'] > 0.10:
        print(f"   ⚠️  Có sex bias (DP = {sex_metrics['dp_diff']:.3f})")
        print(f"   - Phụ nữ bị thiệt: chỉ {sex_metrics['group_1_rate']:.1%} approval rate")
        print(f"   - So với nam giới: {sex_metrics['group_0_rate']:.1%}")
    else:
        print(f"   ✅ Sex bias chấp nhận được (DP = {sex_metrics['dp_diff']:.3f})")
    
    print("\n4. Kết luận:")
    print(f"   - Fairness threshold: DP/EO ≤ 0.10")
    print(f"   - Model {'ĐẠT' if age_metrics['dp_diff'] <= 0.10 and sex_metrics['dp_diff'] <= 0.10 else 'KHÔNG ĐẠT'} yêu cầu fairness")

if __name__ == "__main__":
    main()
