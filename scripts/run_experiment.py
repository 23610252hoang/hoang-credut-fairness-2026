"""
Week 2: Model Comparison with Cross-Validation
目的: 3モデル（LR, RF, XGBoost）を反復評価し、精度と公平性を比較
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    公平性指標を計算
    """
    # Demographic Parity
    group_0_rate = y_pred[sensitive_attr == 0].mean()
    group_1_rate = y_pred[sensitive_attr == 1].mean()
    dp_diff = abs(group_0_rate - group_1_rate)
    
    # Equal Opportunity (TPR difference)
    mask_0_positive = (sensitive_attr == 0) & (y_true == 1)
    mask_1_positive = (sensitive_attr == 1) & (y_true == 1)
    
    tpr_0 = ((y_pred == 1) & mask_0_positive).sum() / mask_0_positive.sum() if mask_0_positive.sum() > 0 else 0
    tpr_1 = ((y_pred == 1) & mask_1_positive).sum() / mask_1_positive.sum() if mask_1_positive.sum() > 0 else 0
    
    eo_diff = abs(tpr_0 - tpr_1)
    
    return {
        'dp_diff': dp_diff,
        'eo_diff': eo_diff,
        'group_0_rate': group_0_rate,
        'group_1_rate': group_1_rate,
        'tpr_0': tpr_0,
        'tpr_1': tpr_1
    }

def preprocess_data(df):
    """
    データ前処理（Week 1と同じ）
    """
    # 必須列チェック
    required_cols = ['target', 'age_binary', 'sex_binary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Features除外
    exclude_cols = ['target', 'age_binary', 'sex_binary', 'age_group', 'sex_group', 'class']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # カテゴリカル変数エンコード
    df_features = df[feature_cols].copy()
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_features.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_features) > 0:
        for col in categorical_features:
            le = LabelEncoder()
            df_features[col + '_encoded'] = le.fit_transform(df_features[col].astype(str))
        
        df_features = df_features.drop(columns=categorical_features)
        df_features.columns = [col.replace('_encoded', '') for col in df_features.columns]
    
    X = df_features.values
    y = df['target'].values
    age_binary = df['age_binary'].values
    sex_binary = df['sex_binary'].values
    
    return X, y, age_binary, sex_binary

def run_single_fold(model, X_train, y_train, X_test, y_test, age_test, sex_test):
    """
    1つのfoldでモデルを訓練・評価
    """
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 訓練
    model.fit(X_train_scaled, y_train)
    
    # 予測
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 性能指標
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # 公平性指標
    age_metrics = calculate_fairness_metrics(y_test, y_pred, age_test)
    sex_metrics = calculate_fairness_metrics(y_test, y_pred, sex_test)
    
    return {
        'accuracy': acc,
        'auc': auc,
        'dp_age': age_metrics['dp_diff'],
        'eo_age': age_metrics['eo_diff'],
        'dp_sex': sex_metrics['dp_diff'],
        'eo_sex': sex_metrics['eo_diff']
    }

def run_cross_validation(model_name, model, X, y, age_binary, sex_binary, n_splits=5):
    """
    Stratified K-Fold Cross-Validation
    """
    print(f"\n{'='*60}")
    print(f"Running {model_name} with {n_splits}-Fold CV")
    print(f"{'='*60}")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        print(f"  Fold {fold_idx}/{n_splits}...", end=' ')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        age_test = age_binary[test_idx]
        sex_test = sex_binary[test_idx]
        
        fold_result = run_single_fold(model, X_train, y_train, X_test, y_test, age_test, sex_test)
        fold_result['fold'] = fold_idx
        results.append(fold_result)
        
        print(f"Acc: {fold_result['accuracy']:.4f}, AUC: {fold_result['auc']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # 平均±標準偏差を計算
    summary = {
        'model': model_name,
        'accuracy_mean': results_df['accuracy'].mean(),
        'accuracy_std': results_df['accuracy'].std(),
        'auc_mean': results_df['auc'].mean(),
        'auc_std': results_df['auc'].std(),
        'dp_age_mean': results_df['dp_age'].mean(),
        'dp_age_std': results_df['dp_age'].std(),
        'eo_age_mean': results_df['eo_age'].mean(),
        'eo_age_std': results_df['eo_age'].std(),
        'dp_sex_mean': results_df['dp_sex'].mean(),
        'dp_sex_std': results_df['dp_sex'].std(),
        'eo_sex_mean': results_df['eo_sex'].mean(),
        'eo_sex_std': results_df['eo_sex'].std()
    }
    
    print(f"\n✅ {model_name} Summary:")
    print(f"  Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  AUC:      {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  DP_Age:   {summary['dp_age_mean']:.4f} ± {summary['dp_age_std']:.4f}")
    print(f"  DP_Sex:   {summary['dp_sex_mean']:.4f} ± {summary['dp_sex_std']:.4f}")
    
    return results_df, summary

def main():
    print("="*60)
    print("WEEK 2: MODEL COMPARISON WITH CROSS-VALIDATION")
    print("="*60)
    
    # データ読み込み
    print("\n📥 Loading data...")
    df = pd.read_csv('data/german_credit_processed.csv')
    print(f"✅ Loaded {len(df)} samples, {len(df.columns)} columns")
    
    # 前処理
    print("\n🔧 Preprocessing...")
    X, y, age_binary, sex_binary = preprocess_data(df)
    print(f"✅ Feature matrix: {X.shape}")
    print(f"✅ Target distribution: {np.bincount(y)}")
    
    # モデル定義（最小限のハイパーパラメータ）
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
    }
    
    # 各モデルでCV実行
    all_fold_results = []
    all_summaries = []
    
    for model_name, model in models.items():
        fold_results, summary = run_cross_validation(
            model_name, model, X, y, age_binary, sex_binary, n_splits=5
        )
        fold_results['model'] = model_name
        all_fold_results.append(fold_results)
        all_summaries.append(summary)
    
    # 結果を保存
    os.makedirs('results', exist_ok=True)
    
    # 全foldの詳細結果
    all_folds_df = pd.concat(all_fold_results, ignore_index=True)
    all_folds_df.to_csv('results/exp_v1_all_folds.csv', index=False)
    print("\n💾 Saved: results/exp_v1_all_folds.csv")
    
    # サマリー（表2）
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv('results/exp_v1_summary.csv', index=False)
    print("💾 Saved: results/exp_v1_summary.csv")
    
    # 表示
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY (表2)")
    print("="*60)
    
    print("\n性能指標:")
    print(summary_df[['model', 'accuracy_mean', 'accuracy_std', 'auc_mean', 'auc_std']].to_string(index=False))
    
    print("\n公平性指標 (Age):")
    print(summary_df[['model', 'dp_age_mean', 'dp_age_std', 'eo_age_mean', 'eo_age_std']].to_string(index=False))
    
    print("\n公平性指標 (Sex):")
    print(summary_df[['model', 'dp_sex_mean', 'dp_sex_std', 'eo_sex_mean', 'eo_sex_std']].to_string(index=False))
    print("\n" + "="*60)
    print("✅ WEEK 2 EXPERIMENT COMPLETED!")
    print("="*60)
    print("\n次のステップ:")
    print("1. python create_comparison_plots.py を実行")
    print("2. results/exp_v1_summary.csv を確認")
    print("3. figs/ フォルダの図を確認")

if __name__ == "__main__":
    main()

