"""
NFL Draft Prediction - Optimized v9 (Robust)
Lower overfitting with simpler features and stronger regularization
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration
RNG = 42
np.random.seed(RNG)
N_FOLDS = 5

print("Loading data...")
train = pd.read_csv("/workspace/train (3).csv")
test = pd.read_csv("/workspace/test (5).csv")

TARGET = "Drafted"
ID_COL = "Id"

y = train[TARGET].values.copy()
train_raw = train.drop(columns=[TARGET]).copy()
n_train = len(train_raw)
n_test = len(test)

all_data = pd.concat([train_raw, test], ignore_index=True)
print(f"Train: {n_train}, Test: {n_test}")

# Performance columns
PERF = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"]
TIME_COLS = ["Sprint_40yd", "Agility_3cone", "Shuttle"]
CAT_COLS = ["School", "Player_Type", "Position_Type", "Position"]

# =============================================================================
# Feature Engineering - Conservative & Robust
# =============================================================================
print("Feature engineering...")

# Basic body metrics
all_data["BMI"] = all_data["Weight"] / (all_data["Height"] ** 2)
all_data["Weight_Height"] = all_data["Weight"] * all_data["Height"]

# MISSINGNESS - This is the most important signal
for col in PERF + ["Age"]:
    all_data[f"{col}_miss"] = all_data[col].isnull().astype(int)
all_data["n_missing"] = all_data[PERF].isnull().sum(axis=1)
all_data["has_all_tests"] = (all_data["n_missing"] == 0).astype(int)

# Performance combinations - Filled with median
all_data["Power_Score"] = all_data["Vertical_Jump"].fillna(0) + all_data["Broad_Jump"].fillna(0)
all_data["Agility_Sum"] = (all_data["Agility_3cone"].fillna(all_data["Agility_3cone"].median()) +
                          all_data["Shuttle"].fillna(all_data["Shuttle"].median()))

# Normalized metrics per KG
all_data["Speed_per_kg"] = all_data["Sprint_40yd"] / all_data["Weight"]
all_data["Vert_per_kg"] = all_data["Vertical_Jump"] / (all_data["Weight"] + 1e-5)

# Global z-scores - Simple
for col in PERF:
    mu, sd = all_data[col].mean(), all_data[col].std() + 1e-9
    z = (all_data[col] - mu) / sd
    all_data[f"{col}_z"] = z if col not in TIME_COLS else -z

all_data["Athleticism"] = sum(all_data[f"{c}_z"].fillna(0) for c in PERF)

# Position-normalized percentiles
print("Computing position percentiles...")
for col in PERF:
    all_data[f"{col}_pct"] = np.nan
    for pos, grp in all_data.groupby("Position"):
        valid = grp[col].notna()
        if valid.sum() > 1:
            ranks = grp.loc[valid, col].rank(pct=True)
            all_data.loc[grp.index[valid], f"{col}_pct"] = 1 - ranks if col in TIME_COLS else ranks

all_data["Athleticism_pct"] = all_data[[f"{c}_pct" for c in PERF]].mean(axis=1)

# Year drift
year_mean = all_data.groupby("Year")["Sprint_40yd"].transform("mean")
all_data["Sprint_vs_year"] = all_data["Sprint_40yd"] - year_mean

# School frequency
train_school_freq = train_raw["School"].value_counts()
all_data["School_Freq"] = all_data["School"].map(train_school_freq).fillna(0)
all_data["School_LogFreq"] = np.log1p(all_data["School_Freq"])

# =============================================================================
# Target Encoding - PROPER K-Fold to avoid ANY leakage
# =============================================================================
print("Target encoding (K-fold)...")

def kfold_target_encoding(train_df, test_df, col, target_col, n_splits=5, smooth=20, seed=42):
    """K-fold target encoding with smoothing - fully avoids leakage"""
    gm = train_df[target_col].mean()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # For training data: K-fold encoding
    train_enc = np.full(len(train_df), gm)
    for tr_idx, va_idx in kf.split(np.arange(len(train_df)), train_df[target_col]):
        tr_vals = train_df[col].iloc[tr_idx]
        tr_y = train_df[target_col].iloc[tr_idx]
        
        means = {}
        for v in tr_vals.unique():
            mask = tr_vals == v
            cnt = mask.sum()
            if cnt > 0:
                mn = tr_y[mask].mean()
                means[v] = (mn * cnt + gm * smooth) / (cnt + smooth)
            else:
                means[v] = gm
        
        train_enc[va_idx] = [means.get(v, gm) for v in train_df[col].iloc[va_idx]]
    
    # For test data: use full train statistics
    means_full = {}
    for v in train_df[col].unique():
        mask = train_df[col] == v
        cnt = mask.sum()
        if cnt > 0:
            mn = train_df[target_col][mask].mean()
            means_full[v] = (mn * cnt + gm * smooth) / (cnt + smooth)
        else:
            means_full[v] = gm
    
    test_enc = test_df[col].map(means_full).fillna(gm).values
    
    return train_enc, test_enc

# Apply K-fold target encoding
for col in CAT_COLS:
    tr_enc, te_enc = kfold_target_encoding(train, test, col, TARGET)
    all_data.loc[:n_train-1, f"{col}_rate"] = tr_enc
    all_data.loc[n_train:, f"{col}_rate"] = te_enc

# =============================================================================
# Feature preparation
# =============================================================================
feature_cols = [c for c in all_data.columns if c not in [ID_COL] + CAT_COLS]
all_feat_cols = feature_cols + CAT_COLS

# For CatBoost
cb_train = all_data.iloc[:n_train][all_feat_cols].copy()
cb_test = all_data.iloc[n_train:][all_feat_cols].copy()

for col in CAT_COLS:
    cb_train[col] = cb_train[col].fillna("unknown").astype(str)
    cb_test[col] = cb_test[col].fillna("unknown").astype(str)

num_cols = [c for c in feature_cols]
imp_cb = SimpleImputer(strategy="median")
cb_train[num_cols] = imp_cb.fit_transform(cb_train[num_cols])
cb_test[num_cols] = imp_cb.transform(cb_test[num_cols])

cat_indices = [all_feat_cols.index(c) for c in CAT_COLS]

# For XGBoost/LightGBM
for col in CAT_COLS:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str).fillna("unknown"))

imp_np = SimpleImputer(strategy="median")
X_train = imp_np.fit_transform(all_data.iloc[:n_train][all_feat_cols])
X_test = imp_np.transform(all_data.iloc[n_train:][all_feat_cols])

print(f"Total features: {len(all_feat_cols)}")

# =============================================================================
# CV Setup
# =============================================================================
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)

# =============================================================================
# CatBoost - More regularization
# =============================================================================
print("\n=== Training CatBoost (regularized) ===")

cb_configs = [
    dict(iterations=800, depth=4, learning_rate=0.03, l2_leaf_reg=10, border_count=64, random_seed=RNG),
    dict(iterations=800, depth=5, learning_rate=0.025, l2_leaf_reg=8, border_count=64, random_seed=RNG+1),
    dict(iterations=600, depth=6, learning_rate=0.04, l2_leaf_reg=15, border_count=32, random_seed=RNG+2),
    dict(iterations=1000, depth=4, learning_rate=0.02, l2_leaf_reg=12, border_count=64, random_seed=RNG+3),
]

cb_oofs = []
cb_tests = []

for cfg_idx, cfg in enumerate(cb_configs):
    oof = np.zeros(n_train)
    test_preds = np.zeros(n_test)
    
    for fold, (tr_idx, va_idx) in enumerate(cv.split(np.arange(n_train), y)):
        model = CatBoostClassifier(
            **cfg,
            cat_features=cat_indices,
            eval_metric="AUC",
            verbose=0,
            od_type="Iter",
            od_wait=100
        )
        
        model.fit(
            cb_train.iloc[tr_idx], y[tr_idx],
            eval_set=(cb_train.iloc[va_idx], y[va_idx])
        )
        
        oof[va_idx] = model.predict_proba(cb_train.iloc[va_idx])[:, 1]
        test_preds += model.predict_proba(cb_test)[:, 1] / N_FOLDS
    
    auc = roc_auc_score(y, oof)
    print(f"CB cfg{cfg_idx + 1}: {auc:.5f}")
    cb_oofs.append(oof)
    cb_tests.append(test_preds)

cb_oof = np.mean(cb_oofs, axis=0)
cb_test = np.mean(cb_tests, axis=0)
print(f"CatBoost ensemble: {roc_auc_score(y, cb_oof):.5f}")

# =============================================================================
# XGBoost - Regularized
# =============================================================================
print("\n=== Training XGBoost ===")

xgb_oof = np.zeros(n_train)
xgb_test = np.zeros(n_test)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.6,
        colsample_bytree=0.5,
        min_child_weight=10,
        gamma=0.3,
        reg_alpha=1.0,
        reg_lambda=3.0,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=RNG,
        n_jobs=-1,
        early_stopping_rounds=100
    )
    
    model.fit(
        X_train[tr_idx], y[tr_idx],
        eval_set=[(X_train[va_idx], y[va_idx])],
        verbose=False
    )
    
    xgb_oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
    xgb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS

print(f"XGBoost: {roc_auc_score(y, xgb_oof):.5f}")

# =============================================================================
# LightGBM - Regularized
# =============================================================================
print("\n=== Training LightGBM ===")

lgb_oof = np.zeros(n_train)
lgb_test = np.zeros(n_test)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    model = lgb.LGBMClassifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.6,
        colsample_bytree=0.5,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=RNG,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train[tr_idx], y[tr_idx],
        eval_set=[(X_train[va_idx], y[va_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    lgb_oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
    lgb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS

print(f"LightGBM: {roc_auc_score(y, lgb_oof):.5f}")

# =============================================================================
# Simple Stacking with Logistic Regression
# =============================================================================
print("\n=== Stacking ===")

stack_train = np.column_stack([cb_oof, xgb_oof, lgb_oof])
stack_test = np.column_stack([cb_test, xgb_test, lgb_test])

stack_oof = np.zeros(n_train)
stack_test_preds = np.zeros(n_test)

for fold, (tr_idx, va_idx) in enumerate(cv.split(stack_train, y)):
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RNG)
    lr.fit(stack_train[tr_idx], y[tr_idx])
    stack_oof[va_idx] = lr.predict_proba(stack_train[va_idx])[:, 1]
    stack_test_preds += lr.predict_proba(stack_test)[:, 1] / N_FOLDS

print(f"Stacking: {roc_auc_score(y, stack_oof):.5f}")

# =============================================================================
# Weighted Ensemble
# =============================================================================
print("\n=== Final Ensemble ===")

models = [
    ("CatBoost", cb_oof, cb_test),
    ("XGB", xgb_oof, xgb_test),
    ("LGB", lgb_oof, lgb_test),
    ("Stack", stack_oof, stack_test_preds)
]

for name, oof, _ in models:
    print(f"  {name}: {roc_auc_score(y, oof):.5f}")

# AUC-weighted
aucs = np.array([roc_auc_score(y, m[1]) for m in models])
weights = aucs / aucs.sum()

print(f"\nWeights: CB={weights[0]:.3f}, XGB={weights[1]:.3f}, LGB={weights[2]:.3f}, Stack={weights[3]:.3f}")

final_oof = sum(weights[i] * models[i][1] for i in range(len(models)))
final_test = sum(weights[i] * models[i][2] for i in range(len(models)))

print(f"Final OOF AUC: {roc_auc_score(y, final_oof):.5f}")

# =============================================================================
# Save
# =============================================================================
test_ids = test[ID_COL].values
submission = pd.DataFrame({
    "Id": test_ids,
    "Drafted": final_test
})

submission.to_csv("/workspace/submission_final.csv", index=False)
print(f"\nSaved submission_final.csv with {len(submission)} rows")
print(submission.head())