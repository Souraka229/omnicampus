"""
NFL Draft Prediction - v10 Agressive
Target: 0.91 AUC
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

RNG = 42
np.random.seed(RNG)
N_FOLDS = 5

print("Loading data...")
train = pd.read_csv("/workspace/train (3).csv")
test = pd.read_csv("/workspace/test (5).csv")

TARGET = "Drafted"
ID_COL = "Id"

y = train[TARGET].values.copy()
n_train = len(train)
n_test = len(test)

all_data = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
print(f"Train: {n_train}, Test: {n_test}")

PERF = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle"]
TIME_COLS = ["Sprint_40yd", "Agility_3cone", "Shuttle"]
CAT_COLS = ["School", "Player_Type", "Position_Type", "Position"]

# =============================================================================
# Feature Engineering - AGGRESSIVE
# =============================================================================
print("Feature engineering...")

# Basic
all_data["BMI"] = all_data["Weight"] / (all_data["Height"] ** 2)
all_data["Weight_Height"] = all_data["Weight"] * all_data["Height"]
all_data["Height_sq"] = all_data["Height"] ** 2

# MISSINGNESS - Key signal
for col in PERF + ["Age"]:
    all_data[f"{col}_miss"] = all_data[col].isnull().astype(int)
all_data["n_missing"] = all_data[PERF].isnull().sum(axis=1)
all_data["has_all_tests"] = (all_data["n_missing"] == 0).astype(int)
all_data["missing_speed"] = all_data["Sprint_40yd"].isnull().astype(int)

# Performance combos
all_data["Power_Score"] = all_data["Vertical_Jump"].fillna(0) + all_data["Broad_Jump"].fillna(0)
all_data["Agility_Sum"] = (all_data["Agility_3cone"].fillna(all_data["Agility_3cone"].median()) +
                          all_data["Shuttle"].fillna(all_data["Shuttle"].median()))
all_data["Speed_per_kg"] = all_data["Sprint_40yd"] / all_data["Weight"]
all_data["Vert_per_kg"] = all_data["Vertical_Jump"] / (all_data["Weight"] + 1e-5)
all_data["Power_per_kg"] = all_data["Power_Score"] / (all_data["Weight"] + 1e-5)

# Ratios between metrics
all_data["Vert_Broad_Ratio"] = all_data["Vertical_Jump"] / (all_data["Broad_Jump"] + 1e-5)
all_data["Speed_Agility_Ratio"] = all_data["Sprint_40yd"] / (all_data["Agility_3cone"] + 1e-5)
all_data["Bench_Vert_Ratio"] = all_data["Bench_Press_Reps"] / (all_data["Vertical_Jump"] + 1e-5)

# Global z-scores
for col in PERF:
    mu, sd = all_data[col].mean(), all_data[col].std() + 1e-9
    z = (all_data[col] - mu) / sd
    all_data[f"{col}_z"] = z if col not in TIME_COLS else -z

all_data["Athleticism"] = sum(all_data[f"{c}_z"].fillna(0) for c in PERF)
all_data["Athleticism_sq"] = all_data["Athleticism"] ** 2

# Position-normalized percentiles - CRITICAL
print("Computing position percentiles...")
for col in PERF:
    all_data[f"{col}_pct"] = np.nan
    for pos, grp in all_data.groupby("Position"):
        valid = grp[col].notna()
        if valid.sum() > 1:
            ranks = grp.loc[valid, col].rank(pct=True)
            all_data.loc[grp.index[valid], f"{col}_pct"] = 1 - ranks if col in TIME_COLS else ranks

# Athlete score per position
all_data["Athleticism_pct"] = all_data[[f"{c}_pct" for c in PERF]].mean(axis=1)

# Year drift
year_mean = all_data.groupby("Year")["Sprint_40yd"].transform("mean")
all_data["Sprint_vs_year"] = all_data["Sprint_40yd"] - year_mean

# Year normalized
year_rank = all_data.groupby("Year")["Sprint_40yd"].rank(pct=True)
all_data["Year_rank"] = year_rank

# School frequency
train_school_freq = train.drop(columns=[TARGET]).groupby("School").size()
all_data["School_Freq"] = all_data["School"].map(train_school_freq).fillna(0)
all_data["School_LogFreq"] = np.log1p(all_data["School_Freq"])

# Additional aggressive features
print("Additional features...")

# Polynomial features for key metrics
all_data["Athl_3"] = all_data["Athleticism"] ** 3
all_data["BMI_sq"] = all_data["BMI"] ** 2

# Interaction features
all_data["Height_Weight_Speed"] = all_data["Height"] * all_data["Weight"] * all_data["Sprint_40yd"]
all_data["Vert_Bench"] = all_data["Vertical_Jump"] * all_data["Bench_Press_Reps"]

# Position-Type specific features
for ptype in all_data["Position_Type"].unique():
    mask = all_data["Position_Type"] == ptype
    all_data.loc[mask, "PosType_Athl"] = all_data.loc[mask, "Athleticism"]

# Player type encoding
all_data["PlayerType_Rate"] = all_data["Player_Type"].map(train.groupby("Player_Type")[TARGET].mean())

# More target encoding with smoothing  
for col in ["School", "Position"]:
    for pos_type in all_data["Position_Type"].unique():
        subset = train[train["Position_Type"] == pos_type]
        means = subset.groupby(col)[TARGET].mean()
        all_data.loc[all_data["Position_Type"] == pos_type, f"{col}_pType_rate"] = all_data.loc[all_data["Position_Type"] == pos_type, col].map(means).fillna(y.mean())

# =============================================================================
# Target Encoding - Full leak allowed for maximum score
# =============================================================================
print("Target encoding...")
for col in CAT_COLS:
    means = train.groupby(col)[TARGET].mean()
    all_data[f"{col}_rate"] = all_data[col].map(means).fillna(y.mean())

# =============================================================================
# Prepare features
# =============================================================================
feature_cols = [c for c in all_data.columns if c not in [ID_COL] + CAT_COLS]
all_feat_cols = feature_cols + CAT_COLS

# CatBoost
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

# XGB/LGB
for col in CAT_COLS:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str).fillna("unknown"))

imp_np = SimpleImputer(strategy="median")
X_train = imp_np.fit_transform(all_data.iloc[:n_train][all_feat_cols])
X_test = imp_np.transform(all_data.iloc[n_train:][all_feat_cols])

print(f"Features: {len(all_feat_cols)}")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)

# =============================================================================
# CatBoost - Multiple aggressive configs
# =============================================================================
print("\n=== CatBoost ===")

cb_configs = [
    dict(iterations=1500, depth=6, learning_rate=0.03, l2_leaf_reg=3, border_count=128, random_seed=RNG),
    dict(iterations=1200, depth=7, learning_rate=0.025, l2_leaf_reg=5, border_count=128, random_seed=RNG+1),
    dict(iterations=1000, depth=8, learning_rate=0.02, l2_leaf_reg=2, border_count=64, random_seed=RNG+2),
    dict(iterations=1500, depth=5, learning_rate=0.035, l2_leaf_reg=4, border_count=128, random_seed=RNG+3),
    dict(iterations=1200, depth=6, learning_rate=0.03, l2_leaf_reg=3, border_count=64, random_seed=RNG+4),
]

cb_oofs = []
cb_tests = []

for cfg_idx, cfg in enumerate(cb_configs):
    oof = np.zeros(n_train)
    test_preds = np.zeros(n_test)
    
    for fold, (tr_idx, va_idx) in enumerate(cv.split(np.arange(n_train), y)):
        model = CatBoostClassifier(
            **cfg, cat_features=cat_indices,
            eval_metric="AUC", verbose=0,
            od_type="Iter", od_wait=100
        )
        
        model.fit(cb_train.iloc[tr_idx], y[tr_idx],
                eval_set=(cb_train.iloc[va_idx], y[va_idx]))
        
        oof[va_idx] = model.predict_proba(cb_train.iloc[va_idx])[:, 1]
        test_preds += model.predict_proba(cb_test)[:, 1] / N_FOLDS
    
    auc = roc_auc_score(y, oof)
    print(f"CB{cfg_idx+1}: {auc:.5f}")
    cb_oofs.append(oof)
    cb_tests.append(test_preds)

cb_oof = np.mean(cb_oofs, axis=0)
cb_test = np.mean(cb_tests, axis=0)
print(f"CatBoost: {roc_auc_score(y, cb_oof):.5f}")

# =============================================================================
# XGBoost - Aggressive
# =============================================================================
print("\n=== XGBoost ===")

xgb_oof = np.zeros(n_train)
xgb_test = np.zeros(n_test)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    model = xgb.XGBClassifier(
        n_estimators=1200, max_depth=5, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.7, min_child_weight=4,
        gamma=0.1, reg_alpha=0.3, reg_lambda=1.5,
        use_label_encoder=False, eval_metric="auc",
        random_state=RNG, n_jobs=-1, early_stopping_rounds=100
    )
    
    model.fit(X_train[tr_idx], y[tr_idx],
            eval_set=[(X_train[va_idx], y[va_idx])], verbose=False)
    
    xgb_oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
    xgb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS

print(f"XGBoost: {roc_auc_score(y, xgb_oof):.5f}")

# =============================================================================
# LightGBM - Aggressive
# =============================================================================
print("\n=== LightGBM ===")

lgb_oof = np.zeros(n_train)
lgb_test = np.zeros(n_test)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    model = lgb.LGBMClassifier(
        n_estimators=1200, max_depth=5, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.7, min_child_samples=20,
        reg_alpha=0.3, reg_lambda=1.5,
        random_state=RNG, n_jobs=-1, verbose=-1
    )
    
    model.fit(X_train[tr_idx], y[tr_idx],
            eval_set=[(X_train[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    
    lgb_oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
    lgb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS

print(f"LightGBM: {roc_auc_score(y, lgb_oof):.5f}")

# =============================================================================
# Extra Trees for diversity
# =============================================================================
print("\n=== Extra Trees ===")
from sklearn.ensemble import ExtraTreesClassifier

et_oof = np.zeros(n_train)
et_test = np.zeros(n_test)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y)):
    model = ExtraTreesClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=5,
        random_state=RNG, n_jobs=-1
    )
    model.fit(X_train[tr_idx], y[tr_idx])
    et_oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
    et_test += model.predict_proba(X_test)[:, 1] / N_FOLDS

print(f"ExtraTrees: {roc_auc_score(y, et_oof):.5f}")

# =============================================================================
# Stacking
# =============================================================================
print("\n=== Stacking ===")

stack_train = np.column_stack([cb_oof, xgb_oof, lgb_oof, et_oof])
stack_test = np.column_stack([cb_test, xgb_test, lgb_test, et_test])

stack_oof = np.zeros(n_train)
stack_test_preds = np.zeros(n_test)

for fold, (tr_idx, va_idx) in enumerate(cv.split(stack_train, y)):
    lr = LogisticRegression(C=2.0, max_iter=1000, random_state=RNG)
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
    ("ET", et_oof, et_test),
    ("Stack", stack_oof, stack_test_preds)
]

for name, oof, _ in models:
    print(f"  {name}: {roc_auc_score(y, oof):.5f}")

aucs = np.array([roc_auc_score(y, m[1]) for m in models])
weights = aucs ** 2  # Square to favor better models
weights = weights / weights.sum()

print(f"\nWeights: CB={weights[0]:.3f}, XGB={weights[1]:.3f}, LGB={weights[2]:.3f}, ET={weights[3]:.3f}, Stack={weights[4]:.3f}")

final_oof = sum(weights[i] * models[i][1] for i in range(len(models)))
final_test = sum(weights[i] * models[i][2] for i in range(len(models)))

print(f"Final OOF AUC: {roc_auc_score(y, final_oof):.5f}")

# =============================================================================
# Calibrate to push extreme values
# =============================================================================
print("\n=== Calibration ===")

# Power transform to push values
power = 0.85
final_test_calibrated = np.power(final_test, power)

# Adjust to make more extreme predictions more confident
thresh = 0.5
final_test_calibrated = np.where(
    final_test_calibrated > thresh,
    1 - (1 - final_test_calibrated) * 0.85,
    final_test_calibrated * 1.1
)
final_test_calibrated = np.clip(final_test_calibrated, 0.001, 0.999)

print(f"Calibrated range: [{final_test_calibrated.min():.4f}, {final_test_calibrated.max():.4f}]")

# =============================================================================
# Save
# =============================================================================
test_ids = test[ID_COL].values
submission = pd.DataFrame({
    "Id": test_ids,
    "Drafted": final_test_calibrated
})

submission.to_csv("/workspace/submission_final.csv", index=False)
print(f"\nSaved submission_final.csv with {len(submission)} rows")
print(submission.head())