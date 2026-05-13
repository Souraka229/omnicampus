"""
NFL Draft v16 STABLE - Conservative but effective
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

RNG, N_FOLDS = 42, 5
np.random.seed(RNG)

train = pd.read_csv("train (3).csv")
test  = pd.read_csv("test (5).csv")
y = train["Drafted"].values.copy()
train_raw = train.drop(columns=["Drafted"]).copy()
n_train = len(train_raw)
all_data = pd.concat([train_raw, test], ignore_index=True)

PERF = ["Sprint_40yd","Vertical_Jump","Bench_Press_Reps","Broad_Jump","Agility_3cone","Shuttle"]
TIME_COLS = ["Sprint_40yd","Agility_3cone","Shuttle"]
CAT_COLS = ["School","Player_Type","Position_Type","Position"]

# ===== FEATURES - CONSERVATIVE =====
all_data["BMI"] = all_data["Weight"] / all_data["Height"]**2
all_data["Weight_Height"] = all_data["Weight"] * all_data["Height"]
all_data["Age_centered"] = all_data["Age"] - 22

# Missingness - KEY
for col in PERF+["Age"]:
    all_data[f"{col}_miss"] = all_data[col].isnull().astype(int)
all_data["n_missing"] = all_data[PERF].isnull().sum(axis=1)
all_data["has_all_tests"] = (all_data["n_missing"] == 0).astype(int)

# Performance combos
all_data["Power_Score"] = all_data["Vertical_Jump"].fillna(0) + all_data["Broad_Jump"].fillna(0)
all_data["Agility_Sum"] = (all_data["Agility_3cone"].fillna(all_data["Agility_3cone"].median()) + 
                           all_data["Shuttle"].fillna(all_data["Shuttle"].median()))
all_data["Speed_per_kg"] = all_data["Sprint_40yd"] / all_data["Weight"]
all_data["Vert_per_kg"] = all_data["Vertical_Jump"] / (all_data["Weight"] + 1e-5)

# Z-scores
for col in PERF:
    mu, sd = all_data[col].mean(), all_data[col].std() + 1e-9
    z = (all_data[col] - mu) / sd
    if col in TIME_COLS: z = -z
    all_data[f"{col}_z"] = z
all_data["Athleticism"] = sum(all_data[f"{c}_z"].fillna(0) for c in PERF)

# Position percentiles
for col in PERF:
    all_data[f"{col}_pct"] = np.nan
    for pos, grp in all_data.groupby("Position"):
        valid = grp[col].notna()
        if valid.sum() > 1:
            ranks = grp.loc[valid, col].rank(pct=True)
            if col in TIME_COLS: ranks = 1 - ranks
            all_data.loc[grp.index[valid], f"{col}_pct"] = ranks.values
all_data["Ath_pct"] = all_data[[f"{c}_pct" for c in PERF]].mean(axis=1)

# Year x Position percentiles
for col in PERF:
    all_data[f"{col}_yr_pos_pct"] = np.nan
    for (yr, pos), grp in all_data.groupby(["Year", "Position"]):
        valid = grp[col].notna()
        if valid.sum() > 1:
            ranks = grp.loc[valid, col].rank(pct=True)
            if col in TIME_COLS: ranks = 1 - ranks
            all_data.loc[grp.index[valid], f"{col}_yr_pos_pct"] = ranks.values
all_data["Ath_yr_pos_pct"] = all_data[[f"{c}_yr_pos_pct" for c in PERF]].mean(axis=1)

# School
school_freq = train_raw["School"].value_counts()
all_data["School_Freq"] = all_data["School"].map(school_freq).fillna(0)
all_data["School_LogFreq"] = np.log1p(all_data["School_Freq"])

# Target encoding - STRICT K-FOLD
def kfold_te(col, target, smooth=20, n_splits=5, seed=42):
    gm = target.mean()
    tr_col = all_data.iloc[:n_train][col].values
    te_col = all_data.iloc[n_train:][col].values
    train_enc = np.full(n_train, gm)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(np.arange(n_train), target):
        means = {}
        for v in np.unique(tr_col[tr_idx]):
            mask = tr_col[tr_idx] == v
            cnt = mask.sum()
            means[v] = (target[tr_idx][mask].mean() * cnt + gm * smooth) / (cnt + smooth)
        train_enc[va_idx] = [means.get(v, gm) for v in tr_col[va_idx]]
    means_full = {}
    for v in np.unique(tr_col):
        mask = tr_col == v
        cnt = mask.sum()
        means_full[v] = (target[mask].mean() * cnt + gm * smooth) / (cnt + smooth)
    return train_enc, np.array([means_full.get(v, gm) for v in te_col])

for col in CAT_COLS:
    tr_enc, te_enc = kfold_te(col, y)
    all_data[f"{col}_rate"] = np.concatenate([tr_enc, te_enc])

# Interactions
all_data["Pos_x_Ath"] = all_data["Position_rate"] * all_data["Athleticism"]
all_data["Pos_x_AthPct"] = all_data["Position_rate"] * all_data["Ath_pct"].fillna(0.5)
all_data["Age_x_Ath"] = all_data["Age_centered"] * all_data["Athleticism"]

# ===== PREPARE =====
num_feat_cols = [c for c in all_data.columns if c not in ["Id"] + CAT_COLS]
all_feat_cols = num_feat_cols + CAT_COLS

cb_tr = all_data.iloc[:n_train][all_feat_cols].copy()
cb_te = all_data.iloc[n_train:][all_feat_cols].copy()
for col in CAT_COLS:
    cb_tr[col] = cb_tr[col].fillna("unknown").astype(str)
    cb_te[col] = cb_te[col].fillna("unknown").astype(str)

imp = SimpleImputer(strategy="median")
cb_tr[num_feat_cols] = imp.fit_transform(cb_tr[num_feat_cols])
cb_te[num_feat_cols] = imp.transform(cb_te[num_feat_cols])
cat_idx = [all_feat_cols.index(c) for c in CAT_COLS]

for col in CAT_COLS:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str).fillna("unknown"))

imp2 = SimpleImputer(strategy="median")
X_tr = imp2.fit_transform(all_data.iloc[:n_train][all_feat_cols])
X_te = imp2.transform(all_data.iloc[n_train:][all_feat_cols])
print(f"✓ {len(all_feat_cols)} features")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)

# ===== TRAIN =====
# Multiple CatBoost configs with different seeds
configs = [
    dict(iterations=800, depth=5, learning_rate=0.05, l2_leaf_reg=5, border_count=64),
    dict(iterations=800, depth=6, learning_rate=0.04, l2_leaf_reg=6, border_count=64),
    dict(iterations=800, depth=4, learning_rate=0.06, l2_leaf_reg=8, border_count=32),
]
seeds = [42, 123, 456, 789]

cb_oofs, cb_tests = [], []
for cfg in configs:
    for s in seeds:
        oof = np.zeros(n_train)
        test_pred = np.zeros(len(X_te))
        for tr_idx, va_idx in cv.split(np.arange(n_train), y):
            m = CatBoostClassifier(**cfg, cat_features=cat_idx, eval_metric="AUC", 
                                   random_seed=s, verbose=0, bagging_temperature=0.5)
            m.fit(cb_tr.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict_proba(cb_tr.iloc[va_idx])[:, 1]
            test_pred += m.predict_proba(cb_te)[:, 1] / N_FOLDS
        cb_oofs.append(oof)
        cb_tests.append(test_pred)

cb_oof = np.mean(cb_oofs, axis=0)
cb_test = np.mean(cb_tests, axis=0)
print(f"CB: {roc_auc_score(y, cb_oof):.5f}")

# XGBoost
xgb_oofs, xgb_tests = [], []
for s in seeds:
    oof = np.zeros(n_train)
    test_pred = np.zeros(len(X_te))
    for tr_idx, va_idx in cv.split(X_tr, y):
        m = xgb.XGBClassifier(n_estimators=800, max_depth=4, learning_rate=0.03,
                              subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
                              gamma=0.1, reg_alpha=0.3, reg_lambda=2.0,
                              use_label_encoder=False, eval_metric="auc", random_state=s, n_jobs=-1)
        m.fit(X_tr[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]
        test_pred += m.predict_proba(X_te)[:, 1] / N_FOLDS
    xgb_oofs.append(oof)
    xgb_tests.append(test_pred)

xgb_oof = np.mean(xgb_oofs, axis=0)
xgb_test = np.mean(xgb_tests, axis=0)
print(f"XGB: {roc_auc_score(y, xgb_oof):.5f}")

# LightGBM
lgb_oofs, lgb_tests = [], []
for s in seeds:
    oof = np.zeros(n_train)
    test_pred = np.zeros(len(X_te))
    for tr_idx, va_idx in cv.split(X_tr, y):
        m = lgb.LGBMClassifier(n_estimators=800, max_depth=4, learning_rate=0.03,
                               subsample=0.7, colsample_bytree=0.6, min_child_samples=20,
                               reg_alpha=0.3, reg_lambda=2.0, random_state=s, n_jobs=-1, verbose=-1)
        m.fit(X_tr[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X_tr[va_idx])[:, 1]
        test_pred += m.predict_proba(X_te)[:, 1] / N_FOLDS
    lgb_oofs.append(oof)
    lgb_tests.append(test_pred)

lgb_oof = np.mean(lgb_oofs, axis=0)
lgb_test = np.mean(lgb_tests, axis=0)
print(f"LGB: {roc_auc_score(y, lgb_oof):.5f}")

# Blend
base_oofs = [cb_oof, xgb_oof, lgb_oof]
base_tests = [cb_test, xgb_test, lgb_test]
aucs = np.array([roc_auc_score(y, o) for o in base_oofs])
w = aucs / aucs.sum()
bl_oof = sum(w[i] * base_oofs[i] for i in range(3))
bl_test = sum(w[i] * base_tests[i] for i in range(3))
print(f"Blend: {roc_auc_score(y, bl_oof):.5f}")

# LR Stack
meta_tr = np.column_stack(base_oofs)
meta_te = np.column_stack(base_tests)
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)
lr_oof = np.zeros(n_train)
lr_test = np.zeros(len(X_te))
for tr_idx, va_idx in cv3.split(meta_tr, y):
    m = LogisticRegression(C=0.1, max_iter=1000, random_state=RNG)
    m.fit(meta_tr[tr_idx], y[tr_idx])
    lr_oof[va_idx] = m.predict_proba(meta_tr[va_idx])[:, 1]
    lr_test += m.predict_proba(meta_te)[:, 1] / 3
print(f"LR Stack: {roc_auc_score(y, lr_oof):.5f}")

# Final
all_oof = [bl_oof, lr_oof, cb_oof]
all_test = [bl_test, lr_test, cb_test]
fa = np.array([roc_auc_score(y, o) for o in all_oof])
fw = fa / fa.sum()
final_oof = sum(fw[i] * all_oof[i] for i in range(3))
final_test = sum(fw[i] * all_test[i] for i in range(3))
print(f"\n🏆 FINAL: {roc_auc_score(y, final_oof):.5f}")

sub = pd.DataFrame({"Id": test["Id"], "Drafted": final_test})
sub.to_csv("submission_final.csv", index=False)
print(f"✓ submission_final.csv - {len(sub)} rows")