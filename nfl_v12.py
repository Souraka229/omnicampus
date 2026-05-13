"""
NFL Draft v12 - Push for 0.9
Aggressive feature engineering + ensemble
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

RNG, N_FOLDS = 42, 5
np.random.seed(RNG)

train = pd.read_csv("/workspace/train (3).csv")
test  = pd.read_csv("/workspace/test (5).csv")
y         = train["Drafted"].values.copy()
train_raw = train.drop(columns=["Drafted"]).copy()
n_train   = len(train_raw)
n_test    = len(test)
all_data  = pd.concat([train_raw, test], ignore_index=True)

PERF      = ["Sprint_40yd","Vertical_Jump","Bench_Press_Reps","Broad_Jump","Agility_3cone","Shuttle"]
TIME_COLS = ["Sprint_40yd","Agility_3cone","Shuttle"]
CAT_COLS  = ["School","Player_Type","Position_Type","Position"]

# ═══ FEATURE ENGINEERING ══════════════════════════════════════════════════════
# Body
all_data["BMI"]            = all_data["Weight"] / all_data["Height"]**2
all_data["Weight_Height"]  = all_data["Weight"] * all_data["Height"]
all_data["Height_sq"]      = all_data["Height"]**2
all_data["Weight_sq"]      = all_data["Weight"]**2
all_data["Age_sq"]         = all_data["Age"]**2
all_data["Age_centered"]   = all_data["Age"] - 22
all_data["Age_young"]      = (all_data["Age"] < 22).astype(int)
all_data["Age_old"]        = (all_data["Age"] > 24).astype(int)

# Missingness - KEY signal
for col in PERF+["Age"]:
    all_data[f"{col}_miss"] = all_data[col].isnull().astype(int)
all_data["n_missing"]       = all_data[PERF].isnull().sum(axis=1)
all_data["has_all_tests"]   = (all_data["n_missing"]==0).astype(int)
all_data["missing_speed"]   = all_data["Sprint_40yd"].isnull().astype(int)
all_data["missing_agility"] = (all_data["Agility_3cone"].isnull()&all_data["Shuttle"].isnull()).astype(int)
all_data["missing_jump"]    = (all_data["Vertical_Jump"].isnull()&all_data["Broad_Jump"].isnull()).astype(int)
all_data["n_miss_sq"]       = all_data["n_missing"]**2

# Performance combos
all_data["Power_Score"]    = all_data["Vertical_Jump"].fillna(0)+all_data["Broad_Jump"].fillna(0)
all_data["Agility_Sum"]    = all_data["Agility_3cone"].fillna(all_data["Agility_3cone"].median())+\
                               all_data["Shuttle"].fillna(all_data["Shuttle"].median())
all_data["Speed_per_kg"]   = all_data["Sprint_40yd"]/all_data["Weight"]
all_data["Vert_per_kg"]    = all_data["Vertical_Jump"]/(all_data["Weight"]+1e-5)
all_data["Power_per_kg"]   = all_data["Power_Score"]/(all_data["Weight"]+1e-5)
all_data["Strength_Speed"] = all_data["Bench_Press_Reps"]/(all_data["Sprint_40yd"]+1e-5)
all_data["Vert_x_Speed"]   = all_data["Vertical_Jump"]/(all_data["Sprint_40yd"]+1e-5)
all_data["Broad_x_Speed"]  = all_data["Broad_Jump"]/(all_data["Sprint_40yd"]+1e-5)
all_data["Speed_sq"]       = all_data["Sprint_40yd"]**2
all_data["Vert_sq"]        = all_data["Vertical_Jump"]**2

# Global z-scores
for col in PERF:
    mu,sd = all_data[col].mean(), all_data[col].std()+1e-9
    z = (all_data[col]-mu)/sd
    if col in TIME_COLS: z=-z
    all_data[f"{col}_z"] = z
all_data["Athleticism"]    = sum(all_data[f"{c}_z"].fillna(0) for c in PERF)
all_data["Athleticism_sq"] = all_data["Athleticism"]**2
all_data["Speed_z_sq"]     = all_data["Sprint_40yd_z"]**2

# Position percentiles
for col in PERF:
    all_data[f"{col}_pct"] = np.nan
    for pos,grp in all_data.groupby("Position"):
        valid = grp[col].notna()
        if valid.sum()>1:
            ranks = grp.loc[valid,col].rank(pct=True)
            if col in TIME_COLS: ranks=1-ranks
            all_data.loc[grp.index[valid],f"{col}_pct"] = ranks.values
all_data["Ath_pct"]    = all_data[[f"{c}_pct" for c in PERF]].mean(axis=1)
all_data["Best_pct"]   = all_data[[f"{c}_pct" for c in PERF]].max(axis=1)
all_data["Worst_pct"]  = all_data[[f"{c}_pct" for c in PERF]].min(axis=1)

# Year × Position
for col in PERF:
    all_data[f"{col}_yr_pos_pct"] = np.nan
    for (yr,pos),grp in all_data.groupby(["Year","Position"]):
        valid = grp[col].notna()
        if valid.sum()>1:
            ranks = grp.loc[valid,col].rank(pct=True)
            if col in TIME_COLS: ranks=1-ranks
            all_data.loc[grp.index[valid],f"{col}_yr_pos_pct"] = ranks.values
all_data["Ath_yr_pos_pct"] = all_data[[f"{c}_yr_pos_pct" for c in PERF]].mean(axis=1)

# Year × Position_Type
for col in PERF:
    all_data[f"{col}_yr_ptype_pct"] = np.nan
    for (yr,pt),grp in all_data.groupby(["Year","Position_Type"]):
        valid = grp[col].notna()
        if valid.sum()>1:
            ranks = grp.loc[valid,col].rank(pct=True)
            if col in TIME_COLS: ranks=1-ranks
            all_data.loc[grp.index[valid],f"{col}_yr_ptype_pct"] = ranks.values

# Year normalized z
for col in PERF:
    yr_mean = all_data.groupby("Year")[col].transform("mean")
    yr_std  = all_data.groupby("Year")[col].transform("std")+1e-9
    z=(all_data[col]-yr_mean)/yr_std
    if col in TIME_COLS: z=-z
    all_data[f"{col}_year_z"] = z

# School
school_freq = train_raw["School"].value_counts()
all_data["School_Freq"]    = all_data["School"].map(school_freq).fillna(0)
all_data["School_LogFreq"] = np.log1p(all_data["School_Freq"])

power5={"Alabama","Ohio State","Georgia","Michigan","Notre Dame","Oklahoma","LSU","Florida",
        "USC","Auburn","Penn State","Texas","Oregon","Tennessee","Florida State","Clemson",
        "Nebraska","Wisconsin","Iowa","Michigan State","Stanford","Washington","UCLA",
        "Virginia Tech","Miami","Pittsburgh","TCU","Baylor","Kansas State","Iowa State",
        "Oklahoma State","West Virginia","Texas A&M","Missouri","Arkansas","Mississippi State",
        "South Carolina","Kentucky","Vanderbilt","Mississippi"}
all_data["is_power5"] = all_data["School"].isin(power5).astype(int)

yr_school_cnt = all_data.groupby(["Year","School"])["Id"].transform("count")
all_data["yr_school_cnt"]     = yr_school_cnt
all_data["yr_school_cnt_log"] = np.log1p(yr_school_cnt)
yr_cnt = all_data.groupby("Year")["Id"].transform("count")
all_data["yr_size"] = yr_cnt

# K-fold target encoding
def kfold_te(col,target,smooth=20,n_splits=5,seed=42):
    gm=target.mean()
    tr_col=all_data.iloc[:n_train][col].values
    te_col=all_data.iloc[n_train:][col].values
    train_enc=np.full(n_train,gm)
    kf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for tr_idx,va_idx in kf.split(np.arange(n_train),target):
        means={}
        for v in np.unique(tr_col[tr_idx]):
            mask=tr_col[tr_idx]==v; cnt=mask.sum()
            means[v]=(target[tr_idx][mask].mean()*cnt+gm*smooth)/(cnt+smooth)
        train_enc[va_idx]=[means.get(v,gm) for v in tr_col[va_idx]]
    means_full={}
    for v in np.unique(tr_col):
        mask=tr_col==v; cnt=mask.sum()
        means_full[v]=(target[mask].mean()*cnt+gm*smooth)/(cnt+smooth)
    return train_enc,np.array([means_full.get(v,gm) for v in te_col])

for col in CAT_COLS:
    tr_enc,te_enc = kfold_te(col,y)
    all_data[f"{col}_rate"] = np.concatenate([tr_enc,te_enc])

all_data["School_Year"] = all_data["School"].astype(str)+"_"+all_data["Year"].astype(str)
tr_enc,te_enc = kfold_te("School_Year",y,smooth=5)
all_data["School_Year_rate"] = np.concatenate([tr_enc,te_enc])
all_data.drop(columns=["School_Year"],inplace=True)

all_data["Pos_Year"] = all_data["Position"].astype(str)+"_"+all_data["Year"].astype(str)
tr_enc,te_enc = kfold_te("Pos_Year",y,smooth=10)
all_data["Pos_Year_rate"] = np.concatenate([tr_enc,te_enc])
all_data.drop(columns=["Pos_Year"],inplace=True)

# More interactions
all_data["Pos_x_Ath"]       = all_data["Position_rate"]*all_data["Athleticism"]
all_data["PosT_x_Ath"]      = all_data["Position_Type_rate"]*all_data["Athleticism"]
all_data["Sch_x_Ath"]       = all_data["School_rate"]*all_data["Athleticism"]
all_data["Pos_x_Spd"]       = all_data["Position_rate"]*all_data["Sprint_40yd_pct"].fillna(0.5)
all_data["Pos_x_AthPct"]    = all_data["Position_rate"]*all_data["Ath_pct"].fillna(0.5)
all_data["Pos_x_Missing"]   = all_data["Position_rate"]*all_data["n_missing"]
all_data["Sch_x_Power5"]    = all_data["School_rate"]*all_data["is_power5"]
all_data["SchYr_x_Ath"]     = all_data["School_Year_rate"]*all_data["Athleticism"]
all_data["Age_x_Ath"]       = all_data["Age_centered"]*all_data["Athleticism"]
all_data["Pos_x_YrAthPct"]  = all_data["Position_rate"]*all_data["Ath_yr_pos_pct"].fillna(0.5)

# ═══ PREPARE ══════════════════════════════════════════════════════════════════
num_feat_cols = [c for c in all_data.columns if c not in ["Id"]+CAT_COLS]
all_feat_cols = num_feat_cols+CAT_COLS

cb_tr=all_data.iloc[:n_train][all_feat_cols].copy()
cb_te=all_data.iloc[n_train:][all_feat_cols].copy()
for col in CAT_COLS:
    cb_tr[col]=cb_tr[col].fillna("unknown").astype(str)
    cb_te[col]=cb_te[col].fillna("unknown").astype(str)
imp=SimpleImputer(strategy="median")
cb_tr[num_feat_cols]=imp.fit_transform(cb_tr[num_feat_cols])
cb_te[num_feat_cols]=imp.transform(cb_te[num_feat_cols])
cat_idx=[all_feat_cols.index(c) for c in CAT_COLS]

for col in CAT_COLS:
    le=LabelEncoder(); all_data[col]=le.fit_transform(all_data[col].astype(str))
imp2=SimpleImputer(strategy="median")
X_tr=imp2.fit_transform(all_data.iloc[:n_train][all_feat_cols])
X_te=imp2.transform(all_data.iloc[n_train:][all_feat_cols])
print(f"✓ {len(all_feat_cols)} features")

cv=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=RNG)

def run_cb(params, seed=RNG):
    oof=np.zeros(n_train); tp=np.zeros(n_test)
    for tr_idx,va_idx in cv.split(np.arange(n_train),y):
        m=CatBoostClassifier(**params,cat_features=cat_idx,
                             eval_metric="AUC",random_seed=seed,verbose=0)
        m.fit(cb_tr.iloc[tr_idx],y[tr_idx])
        oof[va_idx]=m.predict_proba(cb_tr.iloc[va_idx])[:,1]
        tp+=m.predict_proba(cb_te)[:,1]/N_FOLDS
    return oof,tp

def run_sk(model):
    oof=np.zeros(n_train); tp=np.zeros(n_test)
    for tr_idx,va_idx in cv.split(X_tr,y):
        model.fit(X_tr[tr_idx],y[tr_idx])
        oof[va_idx]=model.predict_proba(X_tr[va_idx])[:,1]
        tp+=model.predict_proba(X_te)[:,1]/N_FOLDS
    return oof,tp

# ── Multiple CatBoost models ────────────────────────────────────────────────────
print("CB-A..."); cba_oof,cba_tp=run_cb(dict(iterations=1200,depth=5,learning_rate=0.05,
    l2_leaf_reg=3,border_count=64,bagging_temperature=0.3,random_strength=0.5))
print(f"  {roc_auc_score(y,cba_oof):.5f}")

print("CB-B..."); cbb_oof,cbb_tp=run_cb(dict(iterations=1200,depth=6,learning_rate=0.04,
    l2_leaf_reg=6,border_count=128,bagging_temperature=0.6,random_strength=1.0),seed=43)
print(f"  {roc_auc_score(y,cbb_oof):.5f}")

print("CB-C..."); cbc_oof,cbc_tp=run_cb(dict(iterations=1200,depth=4,learning_rate=0.06,
    l2_leaf_reg=8,border_count=64,bagging_temperature=1.0,random_strength=2.0),seed=44)
print(f"  {roc_auc_score(y,cbc_oof):.5f}")

print("CB-D..."); cbd_oof,cbd_tp=run_cb(dict(iterations=1500,depth=5,learning_rate=0.03,
    l2_leaf_reg=4,border_count=128,bagging_temperature=0.5,random_strength=0.8),seed=45)
print(f"  {roc_auc_score(y,cbd_oof):.5f}")

print("CB-E..."); cbe_oof,cbe_tp=run_cb(dict(iterations=1000,depth=7,learning_rate=0.025,
    l2_leaf_reg=5,border_count=64,bagging_temperature=0.8,random_strength=1.5),seed=46)
print(f"  {roc_auc_score(y,cbe_oof):.5f}")

# XGBoost
print("XGB..."); xgb_oof,xgb_tp=run_sk(xgb.XGBClassifier(
    n_estimators=1000,max_depth=4,learning_rate=0.025,
    subsample=0.75,colsample_bytree=0.65,min_child_weight=5,gamma=0.1,
    reg_alpha=0.3,reg_lambda=2.0,use_label_encoder=False,eval_metric="auc",
    random_state=RNG,n_jobs=-1,verbosity=0))
print(f"  {roc_auc_score(y,xgb_oof):.5f}")

# LightGBM
print("LGB..."); lgb_oof,lgb_tp=run_sk(lgb.LGBMClassifier(
    n_estimators=1000,max_depth=5,learning_rate=0.025,num_leaves=31,
    subsample=0.75,colsample_bytree=0.65,min_child_samples=20,
    reg_alpha=0.3,reg_lambda=2.0,random_state=RNG,n_jobs=-1,verbose=-1))
print(f"  {roc_auc_score(y,lgb_oof):.5f}")

# Extra Trees
print("ET..."); et_oof,et_tp=run_sk(ExtraTreesClassifier(
    n_estimators=500,max_depth=12,min_samples_leaf=4,
    random_state=RNG,n_jobs=-1))
print(f"  {roc_auc_score(y,et_oof):.5f}")

# Random Forest
print("RF..."); rf_oof,rf_tp=run_sk(RandomForestClassifier(
    n_estimators=400,max_depth=10,min_samples_leaf=5,
    random_state=RNG,n_jobs=-1))
print(f"  {roc_auc_score(y,rf_oof):.5f}")

# MLP
print("MLP...")
sc=StandardScaler(); Xs=sc.fit_transform(X_tr); Xts=sc.transform(X_te)
mlp_oof=np.zeros(n_train); mlp_tp=np.zeros(n_test)
for tr_idx,va_idx in cv.split(Xs,y):
    m=MLPClassifier(hidden_layer_sizes=(256,128,64),alpha=0.01,
                    learning_rate_init=0.001,max_iter=300,random_state=RNG,
                    early_stopping=True,validation_fraction=0.1,n_iter_no_change=15)
    m.fit(Xs[tr_idx],y[tr_idx])
    mlp_oof[va_idx]=m.predict_proba(Xs[va_idx])[:,1]
    mlp_tp+=m.predict_proba(Xts)[:,1]/N_FOLDS
print(f"  {roc_auc_score(y,mlp_oof):.5f}")

# Stacking
base_oofs =[cba_oof,cbb_oof,cbc_oof,cbd_oof,cbe_oof,xgb_oof,lgb_oof,et_oof,rf_oof,mlp_oof]
base_tests=[cba_tp,cbb_tp,cbc_tp,cbd_tp,cbe_tp,xgb_tp,lgb_tp,et_tp,rf_tp,mlp_tp]
meta_tr=np.column_stack(base_oofs); meta_te=np.column_stack(base_tests)

cv3=StratifiedKFold(n_splits=3,shuffle=True,random_state=RNG)
lr_oof=np.zeros(n_train); lr_te=np.zeros(n_test)
for tr_idx,va_idx in cv3.split(meta_tr,y):
    m=LogisticRegression(C=0.1,max_iter=1000,random_state=RNG)
    m.fit(meta_tr[tr_idx],y[tr_idx])
    lr_oof[va_idx]=m.predict_proba(meta_tr[va_idx])[:,1]
    lr_te+=m.predict_proba(meta_te)[:,1]/3
print(f"\nLR Stack: {roc_auc_score(y,lr_oof):.5f}")

# Blend
aucs=np.array([roc_auc_score(y,o) for o in base_oofs])
w=aucs/aucs.sum()
bl_oof =sum(w[i]*base_oofs[i] for i in range(len(w)))
bl_test=sum(w[i]*base_tests[i] for i in range(len(w)))
print(f"Blend: {roc_auc_score(y,bl_oof):.5f}")

# Pseudo-labeling
THRESH=0.85
pmask=(bl_test>THRESH)|(bl_test<(1-THRESH))
print(f"Pseudo: {pmask.sum()} samples")
if pmask.sum()>30:
    plabels=(bl_test[pmask]>0.5).astype(int)
    cb_aug=pd.concat([cb_tr,cb_te[pmask].reset_index(drop=True)],ignore_index=True)
    y_aug=np.concatenate([y,plabels])
    aug_idx=np.arange(n_train,len(y_aug))
    p_oof=np.zeros(n_train); p_tp=np.zeros(n_test)
    for tr_idx,va_idx in cv.split(np.arange(n_train),y):
        full_tr=np.concatenate([tr_idx,aug_idx])
        m=CatBoostClassifier(iterations=1200,depth=5,learning_rate=0.05,l2_leaf_reg=3,
                             cat_features=cat_idx,eval_metric="AUC",random_seed=RNG,verbose=0)
        m.fit(cb_aug.iloc[full_tr],y_aug[full_tr])
        p_oof[va_idx]=m.predict_proba(cb_tr.iloc[va_idx])[:,1]
        p_tp+=m.predict_proba(cb_te)[:,1]/N_FOLDS
    p_auc=roc_auc_score(y,p_oof)
    print(f"Pseudo: {p_auc:.5f}")

# Final
all_f=[(bl_oof,bl_test),(lr_oof,lr_te),(p_oof,p_tp)]
fa=np.array([roc_auc_score(y,o) for o,_ in all_f if roc_auc_score(y,o)>0]); fw=fa/fa.sum()
mega_oof =sum(fw[i]*all_f[i][0] for i in range(len(all_f)))
mega_test=sum(fw[i]*all_f[i][1] for i in range(len(all_f)))
print(f"\n🏆 FINAL: {roc_auc_score(y,mega_oof):.5f}")

sub=pd.DataFrame({"Id":test["Id"],"Drafted":mega_test})
sub.to_csv("/workspace/submission_final.csv",index=False)
print(f"✓ submission_final.csv — {len(sub)} rows")