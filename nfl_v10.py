"""
NFL Draft v10 FAST — Seed averaging 9 CB + 3 XGB/LGB, 5-fold
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

train = pd.read_csv("/workspace/train (3).csv")
test  = pd.read_csv("/workspace/test (5).csv")
y         = train["Drafted"].values.copy()
train_raw = train.drop(columns=["Drafted"]).copy()
n_train   = len(train_raw)
all_data  = pd.concat([train_raw, test], ignore_index=True)

PERF      = ["Sprint_40yd","Vertical_Jump","Bench_Press_Reps","Broad_Jump","Agility_3cone","Shuttle"]
TIME_COLS = ["Sprint_40yd","Agility_3cone","Shuttle"]
CAT_COLS  = ["School","Player_Type","Position_Type","Position"]

# Body
all_data["BMI"]            = all_data["Weight"]/all_data["Height"]**2
all_data["Weight_Height"]  = all_data["Weight"]*all_data["Height"]
all_data["Age_sq"]         = all_data["Age"]**2
all_data["Age_young"]      = (all_data["Age"]<22).astype(int)
all_data["Age_centered"]   = all_data["Age"]-22

# Missingness
for col in PERF+["Age"]:
    all_data[f"{col}_miss"]=all_data[col].isnull().astype(int)
all_data["n_missing"]       = all_data[PERF].isnull().sum(axis=1)
all_data["has_all_tests"]   = (all_data["n_missing"]==0).astype(int)
all_data["missing_speed"]   = all_data["Sprint_40yd"].isnull().astype(int)
all_data["missing_agility"] = (all_data["Agility_3cone"].isnull()&all_data["Shuttle"].isnull()).astype(int)
all_data["n_miss_sq"]       = all_data["n_missing"]**2

# Performance combos
all_data["Power_Score"]    = all_data["Vertical_Jump"].fillna(0)+all_data["Broad_Jump"].fillna(0)
all_data["Agility_Sum"]    = (all_data["Agility_3cone"].fillna(all_data["Agility_3cone"].median())+
                               all_data["Shuttle"].fillna(all_data["Shuttle"].median()))
all_data["Speed_per_kg"]   = all_data["Sprint_40yd"]/all_data["Weight"]
all_data["Vert_per_kg"]    = all_data["Vertical_Jump"]/(all_data["Weight"]+1e-5)
all_data["Power_per_kg"]   = all_data["Power_Score"]/(all_data["Weight"]+1e-5)
all_data["Strength_Speed"] = all_data["Bench_Press_Reps"]/(all_data["Sprint_40yd"]+1e-5)
all_data["Vert_x_Speed"]   = all_data["Vertical_Jump"]/(all_data["Sprint_40yd"]+1e-5)

# Z-scores
for col in PERF:
    mu,sd=all_data[col].mean(),all_data[col].std()+1e-9
    z=(all_data[col]-mu)/sd
    if col in TIME_COLS: z=-z
    all_data[f"{col}_z"]=z
all_data["Athleticism"]=sum(all_data[f"{c}_z"].fillna(0) for c in PERF)
all_data["Athleticism_sq"]=all_data["Athleticism"]**2

# Position percentiles
for col in PERF:
    all_data[f"{col}_pct"]=np.nan
    for pos,grp in all_data.groupby("Position"):
        valid=grp[col].notna()
        if valid.sum()>1:
            ranks=grp.loc[valid,col].rank(pct=True)
            if col in TIME_COLS: ranks=1-ranks
            all_data.loc[grp.index[valid],f"{col}_pct"]=ranks.values
all_data["Ath_pct"] =all_data[[f"{c}_pct" for c in PERF]].mean(axis=1)
all_data["Best_pct"]=all_data[[f"{c}_pct" for c in PERF]].max(axis=1)

# Year z
for col in PERF:
    yr_mean=all_data.groupby("Year")[col].transform("mean")
    yr_std =all_data.groupby("Year")[col].transform("std")+1e-9
    z=(all_data[col]-yr_mean)/yr_std
    if col in TIME_COLS: z=-z
    all_data[f"{col}_year_z"]=z

# School
school_freq=train_raw["School"].value_counts()
all_data["School_Freq"]    =all_data["School"].map(school_freq).fillna(0)
all_data["School_LogFreq"] =np.log1p(all_data["School_Freq"])
power5={"Alabama","Ohio State","Georgia","Michigan","Notre Dame","Oklahoma","LSU","Florida",
        "USC","Auburn","Penn State","Texas","Oregon","Tennessee","Florida State","Clemson",
        "Nebraska","Wisconsin","Iowa","Michigan State","Stanford","Washington","UCLA",
        "Virginia Tech","Miami","Pittsburgh","TCU","Baylor","Kansas State","Iowa State",
        "Oklahoma State","West Virginia","Texas A&M","Missouri","Arkansas","Mississippi State",
        "South Carolina","Kentucky","Vanderbilt","Mississippi"}
all_data["is_power5"]=all_data["School"].isin(power5).astype(int)
all_data["yr_school_cnt"]=all_data.groupby(["Year","School"])["Id"].transform("count")
all_data["yr_school_log"] =np.log1p(all_data["yr_school_cnt"])

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
    tr_enc,te_enc=kfold_te(col,y)
    all_data[f"{col}_rate"]=np.concatenate([tr_enc,te_enc])

# Interactions
all_data["Pos_x_Ath"]    =all_data["Position_rate"]*all_data["Athleticism"]
all_data["PosT_x_Ath"]   =all_data["Position_Type_rate"]*all_data["Athleticism"]
all_data["Sch_x_Ath"]    =all_data["School_rate"]*all_data["Athleticism"]
all_data["Pos_x_AthPct"] =all_data["Position_rate"]*all_data["Ath_pct"].fillna(0.5)
all_data["Pos_x_Missing"]=all_data["Position_rate"]*all_data["n_missing"]
all_data["Age_x_Ath"]    =all_data["Age_centered"]*all_data["Athleticism"]

num_feat_cols=[c for c in all_data.columns if c not in ["Id"]+CAT_COLS]
all_feat_cols=num_feat_cols+CAT_COLS

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

# 3 CB configs × 3 seeds = 9 models
configs=[
    dict(iterations=800,depth=5,learning_rate=0.05,l2_leaf_reg=3,border_count=64,bagging_temperature=0.3,random_strength=0.5),
    dict(iterations=800,depth=6,learning_rate=0.04,l2_leaf_reg=6,border_count=128,bagging_temperature=0.6,random_strength=1.0),
    dict(iterations=800,depth=4,learning_rate=0.06,l2_leaf_reg=8,border_count=64,bagging_temperature=1.0,random_strength=2.0),
]
cb_seeds=[42,123,777]

all_cb_oofs=[]; all_cb_tps=[]
for ci,cfg in enumerate(configs):
    for s in cb_seeds:
        oof=np.zeros(n_train); tp=np.zeros(len(X_te))
        for tr_idx,va_idx in cv.split(np.arange(n_train),y):
            m=CatBoostClassifier(**cfg,cat_features=cat_idx,eval_metric="AUC",random_seed=s,verbose=0)
            m.fit(cb_tr.iloc[tr_idx],y[tr_idx])
            oof[va_idx]=m.predict_proba(cb_tr.iloc[va_idx])[:,1]
            tp+=m.predict_proba(cb_te)[:,1]/N_FOLDS
        all_cb_oofs.append(oof); all_cb_tps.append(tp)
        print(f"CB cfg{ci+1} s{s}: {roc_auc_score(y,oof):.5f}")

cb_oof=np.mean(all_cb_oofs,axis=0); cb_tp=np.mean(all_cb_tps,axis=0)
print(f"CB 9-model avg: {roc_auc_score(y,cb_oof):.5f}")

# XGB 3-seed
xgb_seeds=[42,123,777]
xgb_oofs=[]; xgb_tps=[]
for s in xgb_seeds:
    oof=np.zeros(n_train); tp=np.zeros(len(X_te))
    for tr_idx,va_idx in cv.split(X_tr,y):
        m=xgb.XGBClassifier(n_estimators=800,max_depth=4,learning_rate=0.03,subsample=0.75,
         colsample_bytree=0.65,min_child_weight=5,gamma=0.1,reg_alpha=0.3,
         reg_lambda=2.0,use_label_encoder=False,eval_metric="auc",random_state=s,n_jobs=-1,verbosity=0)
        m.fit(X_tr[tr_idx],y[tr_idx])
        oof[va_idx]=m.predict_proba(X_tr[va_idx])[:,1]
        tp+=m.predict_proba(X_te)[:,1]/N_FOLDS
    xgb_oofs.append(oof); xgb_tps.append(tp)
xgb_oof=np.mean(xgb_oofs,axis=0); xgb_tp=np.mean(xgb_tps,axis=0)
print(f"XGB 3-seed avg: {roc_auc_score(y,xgb_oof):.5f}")

# LGB 3-seed
lgb_oofs=[]; lgb_tps=[]
for s in xgb_seeds:
    oof=np.zeros(n_train); tp=np.zeros(len(X_te))
    for tr_idx,va_idx in cv.split(X_tr,y):
        m=lgb.LGBMClassifier(n_estimators=800,max_depth=5,learning_rate=0.03,num_leaves=31,
         subsample=0.75,colsample_bytree=0.65,min_child_samples=20,
         reg_alpha=0.3,reg_lambda=2.0,random_state=s,n_jobs=-1,verbose=-1)
        m.fit(X_tr[tr_idx],y[tr_idx])
        oof[va_idx]=m.predict_proba(X_tr[va_idx])[:,1]
        tp+=m.predict_proba(X_te)[:,1]/N_FOLDS
    lgb_oofs.append(oof); lgb_tps.append(tp)
lgb_oof=np.mean(lgb_oofs,axis=0); lgb_tp=np.mean(lgb_tps,axis=0)
print(f"LGB 3-seed avg: {roc_auc_score(y,lgb_oof):.5f}")

# Blend
base_oofs=[cb_oof,xgb_oof,lgb_oof]; base_tests=[cb_tp,xgb_tp,lgb_tp]
aucs=np.array([roc_auc_score(y,o) for o in base_oofs])
w=aucs/aucs.sum()
bl_oof =sum(w[i]*base_oofs[i] for i in range(3))
bl_test=sum(w[i]*base_tests[i] for i in range(3))
print(f"Blend: {roc_auc_score(y,bl_oof):.5f}")

# LR Stack
meta_tr=np.column_stack(base_oofs); meta_te=np.column_stack(base_tests)
cv3=StratifiedKFold(n_splits=3,shuffle=True,random_state=RNG)
lr_oof=np.zeros(n_train); lr_te=np.zeros(len(X_te))
for tr_idx,va_idx in cv3.split(meta_tr,y):
    m=LogisticRegression(C=0.05,max_iter=1000,random_state=RNG)
    m.fit(meta_tr[tr_idx],y[tr_idx])
    lr_oof[va_idx]=m.predict_proba(meta_tr[va_idx])[:,1]
    lr_te+=m.predict_proba(meta_te)[:,1]/3
print(f"LR stack: {roc_auc_score(y,lr_oof):.5f}")

# Pseudo-labeling
THRESH=0.87
pmask=(bl_test>THRESH)|(bl_test<(1-THRESH))
if pmask.sum()>20:
    plabels=(bl_test[pmask]>0.5).astype(int)
    cb_aug=pd.concat([cb_tr,cb_te[pmask].reset_index(drop=True)],ignore_index=True)
    y_aug=np.concatenate([y,plabels]); aug_idx=np.arange(n_train,len(y_aug))
    p_oofs=[]; p_tps=[]
    for s in cb_seeds:
        p_oof=np.zeros(n_train); p_tp=np.zeros(len(X_te))
        for tr_idx,va_idx in cv.split(np.arange(n_train),y):
            full_tr=np.concatenate([tr_idx,aug_idx])
            m=CatBoostClassifier(**configs[0],cat_features=cat_idx,eval_metric="AUC",random_seed=s,verbose=0)
            m.fit(cb_aug.iloc[full_tr],y_aug[full_tr])
            p_oof[va_idx]=m.predict_proba(cb_tr.iloc[va_idx])[:,1]
            p_tp+=m.predict_proba(cb_te)[:,1]/N_FOLDS
        p_oofs.append(p_oof); p_tps.append(p_tp)
    pseudo_oof=np.mean(p_oofs,axis=0); pseudo_tp=np.mean(p_tps,axis=0)
    p_auc=roc_auc_score(y,pseudo_oof)
    print(f"Pseudo: {p_auc:.5f} ({pmask.sum()} samples)")
    if p_auc>roc_auc_score(y,bl_oof):
        bl_oof=0.5*pseudo_oof+0.5*bl_oof; bl_test=0.5*pseudo_tp+0.5*bl_test

# Final
all_f=[(bl_oof,bl_test),(lr_oof,lr_te),(cb_oof,cb_tp)]
fa=np.array([roc_auc_score(y,o) for o,_ in all_f]); fw=fa/fa.sum()
mega_oof =sum(fw[i]*all_f[i][0] for i in range(3))
mega_test=sum(fw[i]*all_f[i][1] for i in range(3))
print(f"\n🏆 FINAL OOF AUC: {roc_auc_score(y,mega_oof):.5f}")

sub=pd.DataFrame({"Id":test["Id"],"Drafted":mega_test})
sub.to_csv("/workspace/submission_final.csv",index=False)
print(f"✓ submission_final.csv — {len(sub)} rows")