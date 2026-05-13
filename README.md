# NFL Draft Prediction - Compétition GCI World 2026

## 📊 Résultats

| Version | OOF AUC | Public Score |
|---------|--------|--------------|
| v13 (final) | 0.829 | - |
| v12 | 0.830 | 0.844 |
| v9 | 0.829 | 0.833 |

## 🎯Objectif
- Score cible: ~0.87-0.88 (plafond du dataset)
- Score actuel: ~0.84

## 📁 Fichiers

### Code
- `nfl_v13.py` - Version finale avec 20 CatBoost seeds

### Data
- `train (3).csv` - Données d'entraînement (2781 lignes)
- `test (5).csv` - Données de test (696 lignes)

### Output
- `submission_final.csv` - Prédictions finales

## 🔧 Modèle

### Feature Engineering
- **Missingness**: Signaux pour chaque test non complété
- **Position percentiles**: Rang relatif par position
- **Target encoding**: K-fold avec smoothing
- **Interactions**: Position×Athleticism, School×Athleticism

### Architecture
```
20 CatBoost (5 configs × 4 seeds)
4 XGBoost seeds
4 LightGBM seeds
↓ 
Blend (AUC-weighted)
↓ 
LR Stacking
↓ 
Pseudo-labeling (seuil 0.88)
↓ 
Final Ensemble
```

### Hyperparamètres
- CV: 5-fold Stratified
- CatBoost: depth 4-7, lr 0.03-0.06
- XGB/LGB: depth 4-5, lr 0.025

## 🚀 Utilisation

```bash
# Entraîner le modèle
python nfl_v13.py

# Submission générée automatiquement
# Fichier: submission_final.csv
```

## 📥 Download

```bash
https://raw.githubusercontent.com/Souraka229/omnicampus/nfl-prediction-submission/submission_final.csv
```

## 🔑 Signaux les plus importants

1. **Missingness** - Joueurs sans tous les tests
2. **Position+Year percentiles** - Rang dans l'année/position
3. **School rate** - Taux de draft par école
4. **Athleticism** - Score composite desperateurs

## 📝 Notes

- Le dataset a un plafond de ~0.87-0.88 AUC
- Position seule = 0.59 AUC (baseline)
- Target encoding strict (K-fold) pour éviter leakage
- Pseudo-labeling aide à pousse un peu plus haut

---
**Auteur**: OpenHands AI
**Date**: Mai 2026