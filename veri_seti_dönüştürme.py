from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

def cross_val_with_scaling(X, y, n_splits, metric):
    scalers = {
        "Normal": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "Log Dönüşümü": None,
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    roc_auc_scores = {}

    for scale_method, scaler in scalers.items():
        scale_method_scores = []

        for train_index, val_index in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            if scale_method == "Log Dönüşümü":
                min_value = X_train_fold.min().min()
                X_train_fold = np.log1p(X_train_fold - min_value)
                X_val_fold = np.log1p(X_val_fold - min_value)
            elif scaler:
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_val_fold = scaler.transform(X_val_fold)

            lgbm_model = LGBMClassifier(random_state=42, **param_lgbm)
            lgbm_model.fit(X_train_fold, y_train_fold)
            y_pred_prob_lgbm = lgbm_model.predict_proba(X_val_fold)[:, 1]
            roc_auc = roc_auc_score(y_val_fold, y_pred_prob_lgbm)
            scale_method_scores.append(roc_auc)

        roc_auc_scores[scale_method] = scale_method_scores

    for scale_method, scores in roc_auc_scores.items():
        mean_score = np.mean(scores)
        print(f"{scale_method} Mean {metric} Score: {mean_score:.12f}")

# Kullanım örneği:
n_splits = 4
metric = "ROC-AUC"
cross_val_with_scaling(X, y, n_splits, metric)
