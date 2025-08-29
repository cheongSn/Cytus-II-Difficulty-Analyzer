# 特徵選擇
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from chart_feature_extractor import getXYdata
from kfold_test import doKFoldTest
import warnings


def feature_selection(X, Y, model, feature_names, tol=1e-3):
    feat2idx = {f: i for i, f in enumerate(feature_names)}
    selected = []
    remaining = set(feature_names)
    best_r2 = -np.inf

    while remaining:
        best_feat = None
        best_feat_r2 = best_r2

        for f in sorted(remaining):  # 排序保證確定性
            candidate = selected + [f]
            idxs = [feat2idx[feat] for feat in candidate]
            X_subset = X[:, idxs]

            r2 = doKFoldTest(X_subset, Y, model, 10)

            if r2 > best_feat_r2 + tol:
                best_feat_r2 = r2
                best_feat = f

        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
            best_r2 = best_feat_r2
            print(f"R² {best_r2:.4f}, add {best_feat}")
        else:
            break

    return selected, best_r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    X, Y, feature_name, all_song_name = getXYdata(True)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    selected_features, best_r2 = feature_selection(
        X,
        Y,
        model,
        feature_name,
        tol=1e-3,  # 最小提升幅度
    )

    print("選擇的特徵：", selected_features)
    print("最佳平均 R²：", best_r2)

    warnings.resetwarnings()
