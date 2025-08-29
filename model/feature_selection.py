# 特徵選擇
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from chart_feature_extractor import getXYdata
import warnings

warnings.filterwarnings("ignore")

X, Y, feature_name, all_song_name = getXYdata(True)


def forward_feature_selection_repeat_kfold(
    X, Y, feature_names, n_splits=5, repeats=3, max_features=None, tol=1e-3
):
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

            r2_repeats = []

            for repeat in range(repeats):
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
                r2_scores = []

                for train_idx, test_idx in kf.split(X_subset):
                    X_train, X_test = X_subset[train_idx], X_subset[test_idx]
                    y_train, y_test = Y[train_idx], Y[test_idx]

                    # model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model = XGBRegressor(
                        n_estimators=100, learning_rate=0.1, random_state=42
                    )
                    model.fit(X_train, y_train)

                    # model = TabNetRegressor(optimizer_params=dict(lr=3e-2), verbose=0)
                    # y_train = np.reshape(y_train, (-1, 1))
                    # y_test = np.reshape(y_test, (-1, 1))
                    # model.fit(
                    #     X_train, y_train,
                    #     eval_set=[(X_test, y_test)],
                    #     eval_metric=['mse'],
                    #     max_epochs=3000,
                    #     patience=100,
                    #     batch_size=1024,
                    #     virtual_batch_size=128,
                    #     num_workers=0,
                    #     augmentations=augmentations,
                    #     drop_last=False
                    # )

                    y_pred = model.predict(X_test)
                    r2_scores.append(r2_score(y_test, y_pred))

                r2_repeats.append(np.mean(r2_scores))

            avg_r2 = np.mean(r2_repeats)

            if avg_r2 > best_feat_r2 + tol:
                best_feat_r2 = avg_r2
                best_feat = f

        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
            best_r2 = best_feat_r2
            print(f"加入特徵 {best_feat}，平均 R² 提升到 {best_r2:.4f}")
            if max_features and len(selected) >= max_features:
                break
        else:
            break

    return selected, best_r2


selected_features, best_r2 = forward_feature_selection_repeat_kfold(
    X,
    Y,
    feature_name,
    n_splits=10,
    repeats=5,  # 每個特徵做5次 KFold，平均 R²
    max_features=20,  # 最多挑 10 個特徵
    tol=1e-3,  # 最小提升幅度
)

print("選擇的特徵：", selected_features)
print("最佳平均 R²：", best_r2)

warnings.resetwarnings()
