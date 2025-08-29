# 訓練
# from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from chart_feature_extractor import getXYdata
from utils import pad_display
from feature_selection import feature_selection
import numpy as np
import pickle

IS_DO_FEATURE_SELECTION = True
IS_DO_FEATURE_SELECTION = False
X, Y, feature_name, all_song_name = getXYdata(is_use_cache=True)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

if IS_DO_FEATURE_SELECTION:
    selected_features, best_r2 = feature_selection(
        X,
        Y,
        model,
        feature_name,
        tol=1e-3,  # 最小提升幅度
    )
    feat2idx = {f: i for i, f in enumerate(feature_name)}
    idxs = [feat2idx[feat] for feat in selected_features]
    with open("candidate_feature_idx_alpha.pkl", "wb") as f:
        pickle.dump(idxs, f)
    X = X[:, idxs]
else:
    with open("candidate_feature_idx_alpha.pkl", "wb") as f:
        pickle.dump(list(range(len(feature_name))), f)

# 1. 分割訓練與測試集
indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, Y, indices, test_size=0.1, random_state=75
)

print(f"Train size:{len(X_train)}")
print(f"Test size:{len(X_test)}")


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_test = y_test.reshape(-1)
y_pred = y_pred.reshape(-1)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

data = []
for i, idx in enumerate(test_idx):
    song_name = all_song_name[idx]
    data.append(
        (
            idx,
            song_name,
            y_test[i],
            round(y_pred[i], 2),
            round(y_pred[i] - y_test[i], 2),
        )
    )

SORT_BY = 4
# SORT_BY = 3

data = sorted(data, key=lambda x: (x[SORT_BY]))
print()
print(f"ID\t{pad_display('曲名', 32)}\t等級\t預測\t誤差")
for idx, s, t, p, diff in data:
    print(f"{idx}\t{pad_display(s[:30], 35)}\t{t}\t{p:.2f}\t{diff}")

# y_pred = np.round(y_pred, 2)
# print("\t".join(map(str, np.reshape(y_pred, (-1,)))))
# print("\t".join(map(str, np.reshape(y_test, (-1,)))))

print()
print("TEST Mean Squared Error:", mse)
print("TEST R² Score:", r2)
print("===================================================================")

if False:
    y_pred = model.predict(X_train)
    y_train = y_train.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    data = []
    for i, idx in enumerate(train_idx):
        song_name = all_song_name[idx]
        data.append(
            (
                idx,
                song_name,
                y_train[i],
                round(y_pred[i], 2),
                round(y_pred[i] - y_train[i], 2),
            )
        )

    data = sorted(data, key=lambda x: (x[5]))
    print()
    print(f"ID\t{pad_display('曲名', 32)}\t等級\t預測\t誤差")
    for idx, s, t, p, diff in data:
        print(f"{idx}\t{pad_display(s[:30], 35)}\t{t}\t{p:.2f}\t{diff}")

    # y_pred = np.round(y_pred, 2)
    # print("\t".join(map(str, np.reshape(y_pred, (-1,)))))
    # print("\t".join(map(str, np.reshape(y_test, (-1,)))))

    print()
    print("TRAIN Mean Squared Error:", mse)
    print("TRAIN R² Score:", r2)


model.save_model("CytusIIDifficultyAnalyzer_alpha.json")
