# KFold 測試
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from chart_feature_extractor import getXYdata

X, Y, feature_name, all_song_name = getXYdata(True)

# 設定折數
kf = KFold(n_splits=10, shuffle=True, random_state=42)

candidate = [
    "burst_p90",
    "page_space_p90_score",
    "page_space_third_score",
    "burst_song_avg",
    "complex_beat_count",
    "Drag-child",
    "burst_LR_low_max",
    "double_count",
    "SONG_LENGTH",
    "burst_endurance_8",
    "CDrag-head",
    "MAIN_BPM",
    "burst_fifth",
]
feat2idx = {f: i for i, f in enumerate(feature_name)}
idxs = [feat2idx[feat] for feat in candidate]
X_subset = X[:, idxs]
X_subset = X[:]

r2_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

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

    # model = getVoteModel()
    # model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test = y_test.reshape(-1)
    y_pred = y_pred.reshape(-1)

    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)


# 顯示總結
print("平均 R² 分數：", round(np.mean(r2_scores), 4))
